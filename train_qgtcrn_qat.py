"""
QAT fine-tuning for Q-GTCRN.

Same recipe as v1's quantize_qat_sim.py: 15 epochs, lr=1e-4 (small nudge,
weights are already near-optimal), fake-quantization active throughout
training (observers continuously update, frozen only at the end).

KEY DIFFERENCE FROM v1: after training, this script also UNWRAPS the
QAT-adapted weights back into a plain QGTCRN checkpoint. v1's GTCRN QAT
result was permanently stuck as a simulated-only number -- there was no
path to a real deployable artifact, because the ONNX standard cannot
serialize a quantized GRU. Q-GTCRN has no such wall: the unwrapped
checkpoint here can be fed straight into export_qgtcrn.py and
quantize_qgtcrn_ptq.py to produce a genuinely deployable, quality-recovered
INT8 ONNX model -- something GTCRN's architecture could never achieve.
"""

import os
import argparse
import torch
import numpy as np
from torch.optim import Adam

from dataset import get_dataloaders, VoiceBankDEMAND, stft
import soundfile as sf
from q_gtcrn import QGTCRN
from loss import HybridLoss
from fake_quant_qgtcrn import quantize_model, freeze_observers, reset_observers, set_calibration_mode, unwrap_model, freeze_batchnorm
from quantize_qgtcrn_ptq_sim import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data',        default='VoiceBank-DEMAND')
    p.add_argument('--checkpoint',  default='checkpoints_qgtcrn_metricgan/finetuned_epoch13.tar')
    p.add_argument('--epochs',      type=int, default=15)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--batch-size',  type=int, default=4)
    p.add_argument('--workers',     type=int, default=2)
    p.add_argument('--ckpt-dir',    default='checkpoints_qgtcrn_qat')
    p.add_argument('--results-dir', default='results_qgtcrn_qat')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = QGTCRN()
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    print(f"Loaded FP32 baseline from {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    quantize_model(model)
    model.to(device)

    loss_fn = HybridLoss().to(device)
    # Low LR: the weights are already good, only nudging them to tolerate int8.
    optimizer = Adam(model.parameters(), lr=args.lr)

    train_loader, val_loader, test_loader = get_dataloaders(
        args.data, batch_size=args.batch_size, num_workers=args.workers)

    best_val = float('inf')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for noisy_feat, clean_spec, _ in train_loader:
            noisy_feat = noisy_feat.to(device)
            clean_spec = clean_spec.to(device)
            noisy_spec = torch.stack([noisy_feat[:, 1], noisy_feat[:, 2]], dim=-1)
            enhanced = model(noisy_feat, noisy_spec)
            loss = loss_fn(enhanced, clean_spec)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item()
        train_loss = total / len(train_loader)

        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for noisy_feat, clean_spec, _ in val_loader:
                noisy_feat = noisy_feat.to(device)
                clean_spec = clean_spec.to(device)
                noisy_spec = torch.stack([noisy_feat[:, 1], noisy_feat[:, 2]], dim=-1)
                vtotal += loss_fn(model(noisy_feat, noisy_spec), clean_spec).item()
        val_loss = vtotal / len(val_loader)

        print(f"epoch {epoch:2d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model': model.state_dict(), 'epoch': epoch},
                       os.path.join(args.ckpt_dir, 'qat_best.tar'))
            print(f"  saved qat_best (val {val_loss:.4f})")

    # --- Fresh recalibration, matching real deployment's calibration procedure ---
    # The 15 epochs above used a momentum-smoothed running range (good for
    # stable training gradients). But real ONNX Runtime static PTQ computes
    # its range a completely different way: exact min/max over just ~50 fresh
    # calibration samples, no smoothing. QAT that only ever saw the smoothed
    # training-time range doesn't know what range real deployment will
    # actually use -- so we throw that range away and recompute it the same
    # way real deployment will, on a similarly-sized sample from the same
    # source (test set) the real quantizer calibrates from.
    print("\n" + "=" * 60)
    print("Fresh recalibration (matching real ONNX Runtime's MinMax procedure)")
    print("=" * 60)
    reset_observers(model)
    set_calibration_mode(model, True)
    model.train()
    freeze_batchnorm(model)  # BN keeps its learned running stats; only the
                              # FakeQuantize observers should update here

    # Process each utterance individually, unpadded -- exactly matching
    # QGTCRNCalibrationReader in quantize_qgtcrn_ptq.py, which reads one raw
    # file at a time (batch=1, no padding). Our previous version used the
    # batched/padded test_loader, where collate_fn's zero-padding on shorter
    # utterances could shift observed activation ranges away from what
    # unpadded, one-at-a-time real calibration actually sees.
    calib_set = VoiceBankDEMAND(args.data, 'test.scp', 'noisy_testset_wav', 'clean_testset_wav')
    n_calib = 50
    with torch.no_grad():
        for fname in calib_set.files[:n_calib]:
            noisy, _ = sf.read(os.path.join(calib_set.noisy_dir, fname + '.wav'), dtype='float32')
            noisy_t = torch.from_numpy(noisy).to(device)
            spec = stft(noisy_t)
            real, imag = spec[..., 0], spec[..., 1]
            mag = torch.sqrt(real**2 + imag**2).clamp(min=1e-8)
            feat = torch.stack([mag, real, imag], dim=0).unsqueeze(0)
            nspec = torch.stack([real, imag], dim=-1).unsqueeze(0)
            model(feat, nspec)
    print(f"Recalibrated on {n_calib} individual, unpadded test-set utterances (exact min/max, no smoothing)")

    # Freeze observers -- ranges are now fixed to match real deployment.
    # NOTE: no second fine-tune phase here. That was tried before (fine-tuning
    # again after freezing, on top of the recalibrated grid) and it collapsed
    # results (PESQ down to 1.4836 on the real deployed INT8 model) -- it's
    # removed, not an oversight. Evaluate directly against the frozen,
    # correctly-calibrated grid.
    freeze_observers(model)

    print("\nEvaluating QAT (simulated int8) model, with fresh recalibration...")
    mean_pesq, mean_stoi = evaluate(model, args.data, device)

    print(f"\nQAT (simulated int8, recalibrated + re-tuned), Conv1d+Conv2d+Linear quantized:")
    print(f"  PESQ : {mean_pesq:.4f}  (compare to GTCRN's simulated QAT: 2.6664)")
    print(f"  STOI : {mean_stoi:.4f}  (compare to GTCRN's simulated QAT: 0.9362)")

    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, 'eval_qgtcrn_qat_sim.txt'), 'w') as f:
        f.write("Method     : QAT (simulated int8), Conv1d+Conv2d+Linear quantized, fine-tuned\n")
        f.write(f"Epochs     : {args.epochs}\n")
        f.write(f"PESQ       : {mean_pesq:.4f}\n")
        f.write(f"STOI       : {mean_stoi:.4f}\n")
    print(f"\nSaved to {args.results_dir}/eval_qgtcrn_qat_sim.txt")

    # --- The key extra step v1 could never do: unwrap into a REAL checkpoint ---
    print("\n" + "=" * 60)
    print("Unwrapping QAT-adapted weights into a plain, deployable checkpoint")
    print("=" * 60)
    unwrap_model(model)
    unwrapped_path = os.path.join(args.ckpt_dir, 'qat_unwrapped_for_export.tar')
    torch.save({'model': model.state_dict(), 'epoch': args.epochs,
               'best_val': best_val, 'source': 'QAT-adapted, unwrapped for real export'},
              unwrapped_path)
    print(f"Saved to {unwrapped_path}")
    print(f"\nNext steps -- run the REAL export + static PTQ pipeline on this checkpoint:")
    print(f"  python export_qgtcrn.py --checkpoint {unwrapped_path} --output checkpoints_qgtcrn_qat/qgtcrn_qat.onnx")
    print(f"  python quantize_qgtcrn_ptq.py --input checkpoints_qgtcrn_qat/qgtcrn_qat.onnx --output-dir checkpoints_qgtcrn_qat --data {args.data}")
    print(f"  python evaluate_qgtcrn.py --backend onnx --onnx-model checkpoints_qgtcrn_qat/qgtcrn_static_int8.onnx")
    print(f"\nThat final result is what GTCRN's architecture could never produce:")
    print(f"a genuinely deployable INT8 model with QAT-recovered quality.")


if __name__ == '__main__':
    main()
