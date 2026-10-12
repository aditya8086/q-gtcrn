import os
import argparse
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi

from dataset import VoiceBankDEMAND, stft, get_dataloaders, WIN_LEN, HOP_LEN, N_FFT, _WINDOW
from q_gtcrn import QGTCRN
from fake_quant_qgtcrn import quantize_model, freeze_observers


def istft(spec):
    real, imag = spec[..., 0], spec[..., 1]
    return torch.istft(torch.complex(real, imag), N_FFT, HOP_LEN, WIN_LEN,
                       _WINDOW.to(real.device))


def make_inputs(noisy_wav):
    spec = stft(noisy_wav)
    real, imag = spec[..., 0], spec[..., 1]
    mag = torch.sqrt(real**2 + imag**2).clamp(min=1e-8)
    feat = torch.stack([mag, real, imag], dim=0).unsqueeze(0)
    nspec = torch.stack([real, imag], dim=-1).unsqueeze(0)
    return feat, nspec


def calibrate(model, loader, device, n_batches=50):
    # Run batches through the model so the fake-quant observers measure the
    # activation and weight ranges. No gradients, no weight updates.
    model.train()  # observers update only in train mode
    with torch.no_grad():
        for i, (noisy_feat, clean_spec, _) in enumerate(loader):
            if i >= n_batches:
                break
            noisy_feat = noisy_feat.to(device)
            noisy_spec = torch.stack([noisy_feat[:, 1], noisy_feat[:, 2]], dim=-1)
            model(noisy_feat, noisy_spec)
    freeze_observers(model)
    model.eval()


def evaluate(model, data_root, device):
    test_set = VoiceBankDEMAND(data_root, 'test.scp',
                               'noisy_testset_wav', 'clean_testset_wav')
    pesq_scores, stoi_scores = [], []
    with torch.no_grad():
        for fname in tqdm(test_set.files, desc='Evaluating'):
            noisy_wav, _ = sf.read(os.path.join(test_set.noisy_dir, fname + '.wav'), dtype='float32')
            clean_wav, _ = sf.read(os.path.join(test_set.clean_dir, fname + '.wav'), dtype='float32')

            feat, nspec = make_inputs(torch.from_numpy(noisy_wav).to(device))
            out = model(feat, nspec).squeeze(0)
            enhanced = istft(out).cpu().numpy()

            n = len(clean_wav)
            enhanced = enhanced[:n] if len(enhanced) > n else np.pad(enhanced, (0, n - len(enhanced)))
            pesq_scores.append(pesq(16000, clean_wav, enhanced, 'wb'))
            stoi_scores.append(stoi(clean_wav, enhanced, 16000, extended=False))
    return np.mean(pesq_scores), np.mean(stoi_scores)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data',        default='VoiceBank-DEMAND')
    p.add_argument('--checkpoint',  default='checkpoints_qgtcrn_metricgan/finetuned_epoch13.tar')
    p.add_argument('--results-dir', default='results_qgtcrn')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = QGTCRN()
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    quantize_model(model)
    model.to(device)

    train_loader, _, _ = get_dataloaders(args.data, batch_size=4, num_workers=0)
    print("Calibrating observers...")
    calibrate(model, train_loader, device)

    print("Evaluating simulated-int8 model (no fine-tuning)...")
    mean_pesq, mean_stoi = evaluate(model, args.data, device)

    print(f"\nPTQ (simulated int8), Conv1d+Conv2d+Linear quantized, no fine-tuning:")
    print(f"  PESQ : {mean_pesq:.4f}  (compare to GTCRN's simulated PTQ: 2.2474)")
    print(f"  STOI : {mean_stoi:.4f}  (compare to GTCRN's simulated PTQ: 0.9280)")

    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, 'eval_qgtcrn_ptq_sim.txt'), 'w') as f:
        f.write("Method     : PTQ (simulated int8), Conv1d+Conv2d+Linear quantized, no fine-tuning\n")
        f.write("Note       : Q-GTCRN has zero GRUs, so this simulated quantization covers 100%\n")
        f.write("             of the model -- unlike GTCRN's simulated PTQ, which also covered\n")
        f.write("             100% but only because it was necessarily simulated (real static\n")
        f.write("             PTQ crashed entirely on GTCRN's GRU/BatchNorm interaction).\n")
        f.write(f"PESQ       : {mean_pesq:.4f}\n")
        f.write(f"STOI       : {mean_stoi:.4f}\n")
    print(f"\nSaved to {args.results_dir}/eval_qgtcrn_ptq_sim.txt")


if __name__ == '__main__':
    main()
