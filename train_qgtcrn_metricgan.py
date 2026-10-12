"""
Local version of MetricGAN-style discriminator fine-tuning for Q-GTCRN.
Same logic as kaggle_metricgan_finetune_qgtcrn.py, but imports from your
local q_gtcrn.py / dataset.py / loss.py instead of embedding everything.

See kaggle_metricgan_finetune_qgtcrn.py's module docstring for the full
honest design notes (discriminator is our own design, not a verified
reproduction; anchor term + replay buffer are stabilization choices for
a short fine-tune run).

This is CPU-heavy (real PESQ computation every batch) -- expect this to
be slow locally. Prefer running the Kaggle version on GPU for the G/D
forward/backward at least, even though PESQ itself stays CPU-bound
either way.
"""

import os, time, csv, random, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from pesq import pesq
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import VoiceBankDEMAND, collate_fn, WIN_LEN, HOP_LEN, N_FFT, _WINDOW
from q_gtcrn import QGTCRN
from loss import HybridLoss


PESQ_MIN, PESQ_MAX = -0.5, 4.5


class Discriminator(nn.Module):
    """MetricGAN-style discriminator -- our own design, see docstring above."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, padding=2), nn.BatchNorm2d(16), nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 48, kernel_size=5, padding=2), nn.BatchNorm2d(48), nn.LeakyReLU(0.2),
        )
        self.blstm = nn.LSTM(48, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 16),  nn.LeakyReLU(0.2),
            nn.Linear(16, 1),   nn.Sigmoid()
        )

    def forward(self, enhanced_mag, clean_mag):
        x = torch.stack([enhanced_mag, clean_mag], dim=1)
        x = self.conv(x)
        x = x.mean(dim=2)
        x = x.transpose(1, 2)
        x, _ = self.blstm(x)
        x = x.mean(dim=1)
        return self.fc(x).squeeze(-1)


def compute_pesq_batch(enhanced_wavs, clean_wavs, sr=16000):
    scores = []
    for e, c in zip(enhanced_wavs, clean_wavs):
        try:
            score = pesq(sr, c, e, 'wb')
        except Exception:
            score = PESQ_MIN
        scores.append(score)
    return scores


def normalize_pesq(scores):
    t = torch.tensor(scores, dtype=torch.float32)
    return ((t - PESQ_MIN) / (PESQ_MAX - PESQ_MIN)).clamp(0.0, 1.0)


def get_finetune_loader(root, subset_size, batch_size=4, num_workers=0, seed=42):
    train_full = VoiceBankDEMAND(root, 'train.scp',
                                 'noisy_trainset_28spk_wav', 'clean_trainset_28spk_wav')
    rng = random.Random(seed)
    indices = list(range(len(train_full)))
    rng.shuffle(indices)
    subset = Subset(train_full, indices[:subset_size])
    return DataLoader(subset, batch_size=batch_size, shuffle=True,
                      collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)


def pad_stack(tensor_list):
    """Pad a list of (F, T) tensors to a common max T, then stack to (B, F, T).
    Needed because different batches (and therefore different replay-buffer
    entries) can have different T after collate_fn's per-batch padding."""
    T_max = max(t.shape[-1] for t in tensor_list)
    padded = [torch.nn.functional.pad(t, (0, T_max - t.shape[-1])) for t in tensor_list]
    return torch.stack(padded)


def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad = flag


def noisy_spec_from_feat(noisy_feat):
    return torch.stack([noisy_feat[:, 1], noisy_feat[:, 2]], dim=-1)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data',            default='VoiceBank-DEMAND')
    p.add_argument('--base-checkpoint', required=True,
                   help='Path to Run 6 (recipe-fixed) best.tar')
    p.add_argument('--epochs',          type=int, default=15)
    p.add_argument('--batch-size',      type=int, default=4)
    p.add_argument('--subset-size',     type=int, default=2000)
    p.add_argument('--g-lr',            type=float, default=1e-5)
    p.add_argument('--d-lr',            type=float, default=1e-4)
    p.add_argument('--anchor-weight',   type=float, default=0.1)
    p.add_argument('--replay-buffer',   type=int, default=200)
    p.add_argument('--workers',         type=int, default=0)
    p.add_argument('--seed',            type=int, default=42)
    p.add_argument('--ckpt-dir',        default='checkpoints_qgtcrn_metricgan')
    p.add_argument('--results-dir',     default='results_qgtcrn_metricgan')
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.ckpt_dir,    exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    loader = get_finetune_loader(args.data, args.subset_size, batch_size=args.batch_size,
                                 num_workers=args.workers, seed=args.seed)
    print(f"Fine-tuning on {args.subset_size} utterances, {len(loader)} batches/epoch")

    G = QGTCRN().to(device)
    ckpt = torch.load(args.base_checkpoint, map_location=device)
    G.load_state_dict(ckpt['model'])
    print(f"Loaded base checkpoint from epoch {ckpt['epoch']}, best_val {ckpt['best_val']:.4f}")

    D = Discriminator().to(device)
    anchor_loss_fn = HybridLoss().to(device)

    opt_G = AdamW(G.parameters(), lr=args.g_lr)
    opt_D = AdamW(D.parameters(), lr=args.d_lr)

    replay_buffer = deque(maxlen=args.replay_buffer)

    log_path = os.path.join(args.results_dir, 'finetune_log.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'd_loss', 'g_adv_loss', 'g_anchor_loss', 'mean_pesq'])

    history = {'epoch': [], 'pesq': []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        G.train(); D.train()
        d_losses, g_adv_losses, g_anchor_losses, batch_pesqs = [], [], [], []

        for noisy_feat, clean_spec, clean_wav in loader:
            noisy_feat = noisy_feat.to(device)
            clean_spec = clean_spec.to(device)

            enhanced_spec = G(noisy_feat, noisy_spec_from_feat(noisy_feat))

            with torch.no_grad():
                e_r, e_i = enhanced_spec[..., 0], enhanced_spec[..., 1]
                enhanced_wav = torch.istft(torch.complex(e_r, e_i), N_FFT, HOP_LEN, WIN_LEN,
                                           _WINDOW.to(device)).cpu().numpy()
            clean_wav_np = clean_wav.numpy()
            pesq_scores = compute_pesq_batch(enhanced_wav, clean_wav_np)
            pesq_norm = normalize_pesq(pesq_scores).to(device)
            batch_pesqs.extend(pesq_scores)

            enhanced_mag = torch.sqrt(e_r**2 + e_i**2 + 1e-12).detach()
            clean_mag = torch.sqrt(clean_spec[...,0]**2 + clean_spec[...,1]**2 + 1e-12)

            for i in range(enhanced_mag.shape[0]):
                replay_buffer.append((enhanced_mag[i].cpu(), clean_mag[i].cpu(), pesq_norm[i].cpu()))

            # --- Discriminator update ---
            set_requires_grad(D, True)
            opt_D.zero_grad()
            d_pred_enh = D(enhanced_mag, clean_mag)
            loss_d_enh = nn.functional.mse_loss(d_pred_enh, pesq_norm)
            d_pred_clean = D(clean_mag, clean_mag)
            loss_d_clean = nn.functional.mse_loss(d_pred_clean, torch.ones_like(d_pred_clean))

            loss_d_replay = torch.tensor(0.0, device=device)
            if len(replay_buffer) >= args.batch_size:
                sample = random.sample(replay_buffer, args.batch_size)
                r_enh = pad_stack([s[0] for s in sample]).to(device)
                r_clean = pad_stack([s[1] for s in sample]).to(device)
                r_pesq = torch.stack([s[2] for s in sample]).to(device)
                d_pred_replay = D(r_enh, r_clean)
                loss_d_replay = nn.functional.mse_loss(d_pred_replay, r_pesq)

            loss_D = loss_d_enh + loss_d_clean + loss_d_replay
            loss_D.backward()
            opt_D.step()

            # --- Generator update (D frozen) ---
            set_requires_grad(D, False)
            opt_G.zero_grad()
            enhanced_mag_g = torch.sqrt(e_r**2 + e_i**2 + 1e-12)
            d_pred_for_g = D(enhanced_mag_g, clean_mag)
            loss_g_adv = nn.functional.mse_loss(d_pred_for_g, torch.ones_like(d_pred_for_g))
            loss_g_anchor = anchor_loss_fn(enhanced_spec, clean_spec)
            loss_G = loss_g_adv + args.anchor_weight * loss_g_anchor
            loss_G.backward()
            opt_G.step()

            d_losses.append(loss_D.item())
            g_adv_losses.append(loss_g_adv.item())
            g_anchor_losses.append(loss_g_anchor.item())

        mean_d = np.mean(d_losses)
        mean_g_adv = np.mean(g_adv_losses)
        mean_g_anchor = np.mean(g_anchor_losses)
        mean_pesq = np.mean(batch_pesqs)

        print(f"epoch {epoch:2d} | D_loss {mean_d:.4f} | G_adv {mean_g_adv:.4f} | "
              f"G_anchor {mean_g_anchor:.4f} | mean_PESQ {mean_pesq:.4f} | {time.time()-t0:.0f}s")

        history['epoch'].append(epoch)
        history['pesq'].append(mean_pesq)

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f"{mean_d:.6f}", f"{mean_g_adv:.6f}",
                                   f"{mean_g_anchor:.6f}", f"{mean_pesq:.4f}"])

        torch.save({'epoch': epoch, 'model': G.state_dict(),
                   'discriminator': D.state_dict(),
                   'opt_G': opt_G.state_dict(), 'opt_D': opt_D.state_dict()},
                  os.path.join(args.ckpt_dir, f'finetuned_epoch{epoch}.tar'))
        torch.save({'epoch': epoch, 'model': G.state_dict()},
                  os.path.join(args.ckpt_dir, 'last.tar'))

        plt.figure(figsize=(7,4))
        plt.plot(history['epoch'], history['pesq'], label='mean train-batch PESQ')
        plt.xlabel('epoch'); plt.ylabel('PESQ'); plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(args.results_dir, 'pesq_curve.png'), dpi=120)
        plt.close()

    print(f"\nFine-tuning complete. Checkpoints saved to {args.ckpt_dir}")
    print("Run evaluate_qgtcrn.py on each finetuned_epochN.tar to find the best -- "
          "train-batch PESQ above is noisy/optimistic, real held-out eval is the true test.")


if __name__ == '__main__':
    main()