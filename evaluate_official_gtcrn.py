"""
Evaluate the OFFICIAL author's pretrained GTCRN checkpoint
(model_trained_on_vctk.tar) on the real VoiceBank+DEMAND test set, using
the exact same PESQ/STOI methodology as evaluate_qgtcrn.py, so the result
is directly comparable to every Q-GTCRN number we've measured.

This exists to verify: does the paper's claimed PESQ 2.87 actually
reproduce when we run THEIR OWN checkpoint through OUR evaluation
pipeline? This separates "did our reimplementation lose quality" from
"is there a methodology difference between us and the paper."
"""

import os
import argparse
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi

from dataset import VoiceBankDEMAND, stft, WIN_LEN, HOP_LEN, N_FFT, _WINDOW
from gtcrn_official import GTCRN


def istft(spec):
    real, imag = spec[..., 0], spec[..., 1]
    return torch.istft(torch.complex(real, imag), N_FFT, HOP_LEN, WIN_LEN,
                       _WINDOW.to(real.device))


def enhance_torch(model, noisy_wav, device):
    """Official GTCRN takes just (spec) -- single argument, not (feat, spec)
    like our QGTCRN. Different forward signature, handled here."""
    with torch.no_grad():
        spec = stft(noisy_wav.to(device)).unsqueeze(0)   # (1, F, T, 2)
        out = model(spec).squeeze(0)
        return istft(out).cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data',        default='VoiceBank-DEMAND')
    p.add_argument('--checkpoint',  default='checkpoints/model_trained_on_vctk.tar')
    p.add_argument('--results-dir', default='results_gtcrn_official')
    p.add_argument('--save-wavs',   type=int, default=10)
    args = p.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    wav_dir = os.path.join(args.results_dir, 'enhanced_wavs')
    if args.save_wavs > 0:
        os.makedirs(wav_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GTCRN().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Official checkpoints may be a raw state_dict or wrapped in {'model': ...}
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Backend: torch | Device: {device} | checkpoint: {args.checkpoint}")

    test_set = VoiceBankDEMAND(args.data, 'test.scp',
                               'noisy_testset_wav', 'clean_testset_wav')

    pesq_scores, stoi_scores = [], []

    for i, fname in enumerate(tqdm(test_set.files, desc='Evaluating (official GTCRN)')):
        noisy_wav, _ = sf.read(os.path.join(test_set.noisy_dir, fname + '.wav'), dtype='float32')
        clean_wav, _ = sf.read(os.path.join(test_set.clean_dir, fname + '.wav'), dtype='float32')
        noisy_t = torch.from_numpy(noisy_wav)

        enhanced_wav = enhance_torch(model, noisy_t, device).numpy()

        n = len(clean_wav)
        if len(enhanced_wav) > n:
            enhanced_wav = enhanced_wav[:n]
        else:
            enhanced_wav = np.pad(enhanced_wav, (0, n - len(enhanced_wav)))

        pesq_scores.append(pesq(16000, clean_wav, enhanced_wav, 'wb'))
        stoi_scores.append(stoi(clean_wav, enhanced_wav, 16000, extended=False))

        if i < args.save_wavs:
            sf.write(os.path.join(wav_dir, f'{fname}_enhanced.wav'), enhanced_wav, 16000)

    mean_pesq = np.mean(pesq_scores)
    mean_stoi = np.mean(stoi_scores)

    print(f"\nOfficial GTCRN checkpoint results ({len(test_set.files)} utterances):")
    print(f"  PESQ : {mean_pesq:.4f}  (paper claims 2.87)")
    print(f"  STOI : {mean_stoi:.4f}  (paper claims 0.940)")

    out_name = "eval_results_official_gtcrn.txt"
    with open(os.path.join(args.results_dir, out_name), 'w') as f:
        f.write(f"Checkpoint : {args.checkpoint}\n")
        f.write(f"Test files : {len(test_set.files)}\n")
        f.write(f"PESQ       : {mean_pesq:.4f}\n")
        f.write(f"STOI       : {mean_stoi:.4f}\n")

    print(f"\nResults saved to {args.results_dir}/{out_name}")


if __name__ == '__main__':
    main()