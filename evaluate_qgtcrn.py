import os
import argparse
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi

from dataset import VoiceBankDEMAND, stft, WIN_LEN, HOP_LEN, N_FFT, _WINDOW
from q_gtcrn import QGTCRN         


def istft(spec):
    real, imag = spec[..., 0], spec[..., 1]
    return torch.istft(torch.complex(real, imag), N_FFT, HOP_LEN, WIN_LEN,
                       _WINDOW.to(real.device))


def make_inputs(noisy_wav):
    spec = stft(noisy_wav)                                   # (F, T, 2)
    real, imag = spec[..., 0], spec[..., 1]
    mag  = torch.sqrt(real**2 + imag**2).clamp(min=1e-8)
    feat  = torch.stack([mag, real, imag], dim=0).unsqueeze(0)   # (1, 3, F, T)
    nspec = torch.stack([real, imag], dim=-1).unsqueeze(0)       # (1, F, T, 2)
    return feat, nspec


def enhance_torch(model, noisy_wav, device):
    with torch.no_grad():
        feat, nspec = make_inputs(noisy_wav.to(device))
        out = model(feat, nspec).squeeze(0)
        return istft(out).cpu()


def enhance_onnx(session, noisy_wav):
    feat, nspec = make_inputs(noisy_wav)
    out = session.run(None, {
        'noisy_feat': feat.numpy(),
        'noisy_spec': nspec.numpy(),
    })[0]
    out = torch.from_numpy(out).squeeze(0)
    return istft(out).cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data',        default='VoiceBank-DEMAND')
    p.add_argument('--backend',     choices=['torch', 'onnx'], default='torch')
    p.add_argument('--checkpoint',  default='checkpoints_qgtcrn/best.tar')
    p.add_argument('--onnx-model',  default='checkpoints_qgtcrn/model.onnx')
    p.add_argument('--results-dir', default='results_qgtcrn')
    p.add_argument('--save-wavs',   type=int, default=10,
                   help='number of enhanced test wavs to save (0 to disable)')
    args = p.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    wav_dir = os.path.join(args.results_dir, 'enhanced_wavs')
    if args.save_wavs > 0:
        os.makedirs(wav_dir, exist_ok=True)

    if args.backend == 'torch':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = QGTCRN().to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        source = f"torch (epoch {ckpt['epoch']})"
        print(f"Backend: torch | Device: {device} | epoch {ckpt['epoch']}")
    else:
        import onnxruntime as ort
        session = ort.InferenceSession(args.onnx_model, providers=['CPUExecutionProvider'])
        source = f"onnx ({args.onnx_model})"
        print(f"Backend: onnx | {args.onnx_model}")

    test_set = VoiceBankDEMAND(args.data, 'test.scp',
                               'noisy_testset_wav', 'clean_testset_wav')

    pesq_scores, stoi_scores = [], []

    for i, fname in enumerate(tqdm(test_set.files, desc='Evaluating')):
        noisy_wav, _ = sf.read(os.path.join(test_set.noisy_dir, fname + '.wav'), dtype='float32')
        clean_wav, _ = sf.read(os.path.join(test_set.clean_dir, fname + '.wav'), dtype='float32')
        noisy_t = torch.from_numpy(noisy_wav)

        if args.backend == 'torch':
            enhanced_wav = enhance_torch(model, noisy_t, device).numpy()
        else:
            enhanced_wav = enhance_onnx(session, noisy_t).numpy()

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

    print(f"\nTest set results ({len(test_set.files)} utterances):")
    print(f"  PESQ : {mean_pesq:.4f}  (gate: within 0.05 of GTCRN's 2.8201)")
    print(f"  STOI : {mean_stoi:.4f}  (gate: >= 0.938)")

    out_name = f"eval_results_{args.backend}.txt"
    with open(os.path.join(args.results_dir, out_name), 'w') as f:
        f.write(f"Source     : {source}\n")
        f.write(f"Test files : {len(test_set.files)}\n")
        f.write(f"PESQ       : {mean_pesq:.4f}\n")
        f.write(f"STOI       : {mean_stoi:.4f}\n")

    print(f"\nResults saved to {args.results_dir}/{out_name}")
    if args.save_wavs > 0:
        print(f"Enhanced wavs saved to {wav_dir}/")


if __name__ == '__main__':
    main()