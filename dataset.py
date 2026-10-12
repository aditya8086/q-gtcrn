import os
import torch
import soundfile as sf
from torch.utils.data import Dataset, DataLoader

WIN_LEN = 512
HOP_LEN = 256
N_FFT   = 512
_WINDOW = torch.hann_window(WIN_LEN).pow(0.5)


def stft(wav):
    spec = torch.stft(wav, N_FFT, HOP_LEN, WIN_LEN, _WINDOW.to(wav.device),
                      return_complex=True)
    return torch.view_as_real(spec)  # (F, T, 2)


def to_feat(spec):
    # Each time-frequency bin is a complex number with real and imaginary parts.
    # Magnitude captures overall energy; real and imag preserve phase information.
    # All three are stacked as separate channels for the model to learn from.
    real = spec[..., 0]
    imag = spec[..., 1]
    mag  = torch.sqrt(real**2 + imag**2).clamp(min=1e-8)
    return torch.stack([mag, real, imag], dim=0)  # (3, F, T)


class VoiceBankDEMAND(Dataset):
    def __init__(self, root, scp_file, noisy_dir, clean_dir, sr=16000):
        self.noisy_dir = os.path.join(root, noisy_dir)
        self.clean_dir = os.path.join(root, clean_dir)
        self.sr = sr

        with open(os.path.join(root, scp_file)) as f:
            self.files = [l.strip() for l in f if l.strip()]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        noisy, sr = sf.read(os.path.join(self.noisy_dir, fname + '.wav'), dtype='float32')
        clean, _  = sf.read(os.path.join(self.clean_dir, fname + '.wav'), dtype='float32')
        assert sr == self.sr, f"Expected {self.sr}Hz, got {sr}Hz: {fname}"

        noisy = torch.from_numpy(noisy)
        clean = torch.from_numpy(clean)

        # noisy_feat: 3-channel spectrogram input for the model
        # clean_spec: raw STFT of clean speech, used in spectrogram-domain loss
        # clean:      raw waveform, used in time-domain loss
        noisy_feat = to_feat(stft(noisy))  # (3, F, T)
        clean_spec = stft(clean)           # (F, T, 2)
        return noisy_feat, clean_spec, clean


def collate_fn(batch):
    # Utterances in a batch have different lengths; pad to the longest.
    noisy_feats, clean_specs, clean_wavs = zip(*batch)

    T_max = max(x.shape[-1] for x in noisy_feats)
    noisy_pad, clean_spec_pad = [], []
    for nf, cs in zip(noisy_feats, clean_specs):
        pad = T_max - nf.shape[-1]
        noisy_pad.append(torch.nn.functional.pad(nf, (0, pad)))
        clean_spec_pad.append(torch.nn.functional.pad(cs, (0, 0, 0, pad)))

    T_wav_max = max(w.shape[0] for w in clean_wavs)
    clean_wav_pad = [torch.nn.functional.pad(w, (0, T_wav_max - w.shape[0]))
                     for w in clean_wavs]

    return (torch.stack(noisy_pad),       # (B, 3, F, T)
            torch.stack(clean_spec_pad),  # (B, F, T, 2)
            torch.stack(clean_wav_pad))   # (B, samples)


def get_dataloaders(root, batch_size=4, num_workers=4, val_split=0.12):
    train_full = VoiceBankDEMAND(root, 'train.scp',
                                 'noisy_trainset_28spk_wav', 'clean_trainset_28spk_wav')
    test_set   = VoiceBankDEMAND(root, 'test.scp',
                                 'noisy_testset_wav', 'clean_testset_wav')

    n_val   = round(len(train_full) * val_split)
    n_train = len(train_full) - n_val
    train_set, val_set = torch.utils.data.random_split(
        train_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    loader_args = dict(batch_size=batch_size, collate_fn=collate_fn,
                       num_workers=num_workers, pin_memory=True)
    return (DataLoader(train_set, shuffle=True,  **loader_args),
            DataLoader(val_set,   shuffle=False, **loader_args),
            DataLoader(test_set,  shuffle=False, **loader_args))


if __name__ == '__main__':
    root = 'VoiceBank-DEMAND'
    train_loader, val_loader, test_loader = get_dataloaders(root, batch_size=2, num_workers=0)

    noisy_feat, clean_spec, clean_wav = next(iter(train_loader))
    print(f"noisy_feat : {noisy_feat.shape}")
    print(f"clean_spec : {clean_spec.shape}")
    print(f"clean_wav  : {clean_wav.shape}")
    print(f"train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")