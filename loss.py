import torch
import torch.nn as nn

WIN_LEN = 512
HOP_LEN = 256
N_FFT   = 512

# Compression exponents applied before MSE. Compressing magnitudes to a lower
# power emphasises perceptually relevant low-energy regions over loud peaks.
MAG_POWER  = 0.3
PHASE_POWER = 0.7

ALPHA = 0.01   # weight on the time-domain SI-SNR term
BETA  = 0.3    # splits spectrogram loss between magnitude and real/imag terms


def si_snr(pred, target, eps=1e-8):
    # Project target onto pred, then measure the ratio of signal to residual energy.
    proj = (pred * target).sum(-1, keepdim=True) * target \
           / (target.pow(2).sum(-1, keepdim=True) + eps)
    ratio = proj.norm(dim=-1) ** 2 / ((pred - proj).norm(dim=-1) ** 2 + eps)
    return -torch.log10(ratio + eps).mean()


class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('window', torch.hann_window(WIN_LEN).pow(0.5))
        self.mse = nn.MSELoss()

    def forward(self, pred_spec, true_spec):
        # spec: (B, F, T, 2)
        pred_r, pred_i = pred_spec[..., 0], pred_spec[..., 1]
        true_r, true_i = true_spec[..., 0], true_spec[..., 1]

        pred_mag = torch.sqrt(pred_r**2 + pred_i**2 + 1e-12)
        true_mag = torch.sqrt(true_r**2 + true_i**2 + 1e-12)

        mag_loss = self.mse(pred_mag**MAG_POWER, true_mag**MAG_POWER)
        real_loss = self.mse(pred_r / pred_mag**PHASE_POWER, true_r / true_mag**PHASE_POWER)
        imag_loss = self.mse(pred_i / pred_mag**PHASE_POWER, true_i / true_mag**PHASE_POWER)

        pred_wav = torch.istft(torch.complex(pred_r, pred_i), N_FFT, HOP_LEN, WIN_LEN, self.window)
        true_wav = torch.istft(torch.complex(true_r, true_i), N_FFT, HOP_LEN, WIN_LEN, self.window)
        snr_loss = si_snr(pred_wav, true_wav)

        return ALPHA * snr_loss + (1 - BETA) * mag_loss + BETA * (real_loss + imag_loss)


if __name__ == '__main__':
    loss_fn = HybridLoss()
    pred = torch.randn(2, 257, 63, 2)
    true = torch.randn(2, 257, 63, 2)
    print(loss_fn(pred, true))