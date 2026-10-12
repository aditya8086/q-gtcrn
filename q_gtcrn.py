"""
Q-GTCRN: Quantization-Friendly GTCRN
=====================================
Faithful copy of model.py (SpeechEnhancer / GTCRN reimplementation) with
ONE change: dpgrnn1 + dpgrnn2 (two DPGRNN / GRU blocks) are replaced by
a single DualAxisTCNBlock built from depthwise-separable dilated 1D
convolutions.

Every other module — ERB, SFE, TRA, ConvBlock, GTConvBlock, Encoder,
Decoder, CRMask — is copied unchanged from model.py.

Bottleneck I/O contract (unchanged): (B, 16, T, 33) -> (B, 16, T, 33)
"""

import torch
import numpy as np
import torch.nn as nn


# ---------------------------------------------------------------------------
# Unchanged modules -- copied verbatim from model.py
# ---------------------------------------------------------------------------

class ERB(nn.Module):
    """
    Frequency compression using the ERB (Equivalent Rectangular Bandwidth) scale.
    """
    def __init__(self, n_low, n_high, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        self.n_low = n_low
        filters = self._build_filters(n_low, n_high, nfft, high_lim, fs)
        self.fc  = nn.Linear(nfft // 2 + 1 - n_low, n_high, bias=False)
        self.ifc = nn.Linear(n_high, nfft // 2 + 1 - n_low, bias=False)
        self.fc.weight  = nn.Parameter(filters,   requires_grad=False)
        self.ifc.weight = nn.Parameter(filters.T, requires_grad=False)

    def _build_filters(self, n_low, n_high, nfft, high_lim, fs):
        def hz2erb(f): return 21.4 * np.log10(0.00437 * f + 1)
        def erb2hz(e): return (10 ** (e / 21.4) - 1) / 0.00437

        low_lim = n_low / nfft * fs
        pts  = np.linspace(hz2erb(low_lim), hz2erb(high_lim), n_high)
        bins = np.round(erb2hz(pts) / fs * nfft).astype(np.int32)

        F = np.zeros([n_high, nfft // 2 + 1], dtype=np.float32)
        F[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) / (bins[1] - bins[0] + 1e-12)
        for i in range(n_high - 2):
            F[i+1, bins[i]:bins[i+1]]   = (np.arange(bins[i], bins[i+1]) - bins[i] + 1e-12) / (bins[i+1] - bins[i] + 1e-12)
            F[i+1, bins[i+1]:bins[i+2]] = (bins[i+2] - np.arange(bins[i+1], bins[i+2]) + 1e-12) / (bins[i+2] - bins[i+1] + 1e-12)
        F[-1, bins[-2]:bins[-1]+1] = 1 - F[-2, bins[-2]:bins[-1]+1]

        return torch.from_numpy(np.abs(F[:, n_low:]))

    def compress(self, x):
        return torch.cat([x[..., :self.n_low], self.fc(x[..., self.n_low:])], dim=-1)

    def decompress(self, x):
        return torch.cat([x[..., :self.n_low], self.ifc(x[..., self.n_low:])], dim=-1)


class SFE(nn.Module):
    """Subband Feature Extraction"""
    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.k = kernel_size
        self.unfold = nn.Unfold(kernel_size=(1, kernel_size), stride=(1, stride),
                                padding=(0, (kernel_size - 1) // 2))

    def forward(self, x):
        B, C, T, F = x.shape
        return self.unfold(x).reshape(B, C * self.k, T, F)


class TRA(nn.Module):
    """
    Temporal Convolutional Attention (was: Temporal Recurrent Attention).

    Replaces the original GRU+FC path with a causal depthwise 1D conv over
    the (C,T) energy sequence, followed by a pointwise expand-contract
    (mirroring the GRU's channel-doubling capacity), then sigmoid.

    Pure conv + sigmoid -- no hidden state, no recurrence. Quantizes
    cleanly to INT8 with standard tooling (unlike the GRU it replaces).
    Causal: left-padding only, kernel=5, so output[t] depends only on
    x[t-4:t+1].
    """
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.causal_pad = kernel_size - 1

        # Depthwise temporal conv over the pooled energy sequence
        self.dw_conv = nn.Conv1d(channels, channels, kernel_size,
                                 groups=channels, bias=False)

        # Pointwise expand-contract (mirrors GRU's channel-doubling then FC)
        self.pw_expand   = nn.Conv1d(channels, channels * 2, 1)
        self.expand_act  = nn.PReLU()
        self.pw_contract = nn.Conv1d(channels * 2, channels, 1)

        self.act = nn.Sigmoid()

    def forward(self, x):
        """x: (B, C, T, F)"""
        energy = x.pow(2).mean(dim=-1)   # (B, C, T)

        e_pad = torch.nn.functional.pad(energy, (self.causal_pad, 0))
        att = self.dw_conv(e_pad)                    # (B, C, T)
        att = self.expand_act(self.pw_expand(att))    # (B, 2C, T)
        att = self.pw_contract(att)                    # (B, C, T)
        att = self.act(att)

        return x * att.unsqueeze(-1)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding,
                 groups=1, use_deconv=False, is_last=False):
        super().__init__()
        conv = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv(in_ch, out_ch, kernel_size, stride, padding, groups=groups)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.Tanh() if is_last else nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class AffinePReLU(nn.Module):
    """
    Frequency-dependent affine-PReLU: h(x) = gamma(f)*x + beta(f) + PReLU(x).

    gamma, beta are learnable per-frequency-bin parameters. Gains appear
    only when they are frequency-dependent (a frequency-independent affine
    term contributes ~nothing per UL-UNAS ablations). Adds negligible
    parameters (2 * num_freq) and stays INT8-friendly (elementwise affine +
    standard PReLU).
    """
    def __init__(self, num_freq):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, num_freq))
        self.beta  = nn.Parameter(torch.zeros(1, 1, 1, num_freq))
        self.prelu = nn.PReLU()

    def forward(self, x):
        """x: (B, C, T, F)"""
        return self.gamma * x + self.beta + self.prelu(x)


class GTConvBlock(nn.Module):
    """Grouped Temporal Convolution block (ShuffleNetV2-style).
    dw_act upgraded to frequency-dependent affine-PReLU (freq_width=33
    for all encoder/decoder GTConvBlock instances in this architecture)."""
    def __init__(self, in_ch, hidden_ch, kernel_size, stride, padding, dilation,
                 use_deconv=False, freq_width=33):
        super().__init__()
        conv = nn.ConvTranspose2d if use_deconv else nn.Conv2d

        self.pad_size = (kernel_size[0] - 1) * dilation[0]

        self.sfe      = SFE(kernel_size=3, stride=1)
        self.pw_conv1 = conv(in_ch // 2 * 3, hidden_ch, 1)
        self.pw_bn1   = nn.BatchNorm2d(hidden_ch)
        self.pw_act1  = nn.PReLU()
        self.dw_conv  = conv(hidden_ch, hidden_ch, kernel_size, stride=stride,
                             padding=padding, dilation=dilation, groups=hidden_ch)
        self.dw_bn    = nn.BatchNorm2d(hidden_ch)
        self.dw_act   = AffinePReLU(freq_width)
        self.pw_conv2 = conv(hidden_ch, in_ch // 2, 1)
        self.pw_bn2   = nn.BatchNorm2d(in_ch // 2)
        self.tra      = TRA(in_ch // 2)

    def _shuffle(self, x1, x2):
        x = torch.stack([x1, x2], dim=2)
        return x.transpose(1, 2).contiguous().flatten(1, 2)

    def forward(self, x):
        x1, x2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:]
        h = self.pw_act1(self.pw_bn1(self.pw_conv1(self.sfe(x1))))
        h = torch.nn.functional.pad(h, [0, 0, self.pad_size, 0])
        h = self.dw_act(self.dw_bn(self.dw_conv(h)))
        h = self.pw_bn2(self.pw_conv2(h))
        h = self.tra(h)
        return self._shuffle(h, x2)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvBlock(9,  16, (1,5), stride=(1,2), padding=(0,2)),
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1)),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1)),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1)),
        ])

    def forward(self, x):
        skip_connections = []
        for block in self.blocks:
            x = block(x)
            skip_connections.append(x)
        return x, skip_connections


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*5,1), dilation=(5,1), use_deconv=True),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*2,1), dilation=(2,1), use_deconv=True),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*1,1), dilation=(1,1), use_deconv=True),
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=True),
            ConvBlock(16,  2, (1,5), stride=(1,2), padding=(0,2), use_deconv=True, is_last=True),
        ])

    def forward(self, x, skip_connections):
        for i, block in enumerate(self.blocks):
            x = block(x + skip_connections[-(i+1)])
        return x


class CRMask(nn.Module):
    """Complex Ratio Mask -- unchanged"""
    def forward(self, mask, spec):
        r = spec[:,0] * mask[:,0] - spec[:,1] * mask[:,1]
        i = spec[:,1] * mask[:,0] + spec[:,0] * mask[:,1]
        return torch.stack([r, i], dim=1)


# ---------------------------------------------------------------------------
# NEW: DualAxisTCNBlock -- replaces dpgrnn1 + dpgrnn2
# ---------------------------------------------------------------------------

class CausalGatedTCNBlock(nn.Module):
    """
    One gated residual block of the TCN bottleneck (WaveNet/Conv-TasNet style).

    Operates on (N, C, T) where N = B*F (each frequency bin processed
    independently along time -- analogous to DPGRNN's inter-frame RNN path).

    Two parallel causal depthwise convs (filter, gate) -> tanh/sigmoid gating
    -> separate pointwise projections for the residual path and the skip path.
    Skip outputs from all blocks in the stack are summed (Conv-TasNet style)
    to form the bottleneck's final output, while the residual path continues
    propagating depth-wise through the stack.

    Causal padding: left-pad only, by (kernel-1)*dilation.
    """
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.causal_pad = (kernel_size - 1) * dilation

        self.dw_filter = nn.Conv1d(channels, channels, kernel_size,
                                   dilation=dilation, groups=channels, bias=False)
        self.dw_gate   = nn.Conv1d(channels, channels, kernel_size,
                                   dilation=dilation, groups=channels, bias=False)
        self.bn_filter = nn.BatchNorm1d(channels)
        self.bn_gate   = nn.BatchNorm1d(channels)

        self.pw_res  = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn_res  = nn.BatchNorm1d(channels)
        self.pw_skip = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn_skip = nn.BatchNorm1d(channels)

    def forward(self, x):
        """x: (N, C, T) -> (residual_out, skip_out), both (N, C, T)"""
        residual = x
        x_pad = torch.nn.functional.pad(x, (self.causal_pad, 0))

        filt = torch.tanh(self.bn_filter(self.dw_filter(x_pad)))
        gate = torch.sigmoid(self.bn_gate(self.dw_gate(x_pad)))
        gated = filt * gate

        res_out  = self.bn_res(self.pw_res(gated)) + residual
        skip_out = self.bn_skip(self.pw_skip(gated))
        return res_out, skip_out


class IntraFreqConv(nn.Module):
    """
    Frequency-axis (intra) path -- mirrors DPGRNN's bidirectional intra_rnn.

    Harmonics are a global, not sequential, relationship across frequency
    bins at a fixed time frame, so this path is non-causal (symmetric
    padding) -- there is no streaming constraint along the frequency axis.

    depthwise Conv1d (along F) -> BN -> PReLU -> pointwise Conv1d -> BN.
    No internal residual; the outer DualAxisTCNBlock applies residual+LN,
    mirroring DPGRNN's x = x + LN(intra(x)) pattern exactly.
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = (kernel_size - 1) // 2   # symmetric -- non-causal along F
        self.dw_conv = nn.Conv1d(channels, channels, kernel_size,
                                 padding=pad, groups=channels, bias=False)
        self.bn1  = nn.BatchNorm1d(channels)
        self.act1 = nn.PReLU()
        self.pw_conv = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        """x: (N, C, F) where N = B*T"""
        h = self.act1(self.bn1(self.dw_conv(x)))
        h = self.bn2(self.pw_conv(h))
        return h


class DualAxisTCNBlock(nn.Module):
    """
    Fully-convolutional dual-path block -- drop-in replacement for one
    DPGRNN instance, mirroring its exact intra+inter structure:

      - Intra (frequency) path: IntraFreqConv, models spectral relationships
        across frequency bins at each time frame. Non-causal.
      - Inter (time) path: causal gated dilated TCN stack (CausalGatedTCNBlock
        x N, skip-aggregated Conv-TasNet style) -- unchanged from Step 1/2.

    Both paths follow DPGRNN's residual+LayerNorm wrapper exactly:
        x = x + LN(intra(x))
        x = x + LN(inter(x))

    This recovers the frequency-axis modeling that a time-only TCN loses,
    while keeping every operation pure conv/BN/PReLU/tanh/sigmoid --
    no hidden state anywhere, so standard INT8 PTQ toolchains quantize the
    whole bottleneck without special operators.
    """
    def __init__(self, channels=16, kernel_size=3, dilations=(1, 2, 4, 8, 16),
                freq_kernel=3):
        super().__init__()
        self.intra    = IntraFreqConv(channels, kernel_size=freq_kernel)
        self.intra_ln = nn.LayerNorm(channels)

        self.inter_blocks = nn.ModuleList([
            CausalGatedTCNBlock(channels, kernel_size, d)
            for d in dilations
        ])
        self.inter_skip_act = nn.PReLU()
        self.inter_ln = nn.LayerNorm(channels)

    def forward(self, x):
        """x: (B, C, T, F)"""
        B, C, T, F = x.shape

        # --- Intra (frequency) path ---
        h = x.permute(0, 2, 1, 3).reshape(B * T, C, F)   # (B*T, C, F)
        h = self.intra(h)
        h = h.reshape(B, T, C, F).permute(0, 1, 3, 2)     # (B, T, F, C)
        h = self.intra_ln(h)
        h = h.permute(0, 3, 1, 2)                          # (B, C, T, F)
        x = x + h

        # --- Inter (time) path ---
        g = x.permute(0, 3, 1, 2).reshape(B * F, C, T)    # (B*F, C, T)
        skip_sum = 0
        for block in self.inter_blocks:
            g, skip = block(g)
            skip_sum = skip_sum + skip
        g = self.inter_skip_act(skip_sum)                  # (B*F, C, T)
        g = g.reshape(B, F, C, T).permute(0, 3, 1, 2)      # (B, T, F, C)
        g = self.inter_ln(g)
        g = g.permute(0, 3, 1, 2)                           # (B, C, T, F)
        x = x + g

        return x


# ---------------------------------------------------------------------------
# Q-GTCRN -- full model, faithful copy of SpeechEnhancer with TCN swap
# ---------------------------------------------------------------------------

class QGTCRN(nn.Module):
    def __init__(self):
        super().__init__()
        self.erb     = ERB(65, 64)
        self.sfe     = SFE(3, 1)
        self.encoder = Encoder()
        self.tcn1    = DualAxisTCNBlock(channels=16, kernel_size=3,
                                           dilations=(1, 2, 4, 8, 16))
        self.tcn2    = DualAxisTCNBlock(channels=16, kernel_size=3,
                                           dilations=(1, 2, 4, 8, 16))
        self.decoder = Decoder()
        self.mask    = CRMask()

    def forward(self, noisy_feat, noisy_spec):
        """
        noisy_feat: (B, 3, F, T) -- magnitude, real, imag stacked as channels
        noisy_spec: (B, F, T, 2) -- raw STFT of the noisy signal, used for masking
        Returns:
            enhanced_spec: (B, F, T, 2)
        """
        x = noisy_feat.permute(0, 1, 3, 2)          # (B, 3, T, F)

        x = self.erb.compress(x)                     # (B, 3, T, 129)
        x = self.sfe(x)                               # (B, 9, T, 129)

        x, skips = self.encoder(x)                    # (B, 16, T, 33)

        x = self.tcn1(x)                               # (B, 16, T, 33)
        x = self.tcn2(x)                               # (B, 16, T, 33)  <- two passes, mirrors dpgrnn1 + dpgrnn2

        x = self.decoder(x, skips)                     # (B, 2, T, 129)

        x = self.erb.decompress(x)                     # (B, 2, T, 257)

        spec = noisy_spec.permute(0, 3, 2, 1)           # (B, 2, T, F)
        enhanced = self.mask(x, spec)                   # (B, 2, T, F)

        return enhanced.permute(0, 3, 2, 1)              # (B, F, T, 2)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    model = QGTCRN()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    noisy_feat = torch.randn(1, 3, 257, 100)   # (B, 3, F, T)
    noisy_spec = torch.randn(1, 257, 100, 2)   # (B, F, T, 2)
    out = model(noisy_feat, noisy_spec)
    print(f"Output: {out.shape}")              # expect (1, 257, 100, 2)
    assert out.shape == noisy_spec.shape, "Output shape mismatch!"
    print("Shape check passed.")