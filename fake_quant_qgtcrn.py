"""
fake_quant_qgtcrn.py -- adapted from v1's fake_quant.py for Q-GTCRN.

Two differences from v1's version:
1. No QuantGRU wrapper -- Q-GTCRN has zero GRUs anywhere, so there's nothing
   to wrap. (v1 needed this because GTCRN's DPGRNN/TRA were GRU-based.)
2. Added QuantConv1d -- Q-GTCRN's TCN bottleneck uses Conv1d throughout
   (v1's GTCRN was Conv2d/GRU only, no Conv1d).
3. Added unwrap_model() -- v1 never needed this, because GTCRN's QAT result
   was PERMANENTLY simulated-only (the ONNX standard cannot serialize a
   quantized GRU, so there was no path to a real deployable artifact even
   after QAT). Q-GTCRN's static PTQ genuinely works, so after QAT fine-tunes
   the weights to tolerate quantization, we unwrap them back into a plain
   QGTCRN checkpoint and feed that into the REAL export + static PTQ
   pipeline -- producing an actually deployable, quality-recovered INT8
   model. This is a real capability GTCRN could never have.
"""

import torch
import torch.nn as nn


class FakeQuantize(nn.Module):
    """
    Simulates int8 quantization in the forward pass while keeping values in
    float32 and gradients flowing (straight-through estimator).

    A running min/max observer tracks the value range and derives the scale and
    zero-point of an affine int8 mapping r = scale * (q - zero_point), following
    Jacob et al. 2018. During observation the range is updated; once frozen, the
    stored range is used so training and evaluation quantize identically.

    Two range-tracking modes:
    - Training mode (default): momentum-smoothed EMA over many batches across
      many epochs. Good for stable gradients during fine-tuning.
    - Calibration mode (calibration_mode=True): exact running min/max across
      calibration batches, no smoothing -- this matches ONNX Runtime's real
      MinMax calibrator exactly (verified empirically: ORT tracks the true
      global min and global max across the calibration set, not an EMA).
      Use this for a final, small recalibration pass right before freezing,
      so the range QAT's evaluation uses matches what real deployment will
      actually compute -- rather than the smoothed 15-epoch training range,
      which was calibrated on different data with a different procedure.
    """
    def __init__(self, momentum=0.99, symmetric=False):
        super().__init__()
        self.momentum = momentum
        self.symmetric = symmetric
        self.qmin = -128
        self.qmax = 127
        self.register_buffer('running_min', torch.tensor(float('inf')))
        self.register_buffer('running_max', torch.tensor(float('-inf')))
        self.observing = True
        self.calibration_mode = False

    def _update_range(self, x):
        cur_min = x.min().detach()
        cur_max = x.max().detach()
        if torch.isinf(self.running_min):
            self.running_min.copy_(cur_min)
            self.running_max.copy_(cur_max)
        elif self.calibration_mode:
            # Exact global min/max across all calibration batches so far --
            # matches ORT's real MinMaxCalibrater, no smoothing.
            self.running_min.copy_(torch.min(self.running_min, cur_min))
            self.running_max.copy_(torch.max(self.running_max, cur_max))
        else:
            m = self.momentum
            self.running_min.copy_(m * self.running_min + (1 - m) * cur_min)
            self.running_max.copy_(m * self.running_max + (1 - m) * cur_max)

    def _scale_zeropoint(self):
        if self.symmetric:
            bound = torch.max(self.running_max.abs(), self.running_min.abs())
            scale = bound / 127
            zero_point = torch.zeros_like(scale)
        else:
            scale = (self.running_max - self.running_min) / (self.qmax - self.qmin)
            zero_point = self.qmin - torch.round(self.running_min / scale)
        scale = scale.clamp(min=1e-8)
        return scale, zero_point

    def forward(self, x):
        if self.observing and self.training:
            self._update_range(x)
        if torch.isinf(self.running_min):
            return x
        scale, zero_point = self._scale_zeropoint()
        q = torch.round(x / scale + zero_point).clamp(self.qmin, self.qmax)
        dq = (q - zero_point) * scale
        return x + (dq - x).detach()

    def freeze(self):
        self.observing = False

    def reset_range(self):
        """Clear the accumulated range so a fresh calibration pass starts clean,
        not blended with whatever the training-time observer had accumulated."""
        self.running_min.fill_(float('inf'))
        self.running_max.fill_(float('-inf'))
        self.observing = True


class QuantConv1d(nn.Module):
    """Conv1d with fake-quantized weights and output activations.
    New vs v1 -- Q-GTCRN's TCN bottleneck is Conv1d throughout."""
    def __init__(self, conv):
        super().__init__()
        self.conv = conv
        self.w_fq = FakeQuantize(symmetric=True)
        self.a_fq = FakeQuantize(symmetric=False)

    def forward(self, x):
        w = self.w_fq(self.conv.weight)
        out = nn.functional.conv1d(
            x, w, self.conv.bias, self.conv.stride, self.conv.padding,
            self.conv.dilation, self.conv.groups)
        return self.a_fq(out)


class QuantConv2d(nn.Module):
    """Conv2d with fake-quantized weights and output activations. Unchanged from v1."""
    def __init__(self, conv):
        super().__init__()
        self.conv = conv
        self.w_fq = FakeQuantize(symmetric=True)
        self.a_fq = FakeQuantize(symmetric=False)

    def forward(self, x):
        w = self.w_fq(self.conv.weight)
        if isinstance(self.conv, nn.ConvTranspose2d):
            out = nn.functional.conv_transpose2d(
                x, w, self.conv.bias, self.conv.stride, self.conv.padding,
                self.conv.output_padding, self.conv.groups, self.conv.dilation)
        else:
            out = nn.functional.conv2d(
                x, w, self.conv.bias, self.conv.stride, self.conv.padding,
                self.conv.dilation, self.conv.groups)
        return self.a_fq(out)


class QuantLinear(nn.Module):
    """Linear with fake-quantized weights and output activations. Unchanged from v1."""
    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.w_fq = FakeQuantize(symmetric=True)
        self.a_fq = FakeQuantize(symmetric=False)

    def forward(self, x):
        w = self.w_fq(self.linear.weight)
        out = nn.functional.linear(x, w, self.linear.bias)
        return self.a_fq(out)


def _wrap_module(module):
    for name, child in module.named_children():
        if isinstance(child, (nn.Conv2d, nn.ConvTranspose2d)):
            setattr(module, name, QuantConv2d(child))
        elif isinstance(child, nn.Conv1d):
            setattr(module, name, QuantConv1d(child))
        elif isinstance(child, nn.Linear):
            setattr(module, name, QuantLinear(child))
        else:
            _wrap_module(child)


def quantize_model(model):
    """Recursively replace Conv1d/Conv2d/Linear layers with fake-quantized versions."""
    _wrap_module(model)
    return model


def freeze_observers(model):
    for m in model.modules():
        if isinstance(m, FakeQuantize):
            m.freeze()


def reset_observers(model):
    """Clear all accumulated ranges, so a fresh calibration pass starts clean
    rather than blended with the old training-time smoothed range."""
    for m in model.modules():
        if isinstance(m, FakeQuantize):
            m.reset_range()


def set_calibration_mode(model, flag=True):
    """Switch all observers between training mode (momentum-smoothed EMA)
    and calibration mode (exact min/max, matching ONNX Runtime's real
    MinMax calibrator)."""
    for m in model.modules():
        if isinstance(m, FakeQuantize):
            m.calibration_mode = flag


def freeze_batchnorm(model):
    """
    Put every BatchNorm layer into eval mode (uses its learned running
    stats, does not update them) without touching any other module's mode.

    Bug this fixes: FakeQuantize.forward() only updates its observer when
    `self.training` is True, so calling model.train() to wake the observers
    up for calibration also puts every BatchNorm layer into training mode.
    Each calibration forward pass (batch size 1, unpadded, 50 single
    utterances) was then also EMA-updating the BN running stats learned
    over 15 epochs of batched (batch=4), padded training -- the calibration
    step was silently corrupting BatchNorm statistics instead of only
    calibrating the fake-quant ranges it was meant to calibrate.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()


def _unwrap_module(module):
    for name, child in module.named_children():
        if isinstance(child, (QuantConv2d, QuantConv1d)):
            setattr(module, name, child.conv)
        elif isinstance(child, QuantLinear):
            setattr(module, name, child.linear)
        else:
            _unwrap_module(child)


def unwrap_model(model):
    """
    Reverses quantize_model(): extracts the real underlying Conv/Linear
    layers (with their QAT-adapted weights) back into a plain, wrapper-free
    module tree matching the original QGTCRN architecture's state_dict keys.

    This is the step v1's GTCRN pipeline never needed, because GTCRN's QAT
    result could never become a real deployable artifact (no QLinearGRU).
    For Q-GTCRN, this is what lets QAT-adapted weights flow into the real
    export_qgtcrn.py + quantize_qgtcrn_ptq.py pipeline afterward.
    """
    _unwrap_module(model)
    return model
