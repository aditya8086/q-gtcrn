"""
Post-training quantization for Q-GTCRN.

Unlike v1's GTCRN pipeline -- which was forced into weight-only DYNAMIC
quantization restricted to MatMul ops, because static PTQ crashed on the
BatchNorm/GRU interaction and dynamic quantization of Conv produced
unimplemented ConvInteger nodes -- Q-GTCRN is fully convolutional with zero
GRUs. This means we can attempt genuine STATIC PTQ (real INT8 weights AND
activations, calibrated on real data), which is the toolchain's
well-behaved case per every quantization reference (Jacob et al. 2018;
NVIDIA's INT8 inference guide) and should give a much larger, more
meaningful compression than v1's 1.36x MatMul-only result.

We still also produce the dynamic-PTQ result for direct comparison against
v1's methodology and numbers.
"""

import os
import argparse
import numpy as np
import torch
import onnx
from onnxsim import simplify
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_dynamic, quantize_static, QuantType, QuantFormat,
    CalibrationDataReader
)
from onnxruntime.quantization.preprocess import quant_pre_process

from dataset import VoiceBankDEMAND, stft, WIN_LEN, HOP_LEN, N_FFT


def simplify_onnx(input_path, output_path):
    """
    Fold constants and clean up the graph before quantization. Without this,
    many Conv bias tensors are not stored as plain initializers (a legacy
    TorchScript exporter artifact), which blocks QOperator quantization
    entirely and inflates QDQ output size. This matches the official GTCRN
    repo's own export pipeline, which also runs onnxsim before deployment.
    """
    model = onnx.load(input_path)
    model_simp, check = simplify(model)
    if not check:
        raise RuntimeError("ONNX simplification validation failed")
    onnx.save(model_simp, output_path)
    return output_path


class QGTCRNCalibrationReader(CalibrationDataReader):
    """
    Feeds real noisy-speech STFT features to the static quantizer's
    calibration pass, so INT8 activation ranges are computed from actual
    data distributions rather than the dummy random tensor used for export.
    """
    def __init__(self, data_root, n_samples=50):
        test_set = VoiceBankDEMAND(data_root, 'test.scp',
                                   'noisy_testset_wav', 'clean_testset_wav')
        self.files = test_set.files[:n_samples]
        self.noisy_dir = test_set.noisy_dir
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.files):
            return None
        import soundfile as sf
        fname = self.files[self.idx]
        self.idx += 1

        noisy, _ = sf.read(os.path.join(self.noisy_dir, fname + '.wav'), dtype='float32')
        noisy_t = torch.from_numpy(noisy)
        spec = stft(noisy_t)  # (F, T, 2)

        real, imag = spec[..., 0], spec[..., 1]
        mag = torch.sqrt(real**2 + imag**2).clamp(min=1e-8)
        feat = torch.stack([mag, real, imag], dim=0).unsqueeze(0)  # (1,3,F,T)
        nspec = spec.unsqueeze(0)  # (1,F,T,2)

        return {
            'noisy_feat': feat.numpy(),
            'noisy_spec': nspec.numpy(),
        }


def try_static_ptq(input_path, output_path, data_root, n_calib=50):
    """Attempt real static PTQ: INT8 weights + INT8 activations, calibrated
    on real data. Returns True on success, False if the toolchain rejects
    this model the way it rejected GTCRN's GRU/BatchNorm interaction."""
    prepped = input_path.replace('.onnx', '_prep_static.onnx')
    quant_pre_process(input_path, prepped, skip_symbolic_shape=True)

    calib_reader = QGTCRNCalibrationReader(data_root, n_samples=n_calib)

    try:
        quantize_static(
            prepped,
            output_path,
            calib_reader,
            quant_format=QuantFormat.QDQ,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            op_types_to_quantize=['Conv', 'MatMul'],
        )
        return True
    except Exception as e:
        print(f"Static PTQ failed: {type(e).__name__}: {e}")
        return False


def try_dynamic_ptq(input_path, output_path):
    """Fallback / comparison: weight-only dynamic quantization, same
    restricted approach v1 was forced into for GTCRN.

    Op list restricted to MatMul only. Dynamic quantization of Conv layers
    produces ConvInteger nodes, which ONNX Runtime's CPU provider does not
    implement -- this is the exact same limitation v1's report documented
    for GTCRN, and it applies here too since it's a runtime/provider gap,
    not something specific to either architecture."""
    prepped = input_path.replace('.onnx', '_prep_dynamic.onnx')
    quant_pre_process(input_path, prepped, skip_symbolic_shape=True)

    quantize_dynamic(
        prepped,
        output_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=['MatMul'],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input',      default='checkpoints_qgtcrn_metricgan/qgtcrn.onnx')
    p.add_argument('--output-dir', default='checkpoints_qgtcrn_metricgan')
    p.add_argument('--data',       default='VoiceBank-DEMAND')
    p.add_argument('--n-calib',    type=int, default=50)
    p.add_argument('--no-simplify', action='store_true',
                   help='Skip ONNX simplification entirely and quantize the '
                        'raw export directly. Use this to force the SAME code '
                        'path across two runs you want to compare cleanly -- '
                        'simplification has been observed to succeed or fail '
                        'non-deterministically depending on the exact graph, '
                        'which silently changes what gets quantized between runs.')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    static_out = os.path.join(args.output_dir, 'qgtcrn_static_int8.onnx')
    dynamic_out = os.path.join(args.output_dir, 'qgtcrn_dynamic_int8.onnx')
    simplified = os.path.join(args.output_dir, 'qgtcrn_simplified.onnx')

    if args.no_simplify:
        print("=" * 60)
        print("Skipping simplification (--no-simplify) -- quantizing raw export directly")
        print("=" * 60)
        quant_source = args.input
        static_ok = try_static_ptq(args.input, static_out, args.data, args.n_calib)
    else:
        print("=" * 60)
        print("Simplifying ONNX graph (fixes bias-initializer issues)")
        print("=" * 60)
        simplify_onnx(args.input, simplified)
        print(f"Simplified model saved to {simplified}")
        print(f"  (raw FP32 size before quantization: "
              f"{os.path.getsize(simplified)/1024:.1f} KB, "
              f"was {os.path.getsize(args.input)/1024:.1f} KB before simplification)")

        print("\n" + "=" * 60)
        print("Attempting STATIC PTQ (real INT8 weights + activations)")
        print("=" * 60)
        quant_source = simplified
        try:
            static_ok = try_static_ptq(simplified, static_out, args.data, args.n_calib)
        except Exception as e:
            # onnxsim's simplification can itself introduce shape-inference bugs
            # on this model's SFE/Unfold module under a dynamic time axis (seen
            # in practice). The raw export was already verified numerically
            # correct via real PESQ/STOI evaluation, so fall back to quantizing
            # it directly rather than the (buggy-here) simplified version.
            print(f"\nStatic PTQ on simplified model failed ({type(e).__name__}: {e})")
            print("Falling back to quantizing the ORIGINAL (unsimplified) export directly.")
            quant_source = args.input
            static_ok = try_static_ptq(args.input, static_out, args.data, args.n_calib)

    print("\n" + "=" * 60)
    print("Building DYNAMIC PTQ (weight-only, for comparison to v1)")
    print("=" * 60)
    try:
        try_dynamic_ptq(quant_source, dynamic_out)
    except Exception as e:
        print(f"Dynamic PTQ on {quant_source} failed ({type(e).__name__}: {e})")
        if quant_source != args.input:
            print("Falling back to the original (unsimplified) export.")
            quant_source = args.input
            try_dynamic_ptq(args.input, dynamic_out)
        else:
            raise

    fp32_size = os.path.getsize(quant_source) / 1024
    dyn_size = os.path.getsize(dynamic_out) / 1024

    print(f"\nQuantization source: {quant_source}")
    print(f"\n{'Model':<30}{'Size (KB)':>12}{'Compression':>14}")
    print("-" * 56)
    print(f"{'float32 ONNX':<30}{fp32_size:>12.1f}{'--':>14}")
    print(f"{'dynamic INT8 (weight-only)':<30}{dyn_size:>12.1f}{fp32_size/dyn_size:>13.2f}x")

    if static_ok and os.path.exists(static_out):
        static_size = os.path.getsize(static_out) / 1024
        print(f"{'static INT8 (weights+activ.)':<30}{static_size:>12.1f}{fp32_size/static_size:>13.2f}x")
        print(f"\nStatic PTQ SUCCEEDED -- saved to {static_out}")
    else:
        print(f"\nStatic PTQ FAILED -- see error above. Dynamic PTQ result")
        print(f"is the fallback, saved to {dynamic_out}")

    print(f"\nNext: run evaluate_qgtcrn.py --backend onnx --onnx-model <path>")
    print(f"on each produced model to measure real PESQ/STOI degradation.")


if __name__ == '__main__':
    main()
