import argparse
import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort

from q_gtcrn import QGTCRN


def ensure_conv_biases(model, state_dict):
    """
    Several Conv1d layers in this architecture (CausalGatedTCNBlock's
    dw_filter/dw_gate/pw_res/pw_skip, IntraFreqConv's dw_conv/pw_conv) are
    defined with bias=False -- their bias term was meant to come entirely
    from the BatchNorm that originally followed them.

    The Kaggle QAT script's BatchNorm-folding step (fold_conv_bn) computes
    a real bias value for these layers from the folded BN and saves it in
    the checkpoint -- but a fresh QGTCRN() has no bias parameter on these
    modules to receive it. Without this, load_state_dict either errors
    (strict=True, "unexpected key") or -- if strict=False were used
    instead -- would silently DROP that bias correction, which is wrong,
    not just stricter.

    Fix: wherever the checkpoint has a '<module>.bias' key and the live
    module's bias is currently None, create a zero bias parameter of the
    right shape first, so load_state_dict actually loads the real value.
    A plain FP32 checkpoint (no folding) has no such keys, so this is a
    no-op for it -- safe to always call.
    """
    modules = dict(model.named_modules())
    for key, tensor in state_dict.items():
        if key.endswith('.bias'):
            mod_name = key[:-len('.bias')]
            mod = modules.get(mod_name)
            if mod is not None and hasattr(mod, 'bias') and mod.bias is None:
                mod.bias = nn.Parameter(torch.zeros_like(tensor))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints_qgtcrn_metricgan/finetuned_epoch13.tar')
    p.add_argument('--output',     default='checkpoints_qgtcrn_metricgan/qgtcrn.onnx')
    p.add_argument('--opset',      type=int, default=18)
    args = p.parse_args()

    device = torch.device('cpu')
    model = QGTCRN().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    ensure_conv_biases(model, ckpt['model'])
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Dummy inputs matching the model's forward signature. T is arbitrary here;
    # the dynamic axis below lets the exported model accept any length.
    T = 100
    noisy_feat = torch.randn(1, 3, 257, T)
    noisy_spec = torch.randn(1, 257, T, 2)

    export_kwargs = dict(
        input_names=['noisy_feat', 'noisy_spec'],
        output_names=['enhanced_spec'],
        dynamic_axes={
            'noisy_feat':    {3: 'time'},
            'noisy_spec':    {2: 'time'},
            'enhanced_spec': {2: 'time'},
        },
    )

    # Newer PyTorch (2.5+) defaults to the new dynamo-based exporter unless
    # told dynamo=False; older PyTorch doesn't have the dynamo argument at
    # all. Try the modern call first, fall back to the older signature.
    #
    # Also try a descending list of opsets: --opset is the preferred choice
    # (18, to correctly handle the Pad operator per v1's own export notes),
    # but older PyTorch installs may not support it at all. Critically, an
    # opset can "succeed" mechanically (no exception) while still producing
    # numerically WRONG output -- e.g. opset<=15's Reshape 'allowzero'
    # semantics can silently diverge from what the model actually computes.
    # So each candidate is verified against real PyTorch output before being
    # accepted, not just checked for export-time exceptions.
    opset_candidates = [args.opset] + [v for v in [17, 16, 15, 14, 13] if v != args.opset]
    accepted_opset = None
    last_err = None

    with torch.no_grad():
        torch_out = model(noisy_feat, noisy_spec).numpy()

    for opset in opset_candidates:
        try:
            try:
                torch.onnx.export(model, (noisy_feat, noisy_spec), args.output,
                                  dynamo=False, opset_version=opset, **export_kwargs)
            except TypeError:
                torch.onnx.export(model, (noisy_feat, noisy_spec), args.output,
                                  opset_version=opset, **export_kwargs)
        except Exception as e:
            print(f"  opset {opset}: export failed ({type(e).__name__}: {e})")
            last_err = e
            continue

        # Fidelity check -- this is the real gate, not just "did it export"
        try:
            sess = ort.InferenceSession(args.output, providers=['CPUExecutionProvider'])
            onnx_out = sess.run(None, {
                'noisy_feat': noisy_feat.numpy(),
                'noisy_spec': noisy_spec.numpy(),
            })[0]
            diff = np.abs(torch_out - onnx_out).max()
        except Exception as e:
            print(f"  opset {opset}: exported but failed to run/verify ({type(e).__name__}: {e})")
            last_err = e
            continue

        if diff < 5e-3:
            print(f"  opset {opset}: exported and verified (max diff {diff:.2e})")
            accepted_opset = opset
            break
        else:
            print(f"  opset {opset}: exported but numerically WRONG (max diff {diff:.2e} >> 5e-3) -- rejecting, trying next opset")
            last_err = RuntimeError(f"opset {opset} fidelity check failed: diff={diff:.2e}")
            continue

    if accepted_opset is None:
        raise RuntimeError(f"No opset in {opset_candidates} produced a numerically correct export. Last error: {last_err}")

    if accepted_opset != args.opset:
        print(f"\nNote: requested opset {args.opset} unavailable/incorrect on this torch install; used opset {accepted_opset} instead.")

    onnx.checker.check_model(onnx.load(args.output))
    print(f"\nExported to {args.output} (opset {accepted_opset})")

    # Re-run the fidelity numbers one more time for the final printed report
    # (same check as inside the loop, just surfaced clearly at the end).
    sess = ort.InferenceSession(args.output, providers=['CPUExecutionProvider'])
    onnx_out = sess.run(None, {
        'noisy_feat': noisy_feat.numpy(),
        'noisy_spec': noisy_spec.numpy(),
    })[0]
    max_diff = np.abs(torch_out - onnx_out).max()
    print(f"Max abs diff (PyTorch vs ONNX): {max_diff:.2e}")
    # Q-GTCRN has no recurrent layers -- expect this diff to be noticeably
    # tighter than GTCRN's GRU-driven ~1e-3, since there's no per-timestep
    # floating-point accumulation through a recurrence.
    print("Export verified.")

    # Confirm a different length works through the dynamic axis.
    T2 = 250
    onnx_out2 = sess.run(None, {
        'noisy_feat': np.random.randn(1, 3, 257, T2).astype(np.float32),
        'noisy_spec': np.random.randn(1, 257, T2, 2).astype(np.float32),
    })[0]
    print(f"Variable-length check: input T={T2} -> output shape {onnx_out2.shape}")

    # Confirm zero GRU nodes in the exported graph -- the whole point.
    graph = onnx.load(args.output).graph
    op_types = [node.op_type for node in graph.node]
    gru_count = op_types.count('GRU')
    print(f"\nGRU nodes in exported graph: {gru_count}  (expect 0)")
    if gru_count == 0:
        print("Confirmed: fully GRU-free ONNX graph.")
    else:
        print("WARNING: GRU nodes found -- architecture is not fully GRU-free!")


if __name__ == '__main__':
    main()
