import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ICOCNN_ROOT = os.path.join(ROOT, "icoCNN-master")
if ICOCNN_ROOT not in sys.path:
    sys.path.insert(0, ICOCNN_ROOT)

import acousticTrackingDataset as at_dataset
import acousticTrackingLearners as at_learners
import acousticTrackingModels as at_models
import acousticTrackingModules as at_modules
import icoCNN
from acousticTrackingDataset import Parameter


def fake_quantize_tensor(x, num_bits):
    if num_bits is None or num_bits >= 32:
        return x
    if not torch.is_floating_point(x):
        return x

    max_abs = x.detach().abs().max()
    if max_abs.item() == 0.0:
        return x.clone()

    qmax = (1 << (num_bits - 1)) - 1
    scale = max_abs / qmax
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    return q * scale


def quantize_model_weights(model, weight_bits, quantize_bias=False):
    quantized = copy.deepcopy(model)
    with torch.no_grad():
        for name, param in quantized.named_parameters():
            if not torch.is_floating_point(param):
                continue
            if ("bias" in name) and not quantize_bias:
                continue
            param.copy_(fake_quantize_tensor(param, weight_bits))
    return quantized


class FakeQuantWrapper(nn.Module):
    def __init__(self, model, input_bits=None, activation_bits=None):
        super().__init__()
        self.model = model
        self.input_bits = input_bits
        self.activation_bits = activation_bits
        self._hooks = []

        if activation_bits is not None and activation_bits < 32:
            hook_types = (
                icoCNN.ConvIco,
                at_modules.CausConv1d,
                icoCNN.LNormIco,
                icoCNN.PoolIco,
                icoCNN.SmoothVertices,
                icoCNN.CleanVertices,
            )
            for module in self.model.modules():
                if isinstance(module, hook_types):
                    self._hooks.append(module.register_forward_hook(self._activation_hook))

    def _activation_hook(self, module, inputs, output):
        if isinstance(output, torch.Tensor):
            return fake_quantize_tensor(output, self.activation_bits)
        return output

    def forward(self, x):
        if self.input_bits is not None and self.input_bits < 32:
            x = fake_quantize_tensor(x, self.input_bits)
        return self.model(x)

    def close(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


def build_test_dataset(k=4096, trajectory_seconds=20):
    path_test = os.path.join(ROOT, "datasets", "LibriSpeech", "test-clean")
    corpus_dataset_test = at_dataset.LibriSpeechDataset(path_test, trajectory_seconds, return_vad=True)

    windowing = at_dataset.Windowing(k, k * 3 // 4, window=np.hanning)

    return at_dataset.RandomTrajectoryDataset(
        sourceDataset=corpus_dataset_test,
        room_sz=Parameter([3, 3, 2.5], [10, 8, 6]),
        T60=Parameter(0.2, 1.3),
        abs_weights=Parameter([0.5] * 6, [1.0] * 6),
        array_setup=at_dataset.benchmark2_array_setup,
        array_pos=Parameter([0.1, 0.1, 0.1], [0.9, 0.9, 0.5]),
        SNR=Parameter(5, 30),
        nb_points=156,
        transforms=[windowing],
    )


def build_learner(model, device, r=2, k=4096, fs=16000, n_mics=12):
    learner = at_learners.OneSourceTrackingLearner(
        model,
        at_learners.TrackingFromIcoMapsPreprocessor(
            n_mics,
            k,
            r,
            at_dataset.benchmark2_array_setup.mic_pos,
            fs,
            apply_vad=True,
        ),
    )
    if device.type == "cuda":
        learner.cuda()
    else:
        learner.cpu()
    return learner


def evaluate(learner, dataset, batch_size, nb_batches):
    loss, rmsae = learner.test_epoch(dataset, batch_size, nb_batchs=nb_batches)
    return {
        "loss": float(loss.detach().cpu().item() if torch.is_tensor(loss) else loss),
        "rmsae_deg": float(rmsae.detach().cpu().item() if torch.is_tensor(rmsae) else rmsae),
    }


def load_model(model_path, device, r=2, channels=32):
    model = at_models.IcoTempCNN(r, channels)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def summarize_delta(baseline, current):
    return {
        "loss_delta": current["loss"] - baseline["loss"],
        "rmsae_delta_deg": current["rmsae_deg"] - baseline["rmsae_deg"],
        "loss_ratio": current["loss"] / baseline["loss"] if baseline["loss"] != 0 else None,
        "rmsae_ratio": current["rmsae_deg"] / baseline["rmsae_deg"] if baseline["rmsae_deg"] != 0 else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate fake-quantized icoCNN accuracy drift on the PyTorch model.")
    parser.add_argument("--model-path", default=os.path.join(ROOT, "models", "1sourceTracking_icoCNN_robot_K4096_r2_model.bin"))
    parser.add_argument("--r", type=int, default=2)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=5, help="Trajectory batch size. Matches the original 1sourceTracking_icoCNN.py test/eval setting.")
    parser.add_argument("--nb-batches", type=int, default=10, help="Number of evaluation batches. Use --full-eval to run the whole test set.")
    parser.add_argument("--full-eval", action="store_true", help="Evaluate on the full test split instead of a limited number of batches.")
    parser.add_argument("--weight-bits", type=int, nargs="+", default=[16, 12, 8])
    parser.add_argument("--input-bits", type=int, default=16)
    parser.add_argument("--activation-bits", type=int, nargs="+", default=[16, 12, 8], help="Activation fake-quant bit widths, for example: 16 12 8.")
    parser.add_argument("--skip-activation-quant", action="store_true")
    parser.add_argument("--quantize-bias", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = build_test_dataset(k=4096)
    nb_batches = None if args.full_eval else args.nb_batches

    base_model = load_model(args.model_path, device, r=args.r, channels=args.channels)
    baseline_learner = build_learner(base_model, device, r=args.r)
    baseline = evaluate(baseline_learner, dataset, args.batch_size, nb_batches)

    results = {
        "config": {
            "model_path": args.model_path,
            "batch_size": args.batch_size,
            "nb_batches": "full" if args.full_eval else args.nb_batches,
            "input_bits": args.input_bits,
            "weight_bits": args.weight_bits,
            "activation_bits": [] if args.skip_activation_quant else args.activation_bits,
            "quantize_bias": args.quantize_bias,
            "device": str(device),
        },
        "baseline_fp32": baseline,
        "experiments": [],
    }

    print("=== Baseline FP32 ===")
    print(json.dumps(baseline, indent=2, ensure_ascii=False))

    for weight_bits in args.weight_bits:
        quant_weight_model = quantize_model_weights(base_model, weight_bits, quantize_bias=args.quantize_bias)

        # Weight-only quantization
        wrapped = FakeQuantWrapper(quant_weight_model, input_bits=None, activation_bits=None)
        learner = build_learner(wrapped, device, r=args.r)
        weight_only = evaluate(learner, dataset, args.batch_size, nb_batches)
        weight_only_result = {
            "name": f"weight{weight_bits}",
            "weight_bits": weight_bits,
            "input_bits": None,
            "activation_bits": None,
            "metrics": weight_only,
            "delta_vs_fp32": summarize_delta(baseline, weight_only),
        }
        results["experiments"].append(weight_only_result)
        print(f"=== Weight {weight_bits}-bit ===")
        print(json.dumps(weight_only_result, indent=2, ensure_ascii=False))
        wrapped.close()

        if args.skip_activation_quant:
            continue

        for act_bits in args.activation_bits:
            if act_bits >= 32:
                continue
            wrapped = FakeQuantWrapper(
                quantize_model_weights(base_model, weight_bits, quantize_bias=args.quantize_bias),
                input_bits=args.input_bits,
                activation_bits=act_bits,
            )
            learner = build_learner(wrapped, device, r=args.r)
            metrics = evaluate(learner, dataset, args.batch_size, nb_batches)
            result = {
                "name": f"w{weight_bits}_a{act_bits}_in{args.input_bits}",
                "weight_bits": weight_bits,
                "input_bits": args.input_bits,
                "activation_bits": act_bits,
                "metrics": metrics,
                "delta_vs_fp32": summarize_delta(baseline, metrics),
            }
            results["experiments"].append(result)
            print(f"=== Weight {weight_bits}-bit / Act {act_bits}-bit / Input {args.input_bits}-bit ===")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            wrapped.close()

    print("=== Summary JSON ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
