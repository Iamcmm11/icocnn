import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ICOCNN_ROOT = os.path.join(ROOT, "icoCNN-master")
if ICOCNN_ROOT not in sys.path:
    sys.path.insert(0, ICOCNN_ROOT)

import acousticTrackingModels as at_models


def save_flat_txt(path, arr):
    flat = arr.reshape(-1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Shape: {arr.shape}\n")
        for v in flat:
            f.write(f"{float(v):.8f}\n")


def load_layer0_conv_input(path):
    data = np.load(path)
    if data.ndim == 6:
        return data.astype(np.float32)
    if data.ndim == 7:
        return data[0].astype(np.float32)
    if data.ndim == 5:
        b, c, r, t, v = data.shape
        if b != 1 or v != 160:
            raise ValueError(f"Unsupported layer0 input shape: {data.shape}")
        charts, h, w = 5, 4, 8
        reshaped = data.reshape(b, c, r, t, charts, h, w)
        reshaped = reshaped.transpose(0, 3, 1, 2, 4, 5, 6)
        return reshaped[0].astype(np.float32)
    raise ValueError(f"Unsupported layer0 input shape: {data.shape}")


def to_network_input(layer0_conv_input):
    # [T, 1, 1, charts, H, W] -> [B=1, C=1, T, charts, H, W]
    if layer0_conv_input.ndim != 6:
        raise ValueError(f"Unexpected layer0 conv input shape: {layer0_conv_input.shape}")
    t, c, r, charts, h, w = layer0_conv_input.shape
    if c != 1 or r != 1:
        raise ValueError(f"Expected layer0 conv input with C=1 and R=1, got {layer0_conv_input.shape}")
    x = layer0_conv_input[:, 0, 0, :, :, :]  # [T, charts, H, W]
    x = np.expand_dims(x, axis=0)           # [1, T, charts, H, W]
    x = np.expand_dims(x, axis=1)           # [1, 1, T, charts, H, W]
    return x.astype(np.float32)


def parse_layers(spec: str) -> List[int]:
    layers = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        layer = int(part)
        if layer < 2 or layer > 5:
            raise ValueError("Only layers 2-5 are supported")
        layers.append(layer)
    if not layers:
        raise ValueError("No valid layer ids provided")
    return sorted(set(layers))


def main():
    parser = argparse.ArgumentParser(description="Generate layer2-5 ConvIco testdata from PyTorch model")
    parser.add_argument("--model", default="models/1sourceTracking_icoCNN_robot_K4096_r2_model.bin")
    parser.add_argument("--layer0-input", default="hls_testdata/layer0/input_rearranged.npy")
    parser.add_argument("--out-dir", default="hls_testdata/layer2-5")
    parser.add_argument("--layers", default="2,3,4,5")
    parser.add_argument("--time-steps", type=int, default=52)
    args = parser.parse_args()

    target_layers = parse_layers(args.layers)
    out_root = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_root, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    net = at_models.IcoTempCNN(r=2, C=32, smooth_vertices=True)
    model_path = os.path.join(ROOT, args.model)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location="cpu")
        net.load_state_dict(state_dict)
    net.eval()

    captured_inputs: Dict[int, np.ndarray] = {}
    captured_outputs: Dict[int, np.ndarray] = {}
    handles = []

    def make_hook(layer_id):
        def hook(module, inputs, output):
            inp = inputs[0].detach().cpu().numpy()
            out = output.detach().cpu().numpy()
            captured_inputs[layer_id] = np.squeeze(inp, axis=0)   # [T,C,R,charts,H,W]
            captured_outputs[layer_id] = np.squeeze(out, axis=0)  # [T,C,R,charts,H,W]
        return hook

    for layer_id in target_layers:
        handles.append(net.ico_cnn[layer_id].register_forward_hook(make_hook(layer_id)))

    layer0_input = load_layer0_conv_input(os.path.join(ROOT, args.layer0_input))
    if layer0_input.shape[0] < args.time_steps:
        raise ValueError(f"layer0 input has only {layer0_input.shape[0]} frames")
    layer0_input = layer0_input[: args.time_steps]
    network_input = to_network_input(layer0_input)

    with torch.no_grad():
        _ = net(torch.from_numpy(network_input).float())

    for handle in handles:
        handle.remove()

    for layer_id in target_layers:
        layer_dir = os.path.join(out_root, f"layer{layer_id}")
        os.makedirs(layer_dir, exist_ok=True)
        os.makedirs(os.path.join(layer_dir, "debug_intermediate"), exist_ok=True)
        os.makedirs(os.path.join(layer_dir, "debug_intermediate_cpp"), exist_ok=True)

        conv = net.ico_cnn[layer_id]
        layer_input = captured_inputs[layer_id].astype(np.float32)
        layer_output = captured_outputs[layer_id].astype(np.float32)
        weight = conv.weight.detach().cpu().numpy().astype(np.float32)
        bias = conv.bias.detach().cpu().numpy().astype(np.float32) if conv.bias is not None else np.zeros((32,), dtype=np.float32)
        kernel_idx = conv.kernel_expansion_idx.cpu().numpy().astype(np.int32)
        reorder_idx = conv.padding.reorder_idx.cpu().numpy().astype(np.int32)

        np.save(os.path.join(layer_dir, "input_rearranged.npy"), layer_input)
        np.save(os.path.join(layer_dir, "output.npy"), layer_output)
        np.save(os.path.join(layer_dir, "weight.npy"), weight)
        np.save(os.path.join(layer_dir, "bias.npy"), bias)
        np.save(os.path.join(layer_dir, "kernel_expansion_idx.npy"), kernel_idx)
        np.save(os.path.join(layer_dir, "reorder_idx.npy"), reorder_idx)

        save_flat_txt(os.path.join(layer_dir, "input_rearranged.txt"), layer_input)
        save_flat_txt(os.path.join(layer_dir, "output.txt"), layer_output)
        save_flat_txt(os.path.join(layer_dir, "weight.txt"), weight)
        save_flat_txt(os.path.join(layer_dir, "bias.txt"), bias)
        save_flat_txt(os.path.join(layer_dir, "kernel_expansion_idx.txt"), kernel_idx)
        save_flat_txt(os.path.join(layer_dir, "reorder_idx.txt"), reorder_idx)

        with open(os.path.join(layer_dir, "metadata.txt"), "w", encoding="utf-8") as f:
            f.write("# Layer2-5 shared block verification data\n")
            f.write(f"layer={layer_id}\n")
            f.write(f"model={args.model}\n")
            f.write(f"layer0_input={args.layer0_input}\n")
            f.write(f"time_steps={args.time_steps}\n")
            f.write(f"input_shape={layer_input.shape}\n")
            f.write(f"output_shape={layer_output.shape}\n")

        print(f"Generated layer{layer_id} testdata:")
        print(f"  dir={layer_dir}")
        print(f"  input_shape={layer_input.shape}")
        print(f"  output_shape={layer_output.shape}")


if __name__ == "__main__":
    main()
