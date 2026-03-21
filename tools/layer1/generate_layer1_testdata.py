import argparse
import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import acousticTrackingModels as at_models


def save_flat_txt(path, arr):
    flat = arr.reshape(-1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Shape: {arr.shape}\n")
        for v in flat:
            f.write(f"{float(v):.8f}\n")


def load_compatible_state_dict(model, state_dict):
    incompatible = model.load_state_dict(state_dict, strict=False)
    non_buffer_missing = [k for k in incompatible.missing_keys if not k.endswith(".mask")]
    if non_buffer_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint is incompatible with current model.\n"
            f"Unexpected keys: {list(incompatible.unexpected_keys)}\n"
            f"Missing non-buffer keys: {non_buffer_missing}"
        )
    if incompatible.missing_keys:
        print(f"Ignoring {len(incompatible.missing_keys)} missing mask buffers from legacy checkpoint.")


def load_layer0_input(path):
    data = np.load(path)
    # target format: [T, CIN=1, RIN=1, CHARTS, H, W]
    if data.ndim == 6:
        return data.astype(np.float32)
    if data.ndim == 7:
        # [B,T,C,R,charts,H,W]
        return data[0].astype(np.float32)
    if data.ndim == 5:
        # [B,C,R,T,V] -> [T,C,R,charts,H,W]
        b, c, r, t, v = data.shape
        if b != 1 or v != 160:
            raise ValueError(f"Unsupported layer0 input shape: {data.shape}")
        charts, h, w = 5, 4, 8
        reshaped = data.reshape(b, c, r, t, charts, h, w)
        reshaped = reshaped.transpose(0, 3, 1, 2, 4, 5, 6)
        return reshaped[0].astype(np.float32)
    raise ValueError(f"Unsupported layer0 input shape: {data.shape}")


def main():
    parser = argparse.ArgumentParser(description="Generate layer1 testdata from PyTorch model")
    parser.add_argument("--model", default="models/1sourceTracking_icoCNN_robot_K4096_r2_model.bin")
    parser.add_argument("--layer0-input", default="hls_testdata/layer0/input_rearranged.npy")
    parser.add_argument("--out-dir", default="hls_testdata/layer1")
    parser.add_argument("--time-steps", type=int, default=52)
    args = parser.parse_args()

    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    net = at_models.IcoTempCNN(r=2, C=32, smooth_vertices=True)
    if os.path.exists(os.path.join(ROOT, args.model)):
        sd = torch.load(os.path.join(ROOT, args.model), map_location="cpu")
        load_compatible_state_dict(net, sd)
    net.eval()

    l0 = net.ico_cnn[0]
    l1 = net.ico_cnn[1]

    layer0_input = load_layer0_input(os.path.join(ROOT, args.layer0_input))
    if layer0_input.shape[0] < args.time_steps:
        raise ValueError(f"layer0 input has only {layer0_input.shape[0]} frames")
    layer0_input = layer0_input[: args.time_steps]

    x = torch.from_numpy(layer0_input).float()

    with torch.no_grad():
        # layer0 output is layer1 input
        layer1_input = l0(x).cpu().numpy()  # [T,32,6,5,4,8]
        layer1_output = l1(torch.from_numpy(layer1_input).float()).cpu().numpy()  # [T,32,6,5,4,8]

    weight = l1.weight.detach().cpu().numpy()
    bias = l1.bias.detach().cpu().numpy() if l1.bias is not None else np.zeros((32,), dtype=np.float32)
    kernel_idx = l1.kernel_expansion_idx.cpu().numpy().astype(np.int32)
    reorder_idx = l1.padding.reorder_idx.cpu().numpy().astype(np.int32)

    np.save(os.path.join(out_dir, "input_rearranged.npy"), layer1_input)
    np.save(os.path.join(out_dir, "output_layer1.npy"), layer1_output)
    np.save(os.path.join(out_dir, "weight.npy"), weight)
    np.save(os.path.join(out_dir, "bias.npy"), bias)
    np.save(os.path.join(out_dir, "kernel_expansion_idx.npy"), kernel_idx)
    np.save(os.path.join(out_dir, "reorder_idx.npy"), reorder_idx)

    save_flat_txt(os.path.join(out_dir, "input_rearranged.txt"), layer1_input)
    save_flat_txt(os.path.join(out_dir, "output_layer1.txt"), layer1_output)
    save_flat_txt(os.path.join(out_dir, "weight.txt"), weight)
    save_flat_txt(os.path.join(out_dir, "bias.txt"), bias)
    save_flat_txt(os.path.join(out_dir, "kernel_expansion_idx.txt"), kernel_idx)
    save_flat_txt(os.path.join(out_dir, "reorder_idx.txt"), reorder_idx)

    meta = os.path.join(out_dir, "layer1_input_source.txt")
    with open(meta, "w", encoding="utf-8") as f:
        f.write("# Layer1 testdata generated from model and Layer0 input\n")
        f.write(f"model={args.model}\n")
        f.write(f"layer0_input={args.layer0_input}\n")
        f.write(f"time_steps={args.time_steps}\n")

    print("Layer1 testdata generated:")
    print(f"  out_dir={out_dir}")
    print(f"  input shape={layer1_input.shape}")
    print(f"  output shape={layer1_output.shape}")
    print(f"  weight shape={weight.shape}")
    print(f"  kernel_idx shape={kernel_idx.shape}")
    print(f"  reorder_idx shape={reorder_idx.shape}")


if __name__ == "__main__":
    main()
