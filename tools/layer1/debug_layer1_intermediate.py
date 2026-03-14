import os
import sys
import numpy as np
import torch
import einops

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ICOCNN_ROOT = os.path.join(ROOT, "icoCNN-master")
if ICOCNN_ROOT not in sys.path:
    sys.path.insert(0, ICOCNN_ROOT)

from icoCNN.icoCNN import ConvIco

DATA_DIR = os.path.join(ROOT, "hls_testdata", "layer1")
DEBUG_DIR = os.path.join(DATA_DIR, "debug_intermediate")
os.makedirs(DEBUG_DIR, exist_ok=True)


def save_tensor_as_matrix(path, arr, name):
    data = np.asarray(arr)
    flat = data.reshape(-1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {name}\n")
        f.write(f"# Shape: {arr.shape}\n")
        f.write(f"# Min: {flat.min():.8f}, Max: {flat.max():.8f}, Mean: {flat.mean():.8f}\n")
        f.write("#" + "=" * 70 + "\n\n")

        if data.ndim == 5:
            d0, d1, d2, d3, d4 = data.shape
            for i0 in range(d0):
                for i1 in range(d1):
                    for i2 in range(d2):
                        f.write(f"# [{i0}, {i1}, chart{i2}] - Shape: ({d3}, {d4})\n")
                        for i3 in range(d3):
                            row = "  ".join(f"{float(v):.8f}" for v in data[i0, i1, i2, i3, :])
                            f.write(f"  {row}\n")
                        f.write("\n")
        elif data.ndim == 4:
            d0, d1, d2, d3 = data.shape
            for i0 in range(d0):
                for i1 in range(d1):
                    f.write(f"# [{i0}, chart{i1}] - Shape: ({d2}, {d3})\n")
                    for i2 in range(d2):
                        row = "  ".join(f"{float(v):.8f}" for v in data[i0, i1, i2, :])
                        f.write(f"  {row}\n")
                    f.write("\n")
        elif data.ndim == 3:
            d0, d1, d2 = data.shape
            for i0 in range(d0):
                f.write(f"# [slice {i0}] - Shape: ({d1}, {d2})\n")
                for i1 in range(d1):
                    row = "  ".join(f"{float(v):.8f}" for v in data[i0, i1, :])
                    f.write(f"  {row}\n")
                f.write("\n")
        elif data.ndim == 2:
            d0, _ = data.shape
            for i0 in range(d0):
                row = "  ".join(f"{float(v):.8f}" for v in data[i0, :])
                f.write(f"  {row}\n")
        else:
            for v in flat:
                f.write(f"{float(v):.8f}\n")


def main():
    input_path = os.path.join(DATA_DIR, "input_rearranged.npy")
    weight_path = os.path.join(DATA_DIR, "weight.npy")
    bias_path = os.path.join(DATA_DIR, "bias.npy")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing {input_path}, run generate_layer1_testdata.py first")

    x_all = np.load(input_path).astype(np.float32)  # [T, 32, 6, 5, 4, 8]
    weight = np.load(weight_path).astype(np.float32)
    bias = np.load(bias_path).astype(np.float32)

    frame0 = torch.from_numpy(x_all[0])  # [32, 6, 5, 4, 8]

    layer1 = ConvIco(r=2, Cin=32, Cout=32, Rin=6, Rout=6, smooth_vertices=True)
    layer1.weight.data = torch.from_numpy(weight)
    layer1.bias.data = torch.from_numpy(bias)
    layer1.eval()

    save_tensor_as_matrix(
        os.path.join(DEBUG_DIR, "py_frame0_input.txt"),
        frame0.numpy(),
        "Frame0 Input [CIN,RIN,CHARTS,H,W]",
    )

    x_padded = layer1.padding(frame0)
    save_tensor_as_matrix(
        os.path.join(DEBUG_DIR, "py_frame0_padded.txt"),
        x_padded.detach().cpu().numpy(),
        "After PadIco [CIN,RIN,CHARTS,H_PADDED,W_PADDED]",
    )

    # Keep the same intermediate decomposition style as layer0 debug.
    x_reshaped = einops.rearrange(x_padded, "C R charts H W -> (C R) (charts H) W", C=32, R=6, charts=5)
    kernel = layer1.get_kernel()
    kernel_2d = einops.rearrange(kernel, "Cout Rout Cin Rin Hk Wk -> (Cout Rout) (Cin Rin) Hk Wk", Hk=3, Wk=3)
    bias_repeated = einops.repeat(layer1.bias, "Cout -> (Cout Rout)", Cout=32, Rout=6)

    y_conv = torch.nn.functional.conv2d(x_reshaped.unsqueeze(0), kernel_2d, bias_repeated, padding=(1, 1)).squeeze(0)
    y_reshaped = einops.rearrange(y_conv, "(C R) (charts H) W -> C R charts H W", C=32, R=6, charts=5)
    y_unpadded = y_reshaped[..., 1:-1, 1:-1]
    y_final = layer1.process_vertices(y_unpadded)

    save_tensor_as_matrix(
        os.path.join(DEBUG_DIR, "py_frame0_final_output.txt"),
        y_final.detach().cpu().numpy(),
        "Frame0 Final Output [COUT,ROUT,CHARTS,H,W]",
    )

    print("Saved layer1 python debug intermediates:")
    print(f"  {os.path.join(DEBUG_DIR, 'py_frame0_input.txt')}")
    print(f"  {os.path.join(DEBUG_DIR, 'py_frame0_padded.txt')}")
    print(f"  {os.path.join(DEBUG_DIR, 'py_frame0_final_output.txt')}")


if __name__ == "__main__":
    main()
