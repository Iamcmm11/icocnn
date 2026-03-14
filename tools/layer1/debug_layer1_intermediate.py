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


def save_flat(path, arr):
    flat = arr.reshape(-1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Shape: {arr.shape}\n")
        f.write(f"# Min: {flat.min():.8f}, Max: {flat.max():.8f}, Mean: {flat.mean():.8f}\n")
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

    save_flat(os.path.join(DEBUG_DIR, "py_frame0_input.txt"), frame0.numpy())

    x_padded = layer1.padding(frame0)
    save_flat(os.path.join(DEBUG_DIR, "py_frame0_padded.txt"), x_padded.detach().cpu().numpy())

    # Keep the same intermediate decomposition style as layer0 debug.
    x_reshaped = einops.rearrange(x_padded, "C R charts H W -> (C R) (charts H) W", C=32, R=6, charts=5)
    kernel = layer1.get_kernel()
    kernel_2d = einops.rearrange(kernel, "Cout Rout Cin Rin Hk Wk -> (Cout Rout) (Cin Rin) Hk Wk", Hk=3, Wk=3)
    bias_repeated = einops.repeat(layer1.bias, "Cout -> (Cout Rout)", Cout=32, Rout=6)

    y_conv = torch.nn.functional.conv2d(x_reshaped.unsqueeze(0), kernel_2d, bias_repeated, padding=(1, 1)).squeeze(0)
    y_reshaped = einops.rearrange(y_conv, "(C R) (charts H) W -> C R charts H W", C=32, R=6, charts=5)
    y_unpadded = y_reshaped[..., 1:-1, 1:-1]
    y_final = layer1.process_vertices(y_unpadded)

    save_flat(os.path.join(DEBUG_DIR, "py_frame0_final_output.txt"), y_final.detach().cpu().numpy())

    print("Saved layer1 python debug intermediates:")
    print(f"  {os.path.join(DEBUG_DIR, 'py_frame0_input.txt')}")
    print(f"  {os.path.join(DEBUG_DIR, 'py_frame0_padded.txt')}")
    print(f"  {os.path.join(DEBUG_DIR, 'py_frame0_final_output.txt')}")


if __name__ == "__main__":
    main()
