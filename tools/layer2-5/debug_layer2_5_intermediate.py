import argparse
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


def save_tensor_as_matrix(path, arr, name):
    data = np.asarray(arr)
    flat = data.reshape(-1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {name}\n")
        f.write(f"# Shape: {data.shape}\n")
        f.write(f"# Min: {flat.min():.8f}, Max: {flat.max():.8f}, Mean: {flat.mean():.8f}\n")
        f.write("#" + "=" * 70 + "\n\n")
        for v in flat:
            f.write(f"{float(v):.8f}\n")


def main():
    parser = argparse.ArgumentParser(description="Dump Python intermediates for layer2-5 shared ConvIco block")
    parser.add_argument("--layer", type=int, required=True, choices=[2, 3, 4, 5])
    args = parser.parse_args()

    data_dir = os.path.join(ROOT, "hls_testdata", "layer2-5", f"layer{args.layer}")
    debug_dir = os.path.join(data_dir, "debug_intermediate")
    os.makedirs(debug_dir, exist_ok=True)

    input_path = os.path.join(data_dir, "input_rearranged.npy")
    weight_path = os.path.join(data_dir, "weight.npy")
    bias_path = os.path.join(data_dir, "bias.npy")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing {input_path}, run generate_layer2_5_testdata.py first")

    x_all = np.load(input_path).astype(np.float32)  # [T,32,6,5,2,4]
    weight = np.load(weight_path).astype(np.float32)
    bias = np.load(bias_path).astype(np.float32)

    frame0 = torch.from_numpy(x_all[0])  # [32,6,5,2,4]

    conv = ConvIco(r=1, Cin=32, Cout=32, Rin=6, Rout=6, smooth_vertices=True)
    conv.weight.data = torch.from_numpy(weight)
    conv.bias.data = torch.from_numpy(bias)
    conv.eval()

    save_tensor_as_matrix(os.path.join(debug_dir, "py_frame0_input.txt"), frame0.numpy(), "Frame0 Input [CIN,RIN,CHARTS,H,W]")

    x_padded = conv.padding(frame0)
    save_tensor_as_matrix(
        os.path.join(debug_dir, "py_frame0_padded.txt"),
        x_padded.detach().cpu().numpy(),
        "After PadIco [CIN,RIN,CHARTS,H_PADDED,W_PADDED]",
    )

    x_reshaped = einops.rearrange(x_padded, "C R charts H W -> (C R) (charts H) W", C=32, R=6, charts=5)
    kernel = conv.get_kernel()
    kernel_2d = einops.rearrange(kernel, "Cout Rout Cin Rin Hk Wk -> (Cout Rout) (Cin Rin) Hk Wk", Hk=3, Wk=3)
    bias_repeated = einops.repeat(conv.bias, "Cout -> (Cout Rout)", Cout=32, Rout=6)

    y_conv = torch.nn.functional.conv2d(x_reshaped.unsqueeze(0), kernel_2d, bias_repeated, padding=(1, 1)).squeeze(0)
    y_reshaped = einops.rearrange(y_conv, "(C R) (charts H) W -> C R charts H W", C=32, R=6, charts=5)
    y_unpadded = y_reshaped[..., 1:-1, 1:-1]
    y_final = conv.process_vertices(y_unpadded)

    save_tensor_as_matrix(
        os.path.join(debug_dir, "py_frame0_final_output.txt"),
        y_final.detach().cpu().numpy(),
        "Frame0 Final Output [COUT,ROUT,CHARTS,H,W]",
    )

    print(f"Saved Python intermediates for layer{args.layer}: {debug_dir}")


if __name__ == "__main__":
    main()
