import argparse
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_values(path):
    values = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            values.append(float(s))
    return np.array(values, dtype=np.float64)


def compare(py_path, cpp_path, title):
    if not os.path.exists(py_path):
        print(f"[Missing] {py_path}")
        return
    if not os.path.exists(cpp_path):
        print(f"[Missing] {cpp_path}")
        return

    a = load_values(py_path)
    b = load_values(cpp_path)

    print("=" * 72)
    print(title)
    print(f"python={len(a)} values, cpp={len(b)} values")

    if len(a) != len(b):
        print("FAIL: length mismatch")
        return

    diff = np.abs(a - b)
    print(f"Max Error: {diff.max():.8e}")
    print(f"RMSE:     {np.sqrt(np.mean(diff * diff)):.8e}")
    print(f"Mean Abs: {diff.mean():.8e}")
    print("PASS" if diff.max() < 1e-5 else ("WARN" if diff.max() < 1e-3 else "FAIL"))


def main():
    parser = argparse.ArgumentParser(description="Compare Python/C++ intermediates for layer2-5 shared ConvIco block")
    parser.add_argument("--layer", type=int, required=True, choices=[2, 3, 4, 5])
    args = parser.parse_args()

    py_dir = os.path.join(ROOT, "hls_testdata", "layer2-5", f"layer{args.layer}", "debug_intermediate")
    cpp_dir = os.path.join(ROOT, "hls_testdata", "layer2-5", f"layer{args.layer}", "debug_intermediate_cpp")

    compare(os.path.join(py_dir, "py_frame0_input.txt"), os.path.join(cpp_dir, "cpp_frame0_input.txt"), f"Layer{args.layer} Frame0 Input")
    compare(os.path.join(py_dir, "py_frame0_padded.txt"), os.path.join(cpp_dir, "cpp_frame0_padded.txt"), f"Layer{args.layer} After PadIco")
    compare(os.path.join(py_dir, "py_frame0_final_output.txt"), os.path.join(cpp_dir, "cpp_frame0_final_output.txt"), f"Layer{args.layer} Frame0 Final Output")


if __name__ == "__main__":
    main()
