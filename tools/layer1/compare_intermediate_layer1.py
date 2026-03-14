import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY_DIR = os.path.join(ROOT, "hls_testdata", "layer1", "debug_intermediate")
CPP_DIR = os.path.join(ROOT, "hls_testdata", "layer1", "debug_intermediate_cpp")


def load_values(path):
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            for tok in s.replace(",", " ").split():
                try:
                    vals.append(float(tok))
                except ValueError:
                    pass
    return np.array(vals, dtype=np.float64)


def compare(py_file, cpp_file, name):
    py_path = os.path.join(PY_DIR, py_file)
    cpp_path = os.path.join(CPP_DIR, cpp_file)

    if not os.path.exists(py_path):
        print(f"[Missing] {py_path}")
        return
    if not os.path.exists(cpp_path):
        print(f"[Missing] {cpp_path}")
        return

    a = load_values(py_path)
    b = load_values(cpp_path)

    print("=" * 72)
    print(name)
    print(f"python={len(a)} values, cpp={len(b)} values")

    if len(a) != len(b):
        print("FAIL: length mismatch")
        return

    diff = np.abs(a - b)
    print(f"Max Error: {diff.max():.8e}")
    print(f"RMSE:     {np.sqrt(np.mean(diff * diff)):.8e}")
    print(f"Mean Abs: {diff.mean():.8e}")

    if diff.max() < 1e-5:
        print("PASS")
    elif diff.max() < 1e-3:
        print("WARN")
    else:
        print("FAIL")


def main():
    compare("py_frame0_input.txt", "cpp_frame0_input.txt", "1) Frame0 Input")
    compare("py_frame0_padded.txt", "cpp_frame0_padded.txt", "2) After PadIco")
    compare("py_frame0_final_output.txt", "cpp_frame0_final_output.txt", "3) Frame0 Final Output")


if __name__ == "__main__":
    main()
