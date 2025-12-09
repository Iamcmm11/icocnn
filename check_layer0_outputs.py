import numpy as np

# 1. 原始 Hook 保存的 Layer0 输出
print("=== debug_outputs/Layer0_IcoConv_output.npy ===")
layer0 = np.load("debug_outputs/Layer0_IcoConv_output.npy")
print("Layer0_IcoConv_output shape:", layer0.shape)
print("  first 10 values (flatten):")
print(layer0.flatten()[:10])

# 2. 拷贝到 HLS 目录的参考输出
print("\n=== hls_testdata/layer0/output_layer0.npy ===")
hls_ref = np.load("hls_testdata/layer0/output_layer0.npy")
print("output_layer0 shape:", hls_ref.shape)
print("  first 10 values (flatten):")
print(hls_ref.flatten()[:10])

# 3. 再检查 txt 版是否和 npy 一致（只看前几行）
print("\n=== hls_testdata/layer0/output_layer0.txt (first 10 lines) ===")
with open("hls_testdata/layer0/output_layer0.txt", "r") as f:
    for i in range(10):
        line = f.readline()
        if not line:
            break
        print(f"[{i}] {line.strip()}")