import numpy as np

# 加载 Python 和 C++ 的 padding 结果
py_padded = []
with open('hls_testdata/layer0/debug_intermediate/py_frame0_padded.txt', 'r') as f:
    for line in f:
        if line.strip() and not line.startswith('#'):
            vals = line.strip().split()
            for v in vals:
                try:
                    py_padded.append(float(v))
                except:
                    pass

cpp_padded = []
with open('hls_testdata/layer0/debug_intermediate_cpp/cpp_frame0_padded.txt', 'r') as f:
    for line in f:
        if line.strip() and not line.startswith('#'):
            vals = line.strip().split()
            for v in vals:
                try:
                    cpp_padded.append(float(v))
                except:
                    pass

py_padded = np.array(py_padded)
cpp_padded = np.array(cpp_padded)

print("Python padded shape:", py_padded.shape)
print("C++ padded shape:", cpp_padded.shape)

# 找出差异最大的几个位置
diff = np.abs(py_padded - cpp_padded)
max_indices = np.argsort(diff)[-10:][::-1]

print("\nTop 10 differences:")
for i, idx in enumerate(max_indices):
    print(f"{i+1}. Index {idx}: Python={py_padded[idx]:.6f}, C++={cpp_padded[idx]:.6f}, Diff={diff[idx]:.6f}")

# 检查 reorder_idx
reorder_idx = np.load('hls_testdata/layer0/reorder_idx.npy')
print(f"\nreorder_idx shape: {reorder_idx.shape}")

# 输出第一个 chart 的 padding 索引
print("\nChart 0 reorder_idx [0,0,:3,:3]:")
print(reorder_idx[0,0,:3,:3])
