"""
Layer0 中间层对齐调试脚本 - Python 端
输出 ConvIco Layer0 的所有中间计算结果（只针对第0帧，方便与 C++ 对比）

输出文件（保存到 hls_testdata/layer0/debug_intermediate/）：
1. py_frame0_input.txt              - 输入帧 [1, 1, 5, 4, 8]
2. py_frame0_after_clean.txt        - CleanVertices 后 [1, 1, 5, 4, 8]
3. py_frame0_padded.txt             - PadIco 后 [1, 5, 6, 10]
4. py_frame0_reshaped_input.txt     - Reshape 后 [1, 30, 10]
5. py_frame0_kernel.txt             - 卷积核 [192, 1, 3, 3]
6. py_frame0_conv2d_output.txt      - Conv2D 后 [192, 30, 10]
7. py_frame0_reshaped_output.txt    - Reshape 后 [32, 6, 5, 6, 10]
8. py_frame0_unpadded.txt           - 去 padding 后 [32, 6, 5, 4, 8]
9. py_frame0_final_output.txt       - 最终输出 [32, 6, 5, 4, 8]

注意：文件名加 py_ 前缀，与 C++ 端的 cpp_ 前缀区分
"""

import torch
import numpy as np
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from icoCNN.icoCNN import ConvIco

# 配置
DEBUG_DIR = 'hls_testdata/layer0/debug_intermediate'
os.makedirs(DEBUG_DIR, exist_ok=True)

def save_tensor_as_matrix(tensor, filename, name="Tensor"):
    """
    保存 tensor 为 MATLAB 风格的矩阵文本
    每个 2D 切片单独保存，方便查看
    """
    data = tensor.detach().cpu().numpy()
    
    with open(filename, 'w') as f:
        f.write(f"# {name}\n")
        f.write(f"# Shape: {data.shape}\n")
        f.write(f"# Min: {data.min():.8f}, Max: {data.max():.8f}, Mean: {data.mean():.8f}\n")
        f.write("#" + "="*70 + "\n\n")
        
        # 如果是高维，逐层展开
        if data.ndim == 5:  # [C, R, charts, H, W]
            C, R, charts, H, W = data.shape
            for c in range(C):
                for r in range(R):
                    for ch in range(charts):
                        f.write(f"# [{c}, {r}, chart{ch}] - Shape: ({H}, {W})\n")
                        for h in range(H):
                            f.write("  ")
                            for w in range(W):
                                f.write(f"{data[c, r, ch, h, w]:10.6f}  ")
                            f.write("\n")
                        f.write("\n")
        
        elif data.ndim == 4:  # [R, charts, H, W] 或 [C, H, W]
            if data.shape[1] == 5:  # [R, charts=5, H, W]
                R, charts, H, W = data.shape
                for r in range(R):
                    for ch in range(charts):
                        f.write(f"# [R{r}, chart{ch}] - Shape: ({H}, {W})\n")
                        for h in range(H):
                            f.write("  ")
                            for w in range(W):
                                f.write(f"{data[r, ch, h, w]:10.6f}  ")
                            f.write("\n")
                        f.write("\n")
            else:  # [C, charts, H, W]
                C, charts, H, W = data.shape
                for c in range(min(C, 3)):  # 只输出前3个通道
                    for ch in range(charts):
                        f.write(f"# [C{c}, chart{ch}] - Shape: ({H}, {W})\n")
                        for h in range(H):
                            f.write("  ")
                            for w in range(W):
                                f.write(f"{data[c, ch, h, w]:10.6f}  ")
                            f.write("\n")
                        f.write("\n")
                if C > 3:
                    f.write(f"# ... (省略其余 {C-3} 个通道)\n\n")
        
        elif data.ndim == 3:  # [C, H, W]
            C, H, W = data.shape
            for c in range(min(C, 5)):  # 只输出前5个通道
                f.write(f"# [Channel {c}] - Shape: ({H}, {W})\n")
                for h in range(H):
                    f.write("  ")
                    for w in range(W):
                        f.write(f"{data[c, h, w]:10.6f}  ")
                    f.write("\n")
                f.write("\n")
            if C > 5:
                f.write(f"# ... (省略其余 {C-5} 个通道)\n\n")
        
        elif data.ndim == 2:  # [H, W]
            H, W = data.shape
            for h in range(H):
                f.write("  ")
                for w in range(W):
                    f.write(f"{data[h, w]:10.6f}  ")
                f.write("\n")
        
        else:
            # 其他情况，flatten 输出
            flat = data.flatten()
            f.write("# Flattened data:\n")
            for i, val in enumerate(flat[:100]):  # 只输出前100个
                f.write(f"{val:12.8f}  ")
                if (i + 1) % 10 == 0:
                    f.write("\n")
            if len(flat) > 100:
                f.write(f"\n# ... (total {len(flat)} values)\n")
    
    print(f"  Saved: {filename}")


def debug_layer0():
    """
    调试 Layer0 的中间层计算
    """
    print("="*70)
    print("Layer0 中间层调试")
    print("="*70)
    
    # 1. 加载数据
    print("\n[1] Loading data...")
    input_data_raw = np.load('hls_testdata/layer0/input_rearranged.npy')
    
    # 检查形状，可能是 [B, C, R, T, V] 或 [T, C, R, charts, H, W]
    print(f"  Input shape (raw): {input_data_raw.shape}")
    
    # 如果是 reshape 过的格式 [B, C, R, T, V]，需要转回 [B, T, C, R, charts, H, W]
    if len(input_data_raw.shape) == 5 and input_data_raw.shape[-1] == 160:
        # [B, C, R, T, V] -> [B, T, C, R, charts, H, W]
        B, C, R, T, V = input_data_raw.shape
        charts, H, W = 5, 4, 8
        input_data = input_data_raw.reshape(B, C, R, T, charts, H, W)
        input_data = input_data.transpose(0, 3, 1, 2, 4, 5, 6)  # [B, T, C, R, charts, H, W]
        input_data = input_data.squeeze(0)  # [T, C, R, charts, H, W]
        print(f"  Reshaped to: {input_data.shape}")
    elif len(input_data_raw.shape) == 6:
        # 已经是 [T, C, R, charts, H, W] 格式
        input_data = input_data_raw
    elif len(input_data_raw.shape) == 7:
        # [B, T, C, R, charts, H, W] 格式
        input_data = input_data_raw.squeeze(0)
    else:
        raise ValueError(f"不支持的 input 形状: {input_data_raw.shape}")
    
    weight_data = np.load('hls_testdata/layer0/weight.npy')           # [32, 1, 1, 7]
    bias_data = np.load('hls_testdata/layer0/bias.npy')               # [32]
    kernel_exp_idx = np.load('hls_testdata/layer0/kernel_expansion_idx.npy')  # [32, 6, 1, 1, 9, 4]
    reorder_idx = np.load('hls_testdata/layer0/reorder_idx.npy')     # [1, 5, 6, 10]
    
    print(f"  Input shape: {input_data.shape}  # 期望: [T, C, R, charts, H, W]")
    print(f"  Weight shape: {weight_data.shape}")
    
    # 2. 提取第0帧
    frame0_input = torch.from_numpy(input_data[0:1]).float()  # [1, 1, 1, 5, 4, 8]
    # 注意：input_data 的形状是 [T, Cin, Rin, charts, H, W]
    # 我们需要提取为 [Cin, Rin, charts, H, W] 格式
    frame0_input = frame0_input.squeeze(0)  # [1, 1, 5, 4, 8] - 这是 [Cin, Rin, charts, H, W]
    
    save_tensor_as_matrix(frame0_input, 
                          os.path.join(DEBUG_DIR, 'py_frame0_input.txt'),
                          "Frame 0 Input [1, 1, 5, 4, 8]")
    
    # 3. 创建 ConvIco 层并加载权重
    print("\n[2] Creating ConvIco layer...")
    layer0 = ConvIco(r=2, Cin=1, Cout=32, Rin=1, Rout=6, smooth_vertices=True)
    layer0.weight.data = torch.from_numpy(weight_data).float()
    layer0.bias.data = torch.from_numpy(bias_data).float()
    layer0.eval()
    
    # 4. 逐步执行并保存中间结果
    print("\n[3] Running forward pass with intermediate outputs...")
    
    # 4.1 CleanVertices / SmoothVertices (在 ConvIco forward 中的第一步)
    # 注意：ConvIco 的 forward 期望输入是 [..., Cin, Rin, charts, H, W]
    # 而 process_vertices 期望输入是 [..., charts, H, W]
    # 在实际 forward 中，process_vertices 是在 padding 之后应用的
    # 所以我们先跳过这一步，直接从 padding 开始
    
    # 4.2 Padding
    x = frame0_input  # [Cin, Rin, charts, H, W] = [1, 1, 5, 4, 8]
    x_padded = layer0.padding(x)  # [Rin, charts, H_padded, W_padded] = [1, 5, 6, 10]
    save_tensor_as_matrix(x_padded,
                          os.path.join(DEBUG_DIR, 'py_frame0_padded.txt'),
                          "After PadIco [1, 5, 6, 10]")
    
    # 4.3 Reshape 为 2D 卷积格式
    import einops
    x_reshaped = einops.rearrange(x_padded, 'C R charts H W -> (C R) (charts H) W', 
                                   C=1, R=1, charts=5)  # [1, 30, 10]
    save_tensor_as_matrix(x_reshaped,
                          os.path.join(DEBUG_DIR, 'py_frame0_reshaped_input.txt'),
                          "Reshaped Input [1, 30, 10]")
    
    # 4.4 获取卷积核
    kernel = layer0.get_kernel()  # [32, 6, 1, 1, 3, 3]
    kernel_2d = einops.rearrange(kernel, 'Cout Rout Cin Rin Hk Wk -> (Cout Rout) (Cin Rin) Hk Wk',
                                  Hk=3, Wk=3)  # [192, 1, 3, 3]
    save_tensor_as_matrix(kernel_2d,
                          os.path.join(DEBUG_DIR, 'py_frame0_kernel.txt'),
                          "Kernel [192, 1, 3, 3]")
    
    # 4.5 Conv2D
    x_batch = x_reshaped.unsqueeze(0)  # [1, 1, 30, 10]
    bias_repeated = einops.repeat(layer0.bias, 'Cout -> (Cout Rout)', Cout=32, Rout=6)
    
    y_conv = torch.nn.functional.conv2d(x_batch, kernel_2d, bias_repeated, padding=(1, 1))
    y_conv = y_conv.squeeze(0)  # [192, 30, 10]
    save_tensor_as_matrix(y_conv,
                          os.path.join(DEBUG_DIR, 'py_frame0_conv2d_output.txt'),
                          "Conv2D Output [192, 30, 10]")
    
    # 4.6 Reshape 回 icosahedral 格式
    y_reshaped = einops.rearrange(y_conv, '(C R) (charts H) W -> C R charts H W',
                                   C=32, R=6, charts=5)  # [32, 6, 5, 6, 10]
    save_tensor_as_matrix(y_reshaped,
                          os.path.join(DEBUG_DIR, 'py_frame0_reshaped_output.txt'),
                          "Reshaped Output [32, 6, 5, 6, 10]")
    
    # 4.7 去 padding
    y_unpadded = y_reshaped[..., 1:-1, 1:-1]  # [32, 6, 5, 4, 8]
    save_tensor_as_matrix(y_unpadded,
                          os.path.join(DEBUG_DIR, 'py_frame0_unpadded.txt'),
                          "Unpadded Output [32, 6, 5, 4, 8]")
    
    # 4.8 最终 CleanVertices / SmoothVertices
    # process_vertices 期望输入 [..., charts, H, W]
    # y_unpadded 是 [Cout, Rout, charts, H, W]，正好符合
    y_final = layer0.process_vertices(y_unpadded)  # [32, 6, 5, 4, 8]
    save_tensor_as_matrix(y_final,
                          os.path.join(DEBUG_DIR, 'py_frame0_final_output.txt'),
                          "Final Output [32, 6, 5, 4, 8]")
    
    # 5. 对比完整前向传播结果
    print("\n[4] Comparing with full forward pass...")
    y_full = layer0(frame0_input)
    diff = (y_full - y_final).abs()
    print(f"  Diff Max: {diff.max():.10f}")
    print(f"  Diff Mean: {diff.mean():.10f}")
    
    if diff.max() < 1e-6:
        print("  ✓ 中间步骤与完整前向传播一致！")
    else:
        print("  ✗ 存在差异，需要检查中间步骤")
    
    # 6. 保存参考输出（来自 PyTorch 完整输出的第0帧）
    ref_output = np.load('hls_testdata/layer0/output_layer0.npy')  # [1, 52, 32, 6, 5, 4, 8]
    ref_frame0 = torch.from_numpy(ref_output[0, 0]).float()  # [32, 6, 5, 4, 8]
    save_tensor_as_matrix(ref_frame0,
                          os.path.join(DEBUG_DIR, 'py_frame0_reference.txt'),
                          "Reference Output (from full inference) [32, 6, 5, 4, 8]")
    
    print("\n" + "="*70)
    print(f"所有中间层数据已保存到: {DEBUG_DIR}/")
    print("="*70)

if __name__ == "__main__":
    debug_layer0()
