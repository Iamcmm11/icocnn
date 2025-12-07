
import torch
import numpy as np
import os
import sys
import acousticTrackingModels as at_models

# 配置部分
MODEL_PATH = 'models\1sourceTracking_icoCNN_robot_K4096_r2_model.bin'  # 请修改为您实际的权重文件路径
R_LEVEL = 2            # 模型的分辨率 r
CHANNELS = 32          # 模型通道数 C
SAVE_DIR = 'debug_outputs' # 输出数据的保存目录
HLS_DATA_DIR = 'hls_testdata/layer0'  # HLS 验证数据目录

# 确保保存目录存在
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(HLS_DATA_DIR, exist_ok=True)

def extract_layer0_data(model):
    """
    提取 Layer0 (第一个 IcoConv 层) 的权重、偏置、邻居索引
    保存为 HLS C++ testbench 可以读取的格式
    """
    print("\nExtracting Layer0 parameters...")
    
    # 获取第一个 IcoConv 层
    from icoCNN.icoCNN import ConvIco
    
    layer0 = None
    for module in model.modules():
        if isinstance(module, ConvIco):
            layer0 = module
            break
    
    if layer0 is None:
        print("Error: Could not find ConvIco layer in the model!")
        return
    
    print(f"  Found Layer0: {layer0.__class__.__name__}")
    
    # 1. 提取权重 [out_channels, in_channels, Rin, num_neighbors]
    # ConvIco 的权重是 [Cout, Cin, Rin, 7]
    weight = layer0.weight.detach().cpu().numpy()
    print(f"  Weight shape: {weight.shape}")
    
    # 对于 HLS,我们需要将权重重新组织
    # 从 [Cout, Cin, Rin, 7] 转换为 [Cout, Cin*Rin, 7]
    # 但首先检查 Rin 的值
    Cout, Cin, Rin, num_neighbors = weight.shape
    print(f"  Cout={Cout}, Cin={Cin}, Rin={Rin}, num_neighbors={num_neighbors}")
    
    # 重新组织权重: [Cout, Cin*Rin, 7]
    weight_reshaped = weight.reshape(Cout, Cin * Rin, num_neighbors)
    print(f"  Weight reshaped: {weight_reshaped.shape}")
    
    # 2. 提取偏置 [out_channels]
    if layer0.bias is not None:
        bias = layer0.bias.detach().cpu().numpy()
    else:
        bias = np.zeros(Cout)
    print(f"  Bias shape: {bias.shape}")
    
    # 3. 生成邻居索引
    # 对于 icosahedral grid (r=2),我们需要计算每个顶点的邻居
    # 从 padding 层的 reorder_idx 中提取
    print(f"  Generating neighbors for r={layer0.r}...")
    neighbors = generate_ico_neighbors(layer0.r)
    print(f"  Neighbors shape: {neighbors.shape}")
    
    # 获取网格结构信息
    H = 2**layer0.r
    W = 2**(layer0.r + 1)
    num_charts = 5
    
    # 4. 保存为 HLS 可读格式
    # 保存权重
    weight_flat = weight_reshaped.flatten()
    with open(os.path.join(HLS_DATA_DIR, 'weight.txt'), 'w') as f:
        f.write(f"# Shape: {weight_reshaped.shape}\n")
        for val in weight_flat:
            f.write(f"{val:.8f}\n")
    print(f"  ✓ Saved weight.txt: {len(weight_flat)} values")
    
    # 保存偏置
    with open(os.path.join(HLS_DATA_DIR, 'bias.txt'), 'w') as f:
        f.write(f"# Shape: {bias.shape}\n")
        for val in bias:
            f.write(f"{val:.8f}\n")
    print(f"  ✓ Saved bias.txt: {len(bias)} values")
    
    # 保存邻居索引 (整数)
    neighbors_flat = neighbors.flatten()
    with open(os.path.join(HLS_DATA_DIR, 'neighbors.txt'), 'w') as f:
        f.write(f"# Shape: {neighbors.shape}\n")
        for val in neighbors_flat:
            f.write(f"{int(val)}\n")
    print(f"  ✓ Saved neighbors.txt: {len(neighbors_flat)} indices")
    
    # 同时保存 .npy 格式
    np.save(os.path.join(HLS_DATA_DIR, 'weight.npy'), weight_reshaped)
    np.save(os.path.join(HLS_DATA_DIR, 'bias.npy'), bias)
    np.save(os.path.join(HLS_DATA_DIR, 'neighbors.npy'), neighbors)
    
    print(f"\n✓ Layer0 parameters saved to '{HLS_DATA_DIR}/'")
    print("  You can now compile and run the HLS C++ verification!")
    
    # 添加配置信息
    print(f"\n  Configuration for HLS:")
    print(f"    Input channels (Cin*Rin): {Cin * Rin}")
    print(f"    Output channels (Cout): {Cout}")
    print(f"    Neighbors per vertex: {num_neighbors}")
    print(f"    Grid structure: {num_charts} charts x {H}x{W}")

def generate_ico_neighbors(r):
    """
    生成 icosahedral grid 的邻居索引表
    基于 icoCNN 中 PoolIco 的邻居定义
    
    对于卷积操作,每个顶点有 7 个邻居 (包括自己)
    这对应 3x3 卷积核在 icosahedral grid 上的映射
    
    返回: [num_vertices, 7] 的索引数组
    """
    import torch
    
    H = 2**r  # 高度
    W = 2**(r+1)  # 宽度
    num_charts = 5  # icosahedral grid 有 5 个 chart
    
    # 从 icoCNN PoolIco 的邻居定义提取
    # 对于每个位置 (h, w),7个邻居的相对位置为:
    # 中心: (h, w)
    # 6个邻居: 按照 icosahedral 网格的六边形拓扑
    
    # 生成每个 chart 上每个顶点的邻居
    # 总顶点数: 5 * H * W,但有些顶点是重复的 (在边界上)
    # 实际独立顶点数: 10*r^2 + 2
    
    # 先生成 padding 后的邻居索引 (padded grid)
    # 然后映射回原始 grid
    
    # 这里我们使用和 PoolIco 相同的邻居模式
    neighbors_2d = torch.zeros((H, W, 7, 2), dtype=torch.long)
    
    for h in range(H):
        for w in range(W):
            # icosahedral 六边形网格的 7 个邻居 (包括中心)
            # 按照 PoolIco 中的定义
            neighbors_2d[h, w, ...] = torch.tensor([
                [h,   w  ],  # 中心
                [h+1, w  ],  # 下
                [h+1, w+1],  # 右下
                [h,   w+1],  # 右
                [h-1, w  ],  # 上
                [h-1, w-1],  # 左上
                [h,   w-1],  # 左
            ])
    
    # 将 2D 邻居索引转换为 1D 索引
    # 对于单个 chart: index = h * W + w
    num_vertices_per_chart = H * W
    
    # 生成所有 chart 的邻居关系
    all_neighbors = []
    
    for chart_idx in range(num_charts):
        for h in range(H):
            for w in range(W):
                vertex_neighbors = []
                
                for n in range(7):
                    nh, nw = neighbors_2d[h, w, n].tolist()
                    
                    # 处理边界情况 (循环边界)
                    nh = nh % H
                    nw = nw % W
                    
                    # 转换为 1D 索引 (在当前 chart 内)
                    neighbor_1d = nh * W + nw
                    
                    # 加上 chart 偏移
                    neighbor_global = chart_idx * num_vertices_per_chart + neighbor_1d
                    vertex_neighbors.append(neighbor_global)
                
                all_neighbors.append(vertex_neighbors)
    
    # 转换为 numpy 数组
    neighbors = np.array(all_neighbors, dtype=np.int32)
    
    # 注意: 这里生成的是 5*H*W = 5*4*8 = 160 个顶点
    # 但实际上 icosahedral grid 只有 42 个独立顶点
    # 这是因为边界上的顶点在不同 chart 之间是共享的
    
    # 但对于卷积计算,我们可以使用这个扩展的表示
    # 只要输入数据也是按照 [charts, H, W] 排列的
    
    print(f"  Generated neighbors: {neighbors.shape} (expanded representation)")
    print(f"  This represents {num_charts} charts x {H}x{W} = {num_charts * H * W} positions")
    
    return neighbors

def move_layer0_io_data():
    """
    将 Layer0 的输入和输出数据从 debug_outputs 移动到 hls_testdata/layer0/
    并修改为 HLS 所需的格式
    """
    print("\nMoving Layer0 input/output to HLS data directory...")
    
    import shutil
    
    # Layer0 的输入是整个网络的输入经过预处理后的
    # 但 IcoTempCNN 的第一层是 icosahedral 投影，我们需要投影后的结果
    # 这个是 Layer0_IcoConv_input.txt
    
    src_input = os.path.join(SAVE_DIR, 'Layer0_IcoConv_input.txt')
    src_output = os.path.join(SAVE_DIR, 'Layer0_IcoConv_output.txt')
    
    dst_input = os.path.join(HLS_DATA_DIR, 'input.txt')
    dst_output = os.path.join(HLS_DATA_DIR, 'output.txt')
    
    # 检查文件是否存在
    if os.path.exists(src_input):
        shutil.copy(src_input, dst_input)
        print(f"  ✓ Copied input: {src_input} -> {dst_input}")
    else:
        print(f"  ✗ Warning: {src_input} not found!")
    
    if os.path.exists(src_output):
        shutil.copy(src_output, dst_output)
        print(f"  ✓ Copied output: {src_output} -> {dst_output}")
    else:
        print(f"  ✗ Warning: {src_output} not found!")
    
    # 也复制 .npy 文件
    src_input_npy = os.path.join(SAVE_DIR, 'Layer0_IcoConv_input.npy')
    src_output_npy = os.path.join(SAVE_DIR, 'Layer0_IcoConv_output.npy')
    
    if os.path.exists(src_input_npy):
        shutil.copy(src_input_npy, os.path.join(HLS_DATA_DIR, 'input.npy'))
    if os.path.exists(src_output_npy):
        shutil.copy(src_output_npy, os.path.join(HLS_DATA_DIR, 'output.npy'))
    
    print(f"\n✓ All Layer0 data ready in '{HLS_DATA_DIR}/'")
    print("\n=" * 60)
    print("HLS Verification Files Summary:")
    print("=" * 60)
    for fname in ['input.txt', 'weight.txt', 'bias.txt', 'neighbors.txt', 'output.txt']:
        fpath = os.path.join(HLS_DATA_DIR, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath) / 1024
            # 计算行数
            with open(fpath, 'r') as f:
                lines = len(f.readlines())
            print(f"  ✓ {fname:20s} - {size:8.2f} KB - {lines:6d} lines")
        else:
            print(f"  ✗ {fname:20s} - MISSING!")
    print("=" * 60)

def save_debug_tensor(name, tensor, save_full=False, save_dir=SAVE_DIR):
    """保存 Tensor 到文本文件和 .npy 文件，方便 HLS 对比"""
    # 转为 numpy
    if isinstance(tensor, torch.Tensor):
        if tensor.is_cuda:
            data = tensor.detach().cpu().numpy()
        else:
            data = tensor.detach().numpy()
    else:
        data = tensor  # 已经是 numpy 数组
    
    # 1. 保存 .npy (用于 Python 加载对比)
    np.save(os.path.join(save_dir, f"{name}.npy"), data)
    
    # 2. 保存部分数据到 .txt (用于人工查看)
    flat_data = data.flatten()
    with open(os.path.join(save_dir, f"{name}_sample.txt"), 'w') as f:
        f.write(f"# Shape: {data.shape}\n")
        f.write(f"# Mean: {np.mean(data):.6f}, Max: {np.max(data):.6f}, Min: {np.min(data):.6f}\n")
        f.write("First 100 values:\n")
        for i, val in enumerate(flat_data[:100]):
            f.write(f"{val:.6f}\n")
    
    # 3. 如果需要保存完整数据 (用于 HLS C++ testbench)
    if save_full:
        with open(os.path.join(save_dir, f"{name}.txt"), 'w') as f:
            f.write(f"# Shape: {data.shape}\n")
            for val in flat_data:
                f.write(f"{val:.8f}\n")
        print(f"[-] Saved FULL data: {name} | Shape: {data.shape} | {len(flat_data)} values")
    else:
        print(f"[-] Saved sample: {name} | Shape: {data.shape}")

def get_activation_hook(name, save_input=False, save_full=False):
    """创建 Hook 函数"""
    def hook(model, input, output):
        # input 是一个 tuple，取第一个元素
        # output 是输出 tensor
        print(f"[+] Layer Forward: {name}")
        
        # 保存输入 (如果需要)
        if save_input and len(input) > 0:
            save_debug_tensor(f"{name}_input", input[0], save_full=save_full)
        
        # 保存输出
        save_debug_tensor(f"{name}_output", output, save_full=save_full)
    return hook

def register_hooks(model):
    """为模型的关键层注册 Hooks"""
    print("\n--- Registering Hooks ---")
    
    # 1. 注册 IcoCNN 层的 Hook
    # Layer0 需要保存完整的输入和输出
    for i, layer in enumerate(model.ico_cnn):
        layer_name = f"Layer{i}_IcoConv"
        # Layer0 保存完整数据
        save_full = (i == 0)
        save_input = (i == 0)
        layer.register_forward_hook(get_activation_hook(layer_name, save_input=save_input, save_full=save_full))
        print(f"Registered hook for: {layer_name} (save_full={save_full})")

    # 2. 注册 Temporal CNN 层的 Hook
    for i, layer in enumerate(model.temp_cnn):
        layer_name = f"Layer{i}_TempConv"
        layer.register_forward_hook(get_activation_hook(layer_name))
        print(f"Registered hook for: {layer_name}")

    # 3. 注册 LayerNorm 层的 Hook
    for i, layer in enumerate(model.layer_norm):
        layer_name = f"Layer{i}_LayerNorm"
        layer.register_forward_hook(get_activation_hook(layer_name))
        print(f"Registered hook for: {layer_name}")

    # 4. 注册 Pooling 层的 Hook
    for i, layer in enumerate(model.poolings):
        layer_name = f"Layer{i*2+1}_Pooling" # Pooling 通常发生在奇数层后
        layer.register_forward_hook(get_activation_hook(layer_name))
        print(f"Registered hook for: {layer_name}")
    
    # 5. 最后的 SoftArgMax
    model.sam.register_forward_hook(get_activation_hook("Final_SoftArgMax"))
    print("--- Hooks Registered ---\n")

def main():
    # 0. 固定随机种子，确保每次运行输入一致，方便 HLS 对账
    torch.manual_seed(42)
    np.random.seed(42)
    print("Random Seed set to 42 for reproducibility.")

    # 1. 初始化模型
    print("Initializing Model...")
    # 注意：如果您的模型训练时用了 smooth_vertices=True/False，这里要一致
    net = at_models.IcoTempCNN(r=R_LEVEL, C=CHANNELS, smooth_vertices=True)
    
    # 尝试加载权重 (如果文件存在)
    if os.path.exists(MODEL_PATH):
        print(f"Loading weights from {MODEL_PATH}...")
        try:
            # map_location='cpu' 确保在没有 GPU 的机器上也能跑
            state_dict = torch.load(MODEL_PATH, map_location='cpu')
            net.load_state_dict(state_dict)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Running with random initialization instead.")
    else:
        print(f"Weights file not found at {MODEL_PATH}. Running with random weights.")

    net.eval() # 设置为推理模式

    # 2. 注册 Hooks
    register_hooks(net)

    # 3. 构造输入数据
    # 形状参考之前的分析: (Batch, Channels, Time, Charts, H, W)
    # B=1, C=1 (Cin), T=103, Charts=5, H=2^r, W=2^(r+1)
    # 对于 r=2: H=4, W=8
    H = 2**R_LEVEL
    W = 2**(R_LEVEL + 1)
    T = 103 # 典型的帧数
    
    # 随机输入 (模拟 SRP-PHAT map)
    dummy_input = torch.randn(1, 1, T, 5, H, W)
    print(f"\nInput Shape: {dummy_input.shape}")
    save_debug_tensor("input", dummy_input)

    # 4. 执行推理
    print("\n--- Starting Inference ---")
    with torch.no_grad():
        output = net(dummy_input)
    print("--- Inference Finished ---\n")

    print(f"Final Output Shape: {output.shape}")
    # 输出通常是 (Batch, Time, 3) -> (1, 103, 3)
    save_debug_tensor("final_output", output)
    
    print(f"\nAll intermediate outputs saved to '{SAVE_DIR}/'")
    
    # 5. 提取 Layer0 的权重和邻居索引，保存到 HLS 验证目录
    print("\n--- Extracting Layer0 Data for HLS Verification ---")
    extract_layer0_data(net)
    
    # 6. 移动 Layer0 的输入和输出到 HLS 目录
    move_layer0_io_data()

if __name__ == "__main__":
    main()
