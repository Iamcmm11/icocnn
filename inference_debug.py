
import torch
import numpy as np
import os
import sys
import acousticTrackingModels as at_models

# 配置部分
MODEL_PATH = 'models/your_model_weights.bin'  # 请修改为您实际的权重文件路径
R_LEVEL = 2            # 模型的分辨率 r
CHANNELS = 32          # 模型通道数 C
SAVE_DIR = 'debug_outputs' # 输出数据的保存目录

# 确保保存目录存在
os.makedirs(SAVE_DIR, exist_ok=True)

def save_debug_tensor(name, tensor):
    """保存 Tensor 到文本文件和 .npy 文件，方便 HLS 对比"""
    # 转为 numpy
    if tensor.is_cuda:
        data = tensor.detach().cpu().numpy()
    else:
        data = tensor.detach().numpy()
    
    # 1. 保存 .npy (用于 Python 加载对比)
    np.save(os.path.join(SAVE_DIR, f"{name}.npy"), data)
    
    # 2. 保存部分数据到 .txt (用于人工查看或 HLS testbench 读取)
    # 扁平化并只保存前 100 个数，或者按特定格式保存
    flat_data = data.flatten()
    with open(os.path.join(SAVE_DIR, f"{name}_sample.txt"), 'w') as f:
        f.write(f"# Shape: {data.shape}\n")
        f.write(f"# Mean: {np.mean(data):.6f}, Max: {np.max(data):.6f}, Min: {np.min(data):.6f}\n")
        f.write("First 100 values:\n")
        for i, val in enumerate(flat_data[:100]):
            f.write(f"{val:.6f}\n")
    
    print(f"[-] Saved layer output: {name} | Shape: {data.shape}")

def get_activation_hook(name):
    """创建 Hook 函数"""
    def hook(model, input, output):
        # input 是一个 tuple，取第一个元素
        # output 是输出 tensor
        print(f"[+] Layer Forward: {name}")
        # 保存输出
        save_debug_tensor(f"{name}_output", output)
    return hook

def register_hooks(model):
    """为模型的关键层注册 Hooks"""
    print("\n--- Registering Hooks ---")
    
    # 1. 注册 IcoCNN 层的 Hook
    for i, layer in enumerate(model.ico_cnn):
        layer_name = f"Layer{i}_IcoConv"
        layer.register_forward_hook(get_activation_hook(layer_name))
        print(f"Registered hook for: {layer_name}")

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

if __name__ == "__main__":
    main()
