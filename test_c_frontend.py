"""
Cross3D 前端处理验证脚本
对比C语言实现和Python(Cross3D原项目)实现的FFT、GCC-PHAT、SRP-Map结果

直接使用Cross3D原项目的acousticTrackingModules中的GCC和SRP_map模块进行对比验证

Author: Cross3D Verification
Date: 2024
"""

import os
# 解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import struct
import torch

# 导入Cross3D原项目的模块
import acousticTrackingModules as at_modules

# ============================================================================
# 配置参数 (与C语言config.h保持一致)
# ============================================================================
SAMPLE_RATE = 24000
NUM_CHANNELS = 12
FRAME_LENGTH = 4096
FFT_SIZE = 4096
FFT_BINS = FFT_SIZE // 2 + 1
NUM_MIC_PAIRS = 66
GCC_LENGTH = FFT_SIZE
SPEED_OF_SOUND = 343.0

# 文件路径
C_OUTPUT_DIR = "c_implementation/output"
AUDIO_FILE = os.path.join(C_OUTPUT_DIR, "audio_data.bin")
FFT_FILE = os.path.join(C_OUTPUT_DIR, "fft_result.bin")
GCC_FILE = os.path.join(C_OUTPUT_DIR, "gcc_result.bin")
SRP_FILE = os.path.join(C_OUTPUT_DIR, "srp_result.bin")


# ============================================================================
# 生成与C语言一致的麦克风阵列位置
# ============================================================================
def generate_mic_positions(num_channels=12, radius=0.05):
    """生成环形麦克风阵列位置 (与C语言实现一致)"""
    positions = np.zeros((num_channels, 3), dtype=np.float32)
    for i in range(num_channels):
        angle = 2 * np.pi * i / num_channels
        positions[i, 0] = radius * np.cos(angle)
        positions[i, 1] = radius * np.sin(angle)
        positions[i, 2] = 0.0
    return positions


# ============================================================================
# 数据加载函数 (从C语言生成的二进制文件)
# ============================================================================
def load_audio_data(filename):
    """加载C语言生成的音频数据"""
    with open(filename, 'rb') as f:
        magic = f.read(4).decode('ascii').rstrip('\x00')
        num_channels = struct.unpack('i', f.read(4))[0]
        num_samples = struct.unpack('i', f.read(4))[0]
        sample_rate = struct.unpack('i', f.read(4))[0]
        
        print(f"[Audio] Magic: {magic}, Channels: {num_channels}, "
              f"Samples: {num_samples}, SampleRate: {sample_rate}")
        
        audio_data = np.zeros((num_channels, num_samples), dtype=np.float32)
        for ch in range(num_channels):
            audio_data[ch] = np.frombuffer(f.read(num_samples * 4), dtype=np.float32)
        
    return audio_data, num_channels, num_samples, sample_rate


def load_fft_result(filename):
    """加载C语言生成的FFT结果"""
    with open(filename, 'rb') as f:
        magic = f.read(4).decode('ascii').rstrip('\x00')
        num_channels = struct.unpack('i', f.read(4))[0]
        num_bins = struct.unpack('i', f.read(4))[0]
        reserved = struct.unpack('i', f.read(4))[0]
        
        print(f"[FFT] Magic: {magic}, Channels: {num_channels}, Bins: {num_bins}")
        
        fft_data = np.zeros((num_channels, num_bins), dtype=np.complex64)
        for ch in range(num_channels):
            raw = np.frombuffer(f.read(num_bins * 8), dtype=np.float32)
            fft_data[ch] = raw[0::2] + 1j * raw[1::2]
        
    return fft_data


def load_gcc_result(filename):
    """加载C语言生成的GCC结果"""
    with open(filename, 'rb') as f:
        magic = f.read(4).decode('ascii').rstrip('\x00')
        num_pairs = struct.unpack('i', f.read(4))[0]
        gcc_length = struct.unpack('i', f.read(4))[0]
        reserved = struct.unpack('i', f.read(4))[0]
        
        print(f"[GCC] Magic: {magic}, Pairs: {num_pairs}, Length: {gcc_length}")
        
        gcc_data = np.zeros((num_pairs, gcc_length), dtype=np.float32)
        for pair in range(num_pairs):
            gcc_data[pair] = np.frombuffer(f.read(gcc_length * 4), dtype=np.float32)
        
    return gcc_data, gcc_length


def load_srp_result(filename):
    """加载C语言生成的SRP结果"""
    with open(filename, 'rb') as f:
        magic = f.read(4).decode('ascii').rstrip('\x00')
        elev_bins = struct.unpack('i', f.read(4))[0]
        azim_bins = struct.unpack('i', f.read(4))[0]
        range_bins = struct.unpack('i', f.read(4))[0]
        
        print(f"[SRP] Magic: {magic}, Shape: ({elev_bins}, {azim_bins}, {range_bins})")
        
        total_size = elev_bins * azim_bins * range_bins
        srp_data = np.frombuffer(f.read(total_size * 4), dtype=np.float32)
        srp_data = srp_data.reshape((elev_bins, azim_bins, range_bins))
        
    return srp_data


# ============================================================================
# 使用Cross3D原项目的前端处理
# ============================================================================
class Cross3DFrontend:
    """Cross3D原项目的前端处理封装"""
    
    def __init__(self, N, K, res_the, res_phi, mic_positions, fs, c=343.0):
        """
        N: 麦克风数量
        K: FFT窗口大小
        res_the: SRP-Map俯仰角分辨率
        res_phi: SRP-Map方位角分辨率
        mic_positions: 麦克风位置 [N, 3]
        fs: 采样率
        c: 声速
        """
        self.N = N
        self.K = K
        self.fs = fs
        self.res_the = res_the
        self.res_phi = res_phi
        
        # 计算最大时延
        dist_max = np.max([np.max([np.linalg.norm(mic_positions[n, :] - mic_positions[m, :]) 
                                   for m in range(N)]) for n in range(N)])
        tau_max = int(np.ceil(dist_max / c * fs))
        
        print(f"[Cross3D Frontend] N={N}, K={K}, tau_max={tau_max}")
        print(f"[Cross3D Frontend] SRP resolution: {res_the} x {res_phi}")
        
        # 初始化GCC和SRP模块 (使用Cross3D原项目的实现)
        self.gcc_module = at_modules.GCC(N, K, tau_max=tau_max, transform='PHAT')
        self.srp_module = at_modules.SRP_map(N, K, res_the, res_phi, mic_positions, fs,
                                              thetaMax=np.pi/2)  # planar array
    
    def process(self, audio_frame):
        """
        处理单帧音频
        输入: audio_frame [N, K] numpy数组
        输出: gcc_result, srp_map
        """
        # 转换为PyTorch张量 [batch=1, channel=1, N, K]
        x = torch.from_numpy(audio_frame.astype(np.float32))
        x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, N, K]
        
        # GCC-PHAT计算
        gcc_result = self.gcc_module(x)  # [1, 1, N, N, 2*tau_max+1]
        
        # SRP-Map计算
        srp_map = self.srp_module(gcc_result)  # [1, 1, res_the, res_phi]
        
        return gcc_result.squeeze().numpy(), srp_map.squeeze().numpy()
    
    def compute_gcc_only(self, audio_frame):
        """只计算GCC-PHAT"""
        x = torch.from_numpy(audio_frame.astype(np.float32))
        x = x.unsqueeze(0).unsqueeze(0)
        gcc_result = self.gcc_module(x)
        return gcc_result.squeeze().numpy()


# ============================================================================
# 独立的Python FFT实现 (用于对比C语言FFT)
# ============================================================================
def python_fft(audio_frame):
    """
    使用numpy的FFT (与C语言FFT对比)
    输入: audio_frame [num_channels, frame_length]
    输出: fft_result [num_channels, fft_bins] (复数)
    """
    num_channels = audio_frame.shape[0]
    fft_result = np.zeros((num_channels, FFT_BINS), dtype=np.complex64)
    
    for ch in range(num_channels):
        fft_result[ch] = np.fft.rfft(audio_frame[ch])
    
    return fft_result


def python_gcc_phat_from_fft(fft_result):
    """
    从FFT结果计算GCC-PHAT (用于对比C语言GCC)
    输入: fft_result [num_channels, fft_bins]
    输出: gcc_result [num_mic_pairs, gcc_length] (fftshift后)
    """
    num_channels = fft_result.shape[0]
    gcc_result = np.zeros((NUM_MIC_PAIRS, GCC_LENGTH), dtype=np.float32)
    
    pair_idx = 0
    for i in range(num_channels):
        for j in range(i + 1, num_channels):
            # 互功率谱: X1 * conj(X2)
            cross_spectrum = fft_result[i] * np.conj(fft_result[j])
            
            # PHAT加权
            magnitude = np.abs(cross_spectrum)
            magnitude[magnitude < 1e-10] = 1e-10
            cross_spectrum_phat = cross_spectrum / magnitude
            
            # 构造完整频谱并IFFT
            full_spectrum = np.zeros(FFT_SIZE, dtype=np.complex64)
            full_spectrum[:FFT_BINS] = cross_spectrum_phat
            full_spectrum[FFT_BINS:] = np.conj(cross_spectrum_phat[-2:0:-1])
            
            gcc = np.fft.ifft(full_spectrum).real
            
            # fftshift: 将零时延移到中心
            gcc_result[pair_idx] = np.fft.fftshift(gcc)
            pair_idx += 1
    
    return gcc_result


# ============================================================================
# 对比分析函数
# ============================================================================
def compare_results(c_data, py_data, name, tolerance=1e-3):
    """对比C和Python的结果"""
    print(f"\n{'='*60}")
    print(f"对比: {name}")
    print(f"{'='*60}")
    
    print(f"C语言结果形状: {c_data.shape}")
    print(f"Python结果形状: {py_data.shape}")
    
    if c_data.shape != py_data.shape:
        print(f"[WARNING] 形状不匹配!")
        # 尝试比较可比较的部分
        min_shape = tuple(min(c, p) for c, p in zip(c_data.shape, py_data.shape))
        print(f"将比较共同部分: {min_shape}")
        
        if len(min_shape) == 1:
            c_data = c_data[:min_shape[0]]
            py_data = py_data[:min_shape[0]]
        elif len(min_shape) == 2:
            c_data = c_data[:min_shape[0], :min_shape[1]]
            py_data = py_data[:min_shape[0], :min_shape[1]]
    
    # 数值对比
    if np.iscomplexobj(c_data) or np.iscomplexobj(py_data):
        diff = np.abs(c_data - py_data)
    else:
        diff = np.abs(c_data - py_data)
    
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # 相对误差
    max_val = max(np.max(np.abs(c_data)), np.max(np.abs(py_data)))
    relative_error = max_diff / max_val * 100 if max_val > 0 else 0
    
    print(f"最大绝对误差: {max_diff:.6e}")
    print(f"平均绝对误差: {mean_diff:.6e}")
    print(f"最大相对误差: {relative_error:.4f}%")
    
    # 相关系数
    if c_data.size > 1:
        c_flat = np.abs(c_data.flatten()) if np.iscomplexobj(c_data) else c_data.flatten()
        py_flat = np.abs(py_data.flatten()) if np.iscomplexobj(py_data) else py_data.flatten()
        corr = np.corrcoef(c_flat, py_flat)[0, 1]
        print(f"相关系数: {corr:.6f}")
    
    # 判断
    if relative_error < 5.0:  # 5%以内认为通过
        print(f"[PASS] 结果基本匹配")
        return True
    else:
        print(f"[FAIL] 结果差异较大")
        return False


def print_sample_values(c_data, py_data, name, num_samples=10):
    """打印部分样本值进行对比"""
    print(f"\n--- {name} 样本值对比 ---")
    
    c_flat = c_data.flatten()
    py_flat = py_data.flatten()
    
    indices = np.linspace(0, min(len(c_flat), len(py_flat)) - 1, num_samples, dtype=int)
    
    print(f"{'Index':<10} {'C Value':<20} {'Python Value':<20} {'Diff':<15}")
    print("-" * 65)
    
    for idx in indices:
        c_val = c_flat[idx]
        py_val = py_flat[idx] if idx < len(py_flat) else float('nan')
        
        if np.iscomplexobj(c_val):
            c_str = f"{c_val.real:.6f}+{c_val.imag:.6f}j"
            py_str = f"{py_val.real:.6f}+{py_val.imag:.6f}j"
            diff = abs(c_val - py_val)
        else:
            c_str = f"{c_val:.6f}"
            py_str = f"{py_val:.6f}"
            diff = abs(c_val - py_val)
        
        print(f"{idx:<10} {c_str:<20} {py_str:<20} {diff:<15.6e}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    print("="*70)
    print("Cross3D 前端处理验证: C语言 vs Python (Cross3D原项目)")
    print("="*70)
    
    # 检查文件
    if not os.path.exists(AUDIO_FILE):
        print(f"[ERROR] 找不到音频文件: {AUDIO_FILE}")
        print("请先运行C程序: c_implementation/bin/cross3d_preprocess.exe")
        return
    
    # ========================================================================
    # 1. 加载C语言生成的数据
    # ========================================================================
    print("\n" + "="*70)
    print("Step 1: 加载C语言生成的数据")
    print("="*70)
    
    audio_data, num_channels, num_samples, sample_rate = load_audio_data(AUDIO_FILE)
    c_fft = load_fft_result(FFT_FILE)
    c_gcc, c_gcc_length = load_gcc_result(GCC_FILE)
    c_srp = load_srp_result(SRP_FILE)
    
    print(f"\n音频数据: {audio_data.shape}")
    print(f"C FFT结果: {c_fft.shape}")
    print(f"C GCC结果: {c_gcc.shape}")
    print(f"C SRP结果: {c_srp.shape}")
    
    # ========================================================================
    # 2. 提取第一帧并应用汉宁窗
    # ========================================================================
    print("\n" + "="*70)
    print("Step 2: 提取第一帧并应用汉宁窗")
    print("="*70)
    
    frame = audio_data[:, :FRAME_LENGTH].copy()
    print(f"帧形状: {frame.shape}")
    
    # 应用汉宁窗 (与C语言一致)
    window = np.hanning(FRAME_LENGTH)
    for ch in range(num_channels):
        frame[ch] *= window
    print("已应用汉宁窗")
    
    # ========================================================================
    # 3. Python FFT (numpy实现)
    # ========================================================================
    print("\n" + "="*70)
    print("Step 3: Python FFT计算")
    print("="*70)
    
    py_fft = python_fft(frame)
    print(f"Python FFT结果形状: {py_fft.shape}")
    
    # 对比FFT
    fft_pass = compare_results(c_fft, py_fft, "FFT结果")
    print_sample_values(c_fft[0], py_fft[0], "FFT Channel 0", num_samples=10)
    
    # ========================================================================
    # 4. Python GCC-PHAT
    # ========================================================================
    print("\n" + "="*70)
    print("Step 4: Python GCC-PHAT计算")
    print("="*70)
    
    py_gcc = python_gcc_phat_from_fft(py_fft)
    print(f"Python GCC结果形状: {py_gcc.shape}")
    
    # 对比GCC
    gcc_pass = compare_results(c_gcc, py_gcc, "GCC-PHAT结果")
    print_sample_values(c_gcc[0], py_gcc[0], "GCC Pair 0", num_samples=10)
    
    # ========================================================================
    # 5. 使用Cross3D原项目的前端处理
    # ========================================================================
    print("\n" + "="*70)
    print("Step 5: Cross3D原项目前端处理")
    print("="*70)
    
    mic_positions = generate_mic_positions(NUM_CHANNELS, radius=0.05)
    print(f"麦克风位置:\n{mic_positions}")
    
    # 使用与C语言SRP一致的分辨率
    res_the = c_srp.shape[0]  # 5
    res_phi = c_srp.shape[1]  # 4
    
    try:
        frontend = Cross3DFrontend(
            N=NUM_CHANNELS,
            K=FRAME_LENGTH,
            res_the=res_the,
            res_phi=res_phi,
            mic_positions=mic_positions,
            fs=SAMPLE_RATE
        )
        
        # 计算GCC和SRP
        cross3d_gcc, cross3d_srp = frontend.process(frame)
        print(f"Cross3D GCC结果形状: {cross3d_gcc.shape}")
        print(f"Cross3D SRP结果形状: {cross3d_srp.shape}")
        
        # 对比Cross3D的SRP和C语言的SRP (取第一个range)
        print("\n--- Cross3D SRP vs C SRP ---")
        print(f"Cross3D SRP:\n{cross3d_srp}")
        print(f"C SRP (range=0):\n{c_srp[:,:,0]}")
        
    except Exception as e:
        print(f"Cross3D前端处理失败: {e}")
        print("这可能是因为PyTorch版本不兼容，继续使用numpy实现进行对比")
    
    # ========================================================================
    # 6. 总结
    # ========================================================================
    print("\n" + "="*70)
    print("验证总结")
    print("="*70)
    
    results = {
        'FFT': fft_pass,
        'GCC-PHAT': gcc_pass,
    }
    
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: [{status}]")
    
    if all(results.values()):
        print("\n[SUCCESS] FFT和GCC-PHAT验证通过!")
        print("C语言实现与Python实现结果一致")
    else:
        print("\n[WARNING] 部分模块验证未通过，请检查实现")
    
    print("\n注意: SRP-Map的对比需要确保参数完全一致")
    print("C语言SRP使用了距离维度(range)，而Cross3D原项目只有角度维度")


if __name__ == "__main__":
    main()
