"""
对比 Python 和 C++ 的中间层输出
快速定位差异来源
"""

import numpy as np
import os

DEBUG_DIR = 'hls_testdata/layer0/debug_intermediate'
CPP_DEBUG_DIR = 'hls_testdata/layer0/debug_intermediate_cpp'

def load_matrix_from_txt(filename):
    """从文本文件加载矩阵数据（跳过注释行）"""
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # 解析数字
            values = line.split()
            for val in values:
                try:
                    data.append(float(val))
                except:
                    pass
    return np.array(data)

def compare_files(py_file, cpp_file, name):
    """对比两个文件的数据"""
    print(f"\n{'='*70}")
    print(f"对比: {name}")
    print(f"{'='*70}")
    
    if not os.path.exists(py_file):
        print(f"  ✗ Python 文件不存在: {py_file}")
        return
    if not os.path.exists(cpp_file):
        print(f"  ✗ C++ 文件不存在: {cpp_file}")
        return
    
    py_data = load_matrix_from_txt(py_file)
    cpp_data = load_matrix_from_txt(cpp_file)
    
    print(f"  Python 数据: {len(py_data)} 个值")
    print(f"  C++ 数据:    {len(cpp_data)} 个值")
    
    if len(py_data) != len(cpp_data):
        print(f"  ✗ 数据长度不匹配!")
        return
    
    # 计算差异
    diff = np.abs(py_data - cpp_data)
    max_err = diff.max()
    rmse = np.sqrt(np.mean(diff ** 2))
    
    # 统计量对比
    print(f"\n  Python:  Min={py_data.min():.6f}, Max={py_data.max():.6f}, Mean={py_data.mean():.6f}")
    print(f"  C++:     Min={cpp_data.min():.6f}, Max={cpp_data.max():.6f}, Mean={cpp_data.mean():.6f}")
    
    print(f"\n  差异统计:")
    print(f"    Max Error: {max_err:.8f}")
    print(f"    RMSE:      {rmse:.8f}")
    print(f"    Mean Abs Error: {diff.mean():.8f}")
    
    if max_err < 1e-5:
        print(f"  ✓ PASS: 数据完全一致（误差 < 1e-5）")
    elif max_err < 1e-3:
        print(f"  ⚠ 小误差（1e-5 ~ 1e-3）")
    else:
        print(f"  ✗ FAIL: 存在显著差异（误差 > 1e-3）")
        
        # 找出最大误差的位置
        max_idx = diff.argmax()
        print(f"\n  最大误差位置: index={max_idx}")
        print(f"    Python: {py_data[max_idx]:.8f}")
        print(f"    C++:    {cpp_data[max_idx]:.8f}")
        print(f"    Diff:   {diff[max_idx]:.8f}")

def main():
    print("="*70)
    print("Layer0 中间层对比 - Python vs C++")
    print("="*70)
    
    # 定义要对比的文件对
    comparisons = [
        ("py_frame0_input.txt", "cpp_frame0_input.txt", "1. 输入 [1, 1, 5, 4, 8]"),
        ("py_frame0_padded.txt", "cpp_frame0_padded.txt", "2. Padding 后 [1, 5, 6, 10]"),
        ("py_frame0_reshaped_input.txt", "cpp_frame0_reshaped_input.txt", "3. Reshaped 输入 [1, 30, 10]"),
        ("py_frame0_final_output.txt", "cpp_frame0_final_output.txt", "4. 最终输出 [32, 6, 5, 4, 8]"),
    ]
    
    for py_name, cpp_name, desc in comparisons:
        py_file = os.path.join(DEBUG_DIR, py_name)
        cpp_file = os.path.join(CPP_DEBUG_DIR, cpp_name)
        compare_files(py_file, cpp_file, desc)
    
    print("\n" + "="*70)
    print("对比完成！")
    print("="*70)
    print("\n提示：")
    print("  - 如果 padding 后就出现差异，检查 reorder_idx 的索引计算")
    print("  - 如果卷积后出现差异，检查 kernel_expansion 和 conv2d 实现")
    print("  - 如果最终输出差异，检查 CleanVertices/SmoothVertices 逻辑")

if __name__ == "__main__":
    main()
