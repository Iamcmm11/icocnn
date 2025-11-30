import numpy as np
import matplotlib.pyplot as plt
import os

# 请在这里填写您的 .npy 文件路径
FILE_PATH = "1sourceTracking_icoCNN_robot_K4096_r2_predictions_rmsae.npy"  # 修改为您的文件路径

def load_and_display_npy(file_path):
    """
    加载并显示 .npy 文件内容
    
    Args:
        file_path (str): .npy 文件路径
        
    Returns:
        numpy.ndarray or None: 加载的数据或None（如果出错）
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return None
            
        print(f"正在加载文件: {file_path}")
        print(f"文件大小: {os.path.getsize(file_path)} bytes")
        
        # 尝试不同的加载方法
        try:
            # 方法1: 标准加载
            data = np.load(file_path)
            print("使用标准加载方法成功")
        except:
            try:
                # 方法2: 允许pickle
                data = np.load(file_path, allow_pickle=True)
                print("使用allow_pickle=True加载成功")
            except:
                # 方法3: 作为文本文件读取查看内容
                print("尝试以文本方式读取文件内容:")
                with open(file_path, 'rb') as f:
                    header = f.read(100)  # 读取前100字节
                    print(f"文件头: {header}")
                return None
        
        # 显示基本信息
        print(f"数据形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
        if hasattr(data, 'min') and hasattr(data, 'max'):
            print(f"数据范围: [{np.min(data):.4f}, {np.max(data):.4f}]")
            print(f"平均值: {np.mean(data):.4f}")
            print(f"标准差: {np.std(data):.4f}")
        
        # 显示完整数据
        print("\n完整数据内容:")
        print(data)
            
        return data
    except Exception as e:
        print(f"加载文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_data_as_heatmap(data, title="NPY Data Visualization"):
    """
    将数据绘制成热力图，行作为横坐标，列作为纵坐标
    
    Args:
        data (numpy.ndarray): 要绘制的数据（二维数组）
        title (str): 图表标题
    """
    try:
        if not isinstance(data, np.ndarray):
            print("数据不是numpy数组，无法绘制图表")
            return
            
        if len(data.shape) != 2:
            print("只能绘制二维数据")
            return
            
        # 创建图形和轴
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 创建热力图，行作为横坐标，列作为纵坐标
        im = ax.imshow(data, cmap='viridis', aspect='auto')
        
        # 设置坐标轴标签
        ax.set_xlabel('列索引 (Columns)')
        ax.set_ylabel('行索引 (Rows)')
        ax.set_title(title)
        
        # 设置刻度位置
        ax.set_xticks(range(data.shape[1]))
        ax.set_yticks(range(data.shape[0]))
        
        # 在每个格子上显示数值
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f'{data[i, j]:.2f}',
                        ha="center", va="center", color="white", fontsize=10)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('数值')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"绘制图表时出错: {e}")
        import traceback
        traceback.print_exc()

def plot_data_as_line(data, title="NPY Data Visualization"):
    """
    将每行数据绘制成折线图
    
    Args:
        data (numpy.ndarray): 要绘制的数据（二维数组）
        title (str): 图表标题
    """
    try:
        if not isinstance(data, np.ndarray):
            print("数据不是numpy数组，无法绘制图表")
            return
            
        if len(data.shape) != 2:
            print("只能绘制二维数据")
            return
            
        # 创建图形和轴
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 为每一行绘制一条线
        for i in range(data.shape[0]):
            ax.plot(range(data.shape[1]), data[i, :], marker='o', label=f'行 {i}')
        
        ax.set_xlabel('列索引 (Columns)')
        ax.set_ylabel('数值')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"绘制图表时出错: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    主函数
    """
    # 加载并显示数据
    data = load_and_display_npy(FILE_PATH)
    
    if data is not None:
        # 绘制热力图（行作为横坐标，列作为纵坐标）
        plot_data_as_heatmap(data, f"热力图 - Data from {FILE_PATH}")
        
        # 绘制折线图
        plot_data_as_line(data, f"折线图 - Data from {FILE_PATH}")

if __name__ == "__main__":
    main()