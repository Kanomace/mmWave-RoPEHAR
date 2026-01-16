import matplotlib

matplotlib.use('TkAgg')  # 使用TkAgg后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
import os
import re
from tqdm import tqdm


def natural_sort_key(s):
    """
    自然排序键函数，确保数字按数值大小排序
    例如: ['file1', 'file2', 'file10'] 而不是 ['file1', 'file10', 'file2']
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('(\d+)', s)]


def load_and_animate_voxels(folder_path, max_frames=100):
    """
    加载体素数据并创建动画

    参数:
    folder_path: 包含.npy文件的文件夹路径
    max_frames: 最大帧数（限制以避免内存问题）
    """
    # 获取所有npy文件并按自然顺序排序
    npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    npy_files.sort(key=natural_sort_key)  # 使用自然排序

    if not npy_files:
        print(f"在文件夹 {folder_path} 中未找到.npy文件")
        return

    # 限制帧数
    if len(npy_files) > max_frames:
        npy_files = npy_files[:max_frames]

    print(f"已加载 {len(npy_files)} 个文件，按自然排序:")
    for i, f in enumerate(npy_files[:10]):  # 显示前10个文件名
        print(f"  {i + 1}: {f}")
    if len(npy_files) > 10:
        print("  ...")

    # 创建图形和3D轴
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 设置固定视角 (elev=仰角, azim=方位角)
    ax.view_init(elev=30, azim=45)  # 固定视角

    # 设置坐标轴标签和范围
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 加载第一帧以获取数据形状
    first_data = np.load(os.path.join(folder_path, npy_files[0]))
    ax.set_xlim(0, first_data.shape[0])
    ax.set_ylim(0, first_data.shape[1])
    ax.set_zlim(0, first_data.shape[2])

    # 预加载所有数据以提高性能
    print("预加载数据...")
    all_data = []
    for file in tqdm(npy_files, desc="加载数据"):
        data = np.load(os.path.join(folder_path, file))
        all_data.append(data)

    # 更新函数
    def update(frame):
        ax.clear()
        data = all_data[frame]

        # 获取非零体素的坐标
        x, y, z = np.indices(data.shape)
        x, y, z = x[data > 0], y[data > 0], z[data > 0]

        # 创建散点图
        scatter = ax.scatter(x, y, z, c='red', marker='o', s=20, alpha=0.6)

        # 重新设置坐标轴
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(0, first_data.shape[0])
        ax.set_ylim(0, first_data.shape[1])
        ax.set_zlim(0, first_data.shape[2])

        # 设置固定视角
        ax.view_init(elev=30, azim=45)

        # 设置标题
        ax.set_title(f"帧: {frame + 1}/{len(npy_files)} - {npy_files[frame]}")

        return scatter,

    # 创建动画
    ani = animation.FuncAnimation(
        fig, update, frames=len(npy_files),
        interval=100, blit=False, repeat=True
    )

    # 显示动画
    plt.tight_layout()
    plt.show()


# 使用示例
folder_path = r'D:\Ti\Py_mmWave_Roformer\Dataset\huangzhouyang.walk\pHistBytes_clustered_voxel'
load_and_animate_voxels(folder_path, max_frames=5000)  # 限制为50帧测试