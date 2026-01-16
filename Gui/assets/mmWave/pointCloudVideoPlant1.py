import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def natural_sort_key(s):
    """
    自然排序的键函数
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', s)]


def animate_point_cloud_folder(folder_path, file_prefix):
    # 获取文件夹中符合前缀的点云数据文件
    file_list = [f for f in os.listdir(folder_path) if f.startswith(file_prefix) and f.endswith('.xlsx')]
    file_list.sort(key=natural_sort_key)  # 使用自然排序对文件名进行排序

    # 创建画布和子图
    fig = plt.figure()

    # 创建XOZ平面子图
    ax_xoz = fig.add_subplot(121)
    ax_xoz.set_xlabel('X')
    ax_xoz.set_ylabel('Z')

    # 创建YOZ平面子图
    ax_yoz = fig.add_subplot(122)
    ax_yoz.set_xlabel('Y')
    ax_yoz.set_ylabel('Z')

    # 创建体素网格
    voxel_size_xoz = 25
    voxel_size_yoz = (25, 15)  # 修改网格大小
    voxel_xoz = np.zeros((voxel_size_xoz, voxel_size_xoz))
    voxel_yoz = np.zeros(voxel_size_yoz)

    # 创建图例占位符
    legend = None

    def update(frame):
        nonlocal legend  # 使用nonlocal声明以便在函数内部修改外部的legend变量

        # 读取点云数据文件
        file_path = os.path.join(folder_path, file_list[frame])
        df = pd.read_excel(file_path)

        # 提取数据列
        x = df['X']
        y = df['Y']
        z = df['Z']
        snr = df['Noise']

        # 清空体素网格
        voxel_xoz.fill(0)
        voxel_yoz.fill(0)

        # 计算体素网格中每个网格单元的点云数量
        for xi, zi, yi in zip(x, z, y):
            if 2 <= xi <= 8 and 2 <= zi <= 8:
                xi_idx = int((xi - 2) * voxel_size_xoz / 6)
                zi_idx = int((zi - 2) * voxel_size_xoz / 6)
                voxel_xoz[zi_idx, xi_idx] += 1
            if 0.5 <= yi <= 2 and 2 <= zi <= 8:
                yi_idx = int((yi - 0.5) * voxel_size_yoz[0] / 1.5)
                zi_idx = int((zi - 2) * voxel_size_yoz[1] / 6)
                voxel_yoz[yi_idx, zi_idx] += 1

        # 绘制XOZ平面体素化效果
        ax_xoz.imshow(voxel_xoz, cmap='Blues', origin='lower', extent=[2, 8, 2, 8], vmin=0, vmax=np.max(voxel_xoz)*0.1)  # 调整颜色深度
        ax_xoz.set_title('XOZ Plane')

        # 绘制YOZ平面体素化效果
        ax_yoz.imshow(voxel_yoz, cmap='Blues', origin='lower', extent=[0, 2, 2, 8], vmin=0, vmax=np.max(voxel_yoz)*0.9)  # 调整颜色深度
        ax_yoz.set_title('YOZ Plane')

    # 设置坐标轴范围
    ax_xoz.set_xlim(2, 8)
    ax_xoz.set_ylim(2, 8)

    ax_yoz.set_xlim(0, 2)
    ax_yoz.set_ylim(2, 8)

    # 设置背景色为白色
    fig.patch.set_facecolor('white')
    ax_xoz.set_facecolor('white')
    ax_yoz.set_facecolor('white')

    # 创建动画
    num_frames = len(file_list)
    animation = FuncAnimation(fig, update, frames=num_frames, interval=100)

    # 显示动画
    plt.show()


# 播放点云数据文件夹
folder_path = r'C:\Users\Kano\Desktop\radar_toolbox_1_30_01_03\tools\visualizers\Industrial_Visualizer\binData\image\SNR\pHistBytes_clustered'
file_prefix = 'pHistBytes_'
animate_point_cloud_folder(folder_path, file_prefix)