import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 定义体素网格大小
GRID_SIZE = (50, 25, 50)  # (x, y, z)
# 定义点云边界
BOUNDARIES = {
    'x': (-5, 5),   # X轴范围为 -10 到 10 米
    'y': (0, 5),   # Y轴范围为 0 到 10 米
    'z': (-5, 5)    # Z轴范围为 -10 到 10 米
}

def voxelization(data, grid_size, boundaries):
    # 计算每个体素的大小
    voxel_size = (
        (boundaries['x'][1] - boundaries['x'][0]) / grid_size[0],
        (boundaries['y'][1] - boundaries['y'][0]) / grid_size[1],
        (boundaries['z'][1] - boundaries['z'][0]) / grid_size[2]
    )
    # 计算每个点所属的体素坐标
    x_indices = ((data['x'] - boundaries['x'][0]) / voxel_size[0]).astype(int)
    y_indices = ((data['y'] - boundaries['y'][0]) / voxel_size[1]).astype(int)
    z_indices = ((data['z'] - boundaries['z'][0]) / voxel_size[2]).astype(int)
    # 创建体素网格
    voxel_grid = np.zeros(grid_size, dtype=np.uint8)
    # 将点云数据填充到体素网格中
    voxel_grid[x_indices, y_indices, z_indices] = 1
    return voxel_grid

def update_plot(frame):
    try:
        # 读取 Excel 文件数据
        df = pd.read_excel(os.path.join(folder_path, excel_files[frame]))
        # 提取 xyz 列数据
        x = df['X']
        y = df['Y']
        z = df['Z']
        # 创建点云数据 DataFrame
        point_cloud = pd.DataFrame({'x': x, 'y': y, 'z': z})
        # 筛选符合边界范围的点云数据
        point_cloud = point_cloud[
            (point_cloud['x'] >= BOUNDARIES['x'][0]) & (point_cloud['x'] <= BOUNDARIES['x'][1]) &
            (point_cloud['y'] >= BOUNDARIES['y'][0]) & (point_cloud['y'] <= BOUNDARIES['y'][1]) &
            (point_cloud['z'] >= BOUNDARIES['z'][0]) & (point_cloud['z'] <= BOUNDARIES['z'][1])
        ]
        # 进行体素化
        voxel_grid = voxelization(point_cloud, GRID_SIZE, BOUNDARIES)
        # 清除前一帧的绘图
        ax.clear()
        # 绘制体素
        ax.voxels(voxel_grid, facecolors='r', edgecolor='w')
        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # 设置坐标轴范围
        ax.set_xlim(0, GRID_SIZE[0])
        ax.set_ylim(0, GRID_SIZE[1])
        ax.set_zlim(0, GRID_SIZE[2])
        # 设置图形标题
        ax.set_title(f'Frame {frame+1}/{len(excel_files)}')
        # 保存体素文件
        output_file = os.path.join(output_path, f'pHistBytes_clustered_voxel_{frame+1}.npy')
        np.save(output_file, voxel_grid)
        print(f'Saved voxelization for frame {frame+1} to {output_file}')
    except Exception as e:
        print(f"Error occurred in frame {frame+1}: {e}")

# 设置文件夹路径和 Excel 文件列表
folder_path = r'C:\Users\Kano\Desktop\radar_toolbox_1_30_01_03\tools\visualizers\Industrial_Visualizer\binData\image\SNR\pHistBytes_clustered'
excel_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.xlsx')])
# 设置保存体素文件的路径
output_path = r'C:\Users\Kano\Desktop\radar_toolbox_1_30_01_03\tools\visualizers\Industrial_Visualizer\binData\image\SNR\pHistBytes_clustered_voxel1'

# 创建目标文件夹（如果不存在）
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 创建 3D 图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 创建动画
animation = FuncAnimation(fig, update_plot, frames=len(excel_files), interval=200)

# 显示动画
plt.show()