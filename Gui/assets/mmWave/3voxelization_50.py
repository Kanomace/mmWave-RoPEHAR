import os
import pandas as pd
import numpy as np

# 定义体素网格大小
GRID_SIZE = (25, 15, 25)  # (x, y, z)
# 定义点云边界
BOUNDARIES = {
    'x': (-5, 5),   # X轴范围为 -10 到 10 米
    'y': ( 0, 6),   # Y轴范围为 0 到 10 米
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

# 设置文件夹路径
base_folder = r'C:\Users\Kano\Desktop\radar_toolbox_1_30_01_03\tools\visualizers\Industrial_Visualizer\binData\image\SNR\pHistBytes_clustered'
# 设置保存体素文件的文件夹名称
output_folder_name = 'pHistBytes_clustered_voxel'

for folder in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, folder, 'pHistBytes_clustered')
    # folder_path = os.path.join(base_folder, folder)
    if not os.path.isdir(folder_path):
        continue

    # 创建保存体素文件的文件夹
    output_folder_path = os.path.join(base_folder, folder, output_folder_name)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # 获取文件夹中的Excel文件列表
    excel_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.xlsx')])

    for frame, excel_file in enumerate(excel_files):
        try:
            # 读取 Excel 文件数据
            df = pd.read_excel(os.path.join(folder_path, excel_file))
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
            # 保存体素文件
            output_file = os.path.join(output_folder_path, f'{output_folder_name}_{frame+1}.npy')
            np.save(output_file, voxel_grid)
            print(f'Saved voxelization for frame {frame+1} in folder {folder}')
        except Exception as e:
            print(f"Error occurred in frame {frame+1} in folder {folder}: {e}")