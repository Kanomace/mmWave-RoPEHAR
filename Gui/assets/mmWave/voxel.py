import open3d as o3d
import numpy as np
import pandas as pd

def voxelization_and_save(input_file, output_file):
    df = pd.read_excel(input_file)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(df[['X', 'Y', 'Z']].values)

    voxel_size = 0.2
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size)

    voxel_array = np.zeros((50, 25, 50), dtype=np.uint8)

    for voxel in voxel_grid.get_voxels():
        i = int((voxel.grid_index[0] + 5) / voxel_size)
        j = int((voxel.grid_index[1] + 5) / voxel_size)
        k = int((voxel.grid_index[2] + 5) / voxel_size)

        # 检查索引是否在目标范围内
        if 0 <= i < 50 and 0 <= j < 25 and 0 <= k < 50:
            voxel_array[i, j, k] = 1

    np.save(output_file, voxel_array)

    # 绘制体素化结果
    def animation_callback(vis):
        # 逐帧显示体素化结果
        vis.update_geometry(voxel_grid)
        vis.poll_events()
        vis.update_renderer()

    o3d.visualization.draw_geometries_with_animation_callback([voxel_grid], animation_callback)

input_path = r"C:\Users\Kano\Desktop\radar_toolbox_1_30_01_03\tools\visualizers\Industrial_Visualizer\binData\image\SNR\pHistBytes_clustered_voxel1"
output_path = r"C:\Users\Kano\Desktop\radar_toolbox_1_30_01_03\tools\visualizers\Industrial_Visualizer\binData\image\SNR\pHistBytes_clustered_voxel1"

for i in range(1, 31):
    input_file = f"{input_path}\\pHistBytes_clustered_{i}.xlsx"
    output_file = f"{output_path}\\pHistBytes_clustered_{i}.npy"

    voxelization_and_save(input_file, output_file)