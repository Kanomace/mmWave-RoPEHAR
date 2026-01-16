import numpy as np

# 加载npy文件
file_path = r'C:\Users\Kano\Desktop\radar_toolbox_1_30_01_03\tools\visualizers\Industrial_Visualizer\binData\trainData\squat\pHistBytes_clustered_voxel\pHistBytes_clustered_voxel_99.npy'
voxel_data = np.load(file_path)

# 获取体素数据的形状
voxel_shape = voxel_data.shape

# 计算体素体积
voxel_volume = np.prod(voxel_shape)

print("Voxel shape:", voxel_shape)
print("Voxel volume:", voxel_volume)