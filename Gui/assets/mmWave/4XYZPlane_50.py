import numpy as np
import os
import matplotlib.pyplot as plt

# 设置基础文件夹路径
base_folder = r'C:\Users\Kano\Desktop\radar_toolbox_1_30_01_03\tools\visualizers\Industrial_Visualizer\binData\traindata\9sign'

# 遍历每个体素文件夹
for i in range(1, 51):
    folder_path = os.path.join(base_folder, str(i))

    # 创建保存路径文件夹
    save_path_YOZ = os.path.join(folder_path, 'pHistBytes_clustered_voxel_YOZ')
    if not os.path.exists(save_path_YOZ):
        os.makedirs(save_path_YOZ)
    save_path_XOY = os.path.join(folder_path, 'pHistBytes_clustered_voxel_XOY')
    if not os.path.exists(save_path_XOY):
        os.makedirs(save_path_XOY)
    save_path_XOZ = os.path.join(folder_path, 'pHistBytes_clustered_voxel_XOZ')
    if not os.path.exists(save_path_XOZ):
        os.makedirs(save_path_XOZ)

    folder_path = os.path.join(folder_path, 'pHistBytes_clustered_voxel')

    os.makedirs(save_path_YOZ, exist_ok=True)
    os.makedirs(save_path_XOY, exist_ok=True)
    os.makedirs(save_path_XOZ, exist_ok=True)

    # 遍历体素文件夹中的所有.npy文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            try:
                # 读取体素文件
                file_path = os.path.join(folder_path, file_name)
                voxel_data = np.load(file_path)

                # 提取XOY平面
                xoyslice = np.max(voxel_data, axis=2)
                xoyslice = np.rot90(xoyslice)
                xoyslice = np.where(xoyslice > 0, 1, 0)  # 将体素区域设为白色，空白区域设为黑色

                # 提取XOZ平面
                xozslice = np.max(voxel_data, axis=1)
                xozslice = np.rot90(xozslice)
                xozslice = np.where(xozslice > 0, 1, 0)  # 将体素区域设为白色，空白区域设为黑色

                # 提取YOZ平面
                yozslice = np.max(voxel_data, axis=0)
                yozslice = np.rot90(yozslice)
                yozslice = np.where(yozslice > 0, 1, 0)  # 将体素区域设为白色，空白区域设为黑色

                # 保存为图像文件
                base_name = os.path.splitext(file_name)[0]
                save_file_name_YOZ = os.path.join(save_path_YOZ, base_name + '_YOZ.png')
                save_file_name_XOY = os.path.join(save_path_XOY, base_name + '_XOY.png')
                save_file_name_XOZ = os.path.join(save_path_XOZ, base_name + '_XOZ.png')

                plt.imsave(save_file_name_YOZ, yozslice, cmap='gray')
                plt.imsave(save_file_name_XOY, xoyslice, cmap='gray')
                plt.imsave(save_file_name_XOZ, xozslice, cmap='gray')
                print(f'Saved visualized images for {file_name} in folder {i}')
            except Exception as e:
                print(f"Error occurred for {file_name} in folder {i}: {e}")