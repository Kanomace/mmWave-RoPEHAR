import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean
import multiprocessing
"""
修正z轴上的欧氏距离
"""
z_scale_factor = 0.5  # 用于缩放z轴坐标的比例因子
esp_set = 1.0
min_samples_set = 5

def natural_sort_key(s):
    """
    自然排序的键函数
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', s)]

def z_corrected_euclidean(u, v):
    z_distance = abs(u[2] - v[2]) * z_scale_factor
    xy_distance = euclidean(u[:2], v[:2])
    corrected_distance = np.sqrt(xy_distance ** 2 + z_distance ** 2)
    return corrected_distance

def animate_point_cloud_folder(folder_path, file_prefix):
    # 获取文件夹中符合前缀的点云数据文件
    file_list = [f for f in os.listdir(folder_path) if f.startswith(file_prefix) and f.endswith('.xlsx')]
    file_list.sort(key=natural_sort_key)  # 使用自然排序对文件名进行排序

    # 创建画布和子图
    fig = plt.figure()
    ax_original = fig.add_subplot(121, projection='3d')
    ax_original.set_xlabel('X')
    ax_original.set_ylabel('Y')
    ax_original.set_zlabel('Z')
    sc_original = ax_original.scatter([], [], [], c='blue', alpha=0.5)

    ax_clustered = fig.add_subplot(122, projection='3d')
    ax_clustered.set_xlabel('X')
    ax_clustered.set_ylabel('Y')
    ax_clustered.set_zlabel('Z')
    sc_clustered = ax_clustered.scatter([], [], [], c='red')

    # 创建图例占位符
    legend = None

    def update(frame):
        nonlocal legend  # 使用nonlocal声明以便在函数内部修改外部的legend变量

        # 读取点云数据文件
        file_path = os.path.join(folder_path, file_list[frame])
        df = pd.read_excel(file_path)

        # 提取数据列
        x = df['X']+5   # 对 X 坐标进行平移
        y = df['Y']   # 对 Y 坐标进行平移
        z = df['Z']+5   # 对 Z 坐标进行平移
        # z = df['Z'] + 5  # 对 Z 坐标进行平移
        noise = df['SNR']  # 提取 Noise 列数据

        # 将点云数据转换为NumPy数组
        points = np.column_stack((x, y, z))

        # 执行DBSCAN聚类
        dbscan = DBSCAN(eps=esp_set, min_samples=min_samples_set, metric=z_corrected_euclidean)  # 使用自定义的距离度量函数
        labels = dbscan.fit_predict(points)

        # 获取聚类最多的簇的标签和数量
        unique_labels, counts = np.unique(labels, return_counts=True)
        max_cluster_label = unique_labels[np.argmax(counts)]

        # 提取聚类最多的簇的点云数据
        x_cluster = x[labels == max_cluster_label]
        y_cluster = y[labels == max_cluster_label]
        z_cluster = z[labels == max_cluster_label]
        noise_cluster = noise[labels == max_cluster_label]

        # 更新散点图数据点位置和颜色
        sc_original._offsets3d = (x, y, z)
        sc_clustered._offsets3d = (x_cluster, y_cluster, z_cluster)
        sc_clustered.set_array(noise_cluster)

        # 更新标题
        ax_original.set_title(file_list[frame] + ' - Original')
        ax_clustered.set_title(file_list[frame] + ' - Clustered')

        # 生成处理后的簇的DataFrame
        clustered_df = pd.DataFrame({'X': x_cluster, 'Y': y_cluster, 'Z': z_cluster, 'Noise': noise_cluster})

        # 保存处理后的簇为xlsx文件
        output_file_path = os.path.join(output_path, file_list[frame].replace(file_prefix, file_prefix + 'clustered_'))
        clustered_df.to_excel(output_file_path, index=False)

    # 设置坐标轴范围
    ax_original.set_xlim(0, 10)  # X 轴范围为 0 到 10
    ax_original.set_ylim(0, 6)  # Y 轴范围为 0 到 10
    ax_original.set_zlim(0, 10)  # Z 轴范围为 0 到 10
    ax_clustered.set_xlim(0, 10)
    ax_clustered.set_ylim(0, 6)
    ax_clustered.set_zlim(0, 10)

    # 创建动画
    animation = FuncAnimation(fig, update, frames=len(file_list), interval=300)

    # 显示图例
    legend = ax_clustered.legend(*sc_clustered.legend_elements(num=10), title='SNR', loc='lower right')
    ax_clustered.add_artist(legend)

    # 显示动画
    plt.show()

# 示例用法
folder_path = r'C:\Users\Kano\Desktop\radar_toolbox_1_30_01_03\tools\visualizers\Industrial_Visualizer\binData\image\SNR'
output_path = r'C:\Users\Kano\Desktop\radar_toolbox_1_30_01_03\tools\visualizers\Industrial_Visualizer\binData\image\SNR\pHistBytes_clustered'
# 创建目标文件夹（如果不存在）
if not os.path.exists(output_path):
    os.makedirs(output_path)

file_prefix = 'pHistBytes_'
animate_point_cloud_folder(folder_path, file_prefix)
