import os
import numpy as np
import re
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd


def natural_sort_key(s):
    """自然排序键函数，确保数字按数值大小排序"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('(\d+)', s)]


def load_image_pairs(class_folders, window_size=5):
    """
    加载多个类别的XOZ和YOZ图像对

    参数:
    class_folders: 字典，键为类别名，值为包含XOZ和YOZ文件夹路径的字典
    window_size: 滑动窗口大小

    返回:
    X: 处理后的图像序列，形状为 (n_samples, window_size, 2, 25, 25)
    y: 对应的类别标签
    class_names: 类别名称列表
    """
    X = []
    y = []

    # 为每个类别分配一个数字标签
    class_names = list(class_folders.keys())
    class_labels = {class_name: idx for idx, class_name in enumerate(class_names)}

    # 处理每个类别的数据
    for class_name, folders in class_folders.items():
        print(f"处理类别: {class_name}")

        # 获取XOZ和YOZ文件夹路径
        xoz_folder = folders['xoz']
        yoz_folder = folders['yoz']

        # 获取文件夹中的所有图片文件并按自然顺序排序
        xoz_files = [f for f in os.listdir(xoz_folder) if f.endswith('.png') and '_XOZ' in f]
        yoz_files = [f for f in os.listdir(yoz_folder) if f.endswith('.png') and '_YOZ' in f]

        # 按自然顺序排序
        xoz_files.sort(key=natural_sort_key)
        yoz_files.sort(key=natural_sort_key)

        # 确保文件数量一致
        min_files = min(len(xoz_files), len(yoz_files))
        xoz_files = xoz_files[:min_files]
        yoz_files = yoz_files[:min_files]

        print(f"  找到 {min_files} 对图像")

        # 加载和处理图像对
        image_pairs = []
        for xoz_file, yoz_file in tqdm(zip(xoz_files, yoz_files),
                                       total=min_files,
                                       desc=f"处理 {class_name} 图像"):
            # 提取文件名中的数字部分用于验证匹配
            xoz_num = re.findall(r'\d+', xoz_file)[0]
            yoz_num = re.findall(r'\d+', yoz_file)[0]

            # 确保文件匹配
            if xoz_num != yoz_num:
                print(f"警告: 文件不匹配 - {xoz_file} 和 {yoz_file}")
                continue

            # 加载XOZ图像 (25x25)
            xoz_img = Image.open(os.path.join(xoz_folder, xoz_file))
            xoz_array = np.array(xoz_img.convert('L'))  # 转换为灰度图

            # 加载YOZ图像 (15x25)
            yoz_img = Image.open(os.path.join(yoz_folder, yoz_file))
            yoz_array = np.array(yoz_img.convert('L'))  # 转换为灰度图

            # 调整YOZ图像尺寸为25x25
            yoz_resized = resize(yoz_array, (25, 25), preserve_range=True).astype(np.uint8)

            # 将两个图像堆叠为一个2通道图像
            two_channel_img = np.stack([xoz_array, yoz_resized], axis=0)
            image_pairs.append(two_channel_img)

        # 创建滑动窗口
        for i in range(len(image_pairs) - window_size + 1):
            # 获取窗口内的图像序列
            window = image_pairs[i:i + window_size]
            # 将窗口数据添加到X
            X.append(np.array(window))
            # 添加对应的类别标签
            y.append(class_labels[class_name])

    return np.array(X), np.array(y), class_names


# 定义数据文件夹路径
class_folders = {
    "stationary": {
        "xoz": "D:/Ti/Py_mmWave_Roformer/Dataset/stationary.test/pHistBytes_clustered_voxel/pHistBytes_clustered_voxel_XOZ",
        "yoz": "D:/Ti/Py_mmWave_Roformer/Dataset/stationary.test/pHistBytes_clustered_voxel/pHistBytes_clustered_voxel_YOZ"
    },
    "run": {
        "xoz": "D:/Ti/Py_mmWave_Roformer/Dataset/run.test/pHistBytes_clustered_voxel/pHistBytes_clustered_voxel_XOZ",
        "yoz": "D:/Ti/Py_mmWave_Roformer/Dataset/run.test/pHistBytes_clustered_voxel/pHistBytes_clustered_voxel_YOZ"
    },
    "squat": {
        "xoz": "D:/Ti/Py_mmWave_Roformer/Dataset/squat.test/pHistBytes_clustered_voxel/pHistBytes_clustered_voxel_XOZ",
        "yoz": "D:/Ti/Py_mmWave_Roformer/Dataset/squat.test/pHistBytes_clustered_voxel/pHistBytes_clustered_voxel_YOZ"
    },
    "stand": {
        "xoz": "D:/Ti/Py_mmWave_Roformer/Dataset/stand.test/pHistBytes_clustered_voxel/pHistBytes_clustered_voxel_XOZ",
        "yoz": "D:/Ti/Py_mmWave_Roformer/Dataset/stand.test/pHistBytes_clustered_voxel/pHistBytes_clustered_voxel_YOZ"
    },
    "walk": {
        "xoz": "D:/Ti/Py_mmWave_Roformer/Dataset/walk.test/pHistBytes_clustered_voxel/pHistBytes_clustered_voxel_XOZ",
        "yoz": "D:/Ti/Py_mmWave_Roformer/Dataset/walk.test/pHistBytes_clustered_voxel/pHistBytes_clustered_voxel_YOZ"
    }
}

# 加载和处理图像数据
window_size = 5
X, y, class_names = load_image_pairs(class_folders, window_size=window_size)

print(f"数据集形状: X={X.shape}, y={y.shape}")
print(f"类别分布: {np.bincount(y)}")
print(f"类别名称: {class_names}")


# 可视化一些样本
def visualize_samples(X, y, class_names, num_samples=5):
    """可视化一些样本图像"""
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    for i in range(num_samples):
        # 随机选择一个样本
        idx = np.random.randint(0, len(X))

        # 获取中间帧的两个通道图像
        middle_frame = X[idx][len(X[idx]) // 2]  # 取序列中间帧
        xoz_img = middle_frame[0]  # XOZ投影
        yoz_img = middle_frame[1]  # YOZ投影

        # 合并两个通道为RGB图像（仅用于可视化）
        rgb_img = np.stack([xoz_img, yoz_img, np.zeros_like(xoz_img)], axis=2)

        # 显示图像
        axes[i, 0].imshow(xoz_img, cmap='gray')
        axes[i, 0].set_title(f"样本 {idx}: XOZ投影 (类别: {class_names[y[idx]]})")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(yoz_img, cmap='gray')
        axes[i, 1].set_title(f"样本 {idx}: YOZ投影")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(rgb_img)
        axes[i, 2].set_title(f"样本 {idx}: 合并视图")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig('D:/Ti/Py_mmWave_Roformer/sample_visualization.png')
    plt.show()


# 可视化一些样本
visualize_samples(X, y, class_names, num_samples=5)


# 将图像数据转换为时间序列格式
def convert_to_time_series(X, y):
    """
    将图像数据转换为时间序列格式

    参数:
    X: 图像数据，形状为 (n_samples, seq_len, channels, height, width)
    y: 标签

    返回:
    time_series_data: 时间序列数据，形状为 (n_samples, seq_len, channels*height*width)
    y: 标签
    """
    # 重塑数据: (n_samples, seq_len, channels, height, width) -> (n_samples, seq_len, channels*height*width)
    n_samples, seq_len, channels, height, width = X.shape
    time_series_data = X.reshape(n_samples, seq_len, -1)

    return time_series_data, y


# 转换为时间序列格式
X_ts, y_ts = convert_to_time_series(X, y)
print(f"转换后的数据形状: X_ts={X_ts.shape}, y_ts={y_ts.shape}")


# 创建CSV文件以便Informer模型读取
def create_csv_data(X, y, class_names, save_path):
    """创建CSV格式的数据"""
    os.makedirs(save_path, exist_ok=True)

    # 为每个样本创建一个DataFrame
    data_frames = []

    for i in range(X.shape[0]):
        # 获取当前样本的时间序列数据
        sample_data = X[i]  # 形状: (seq_len, features)

        # 创建时间戳 (假设每帧间隔相等)
        timestamps = pd.date_range(start='2023-01-01', periods=X.shape[1], freq='S')

        # 创建列名 (特征名)
        feature_names = [f'feature_{j}' for j in range(X.shape[2])]

        # 创建DataFrame
        df = pd.DataFrame(sample_data, index=timestamps, columns=feature_names)

        # 添加类别标签
        df['label'] = y[i]
        df['class_name'] = class_names[y[i]]

        # 添加样本ID
        df['sample_id'] = i

        data_frames.append(df)

    # 合并所有DataFrame
    combined_df = pd.concat(data_frames)

    # 保存为CSV
    csv_path = os.path.join(save_path, "time_series_data.csv")
    combined_df.to_csv(csv_path, index=True, index_label='date')

    print(f"CSV数据已保存到: {csv_path}")
    print(f"数据形状: {combined_df.shape}")

    return csv_path


# 创建CSV数据
save_path = "D:/Ti/Py_mmWave_Roformer/processed_data"
csv_path = create_csv_data(X_ts, y_ts, class_names, save_path)


# 标准化并保存数据
def normalize_and_save_data(X, y, save_path):
    """标准化数据并保存为numpy文件"""
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)

    # 标准化图像数据 (0-1范围)
    X_normalized = X.astype(np.float32) / 255.0

    # 保存数据
    np.save(os.path.join(save_path, "X.npy"), X_normalized)
    np.save(os.path.join(save_path, "y.npy"), y)

    # 保存类别名称
    with open(os.path.join(save_path, "class_names.txt"), 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")

    print(f"数据已保存到: {save_path}")
    print(f"X形状: {X_normalized.shape}, y形状: {y.shape}")


# 标准化并保存数据
normalize_and_save_data(X_ts, y_ts, save_path)


# 创建PyTorch数据集类
class ImageSequenceDataset(Dataset):
    """图像序列数据集类"""

    def __init__(self, X_path, y_path):
        self.X = np.load(X_path)
        self.y = np.load(y_path)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 获取图像序列
        sequence = torch.FloatTensor(self.X[idx])
        # 获取标签
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return sequence, label


# 创建数据集实例
dataset = ImageSequenceDataset(
    os.path.join(save_path, "X.npy"),
    os.path.join(save_path, "y.npy")
)

# 划分训练集和测试集
indices = list(range(len(dataset)))
train_indices, test_indices = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=dataset.y
)

# 创建子集数据集
from torch.utils.data import Subset

train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

# 创建数据加载器
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")


# 创建数据信息文件供Informer使用
def create_data_info_file(save_path, seq_len, feature_dim, num_classes):
    """创建数据信息文件"""
    info = {
        "seq_len": seq_len,
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "data_format": "time_series"
    }

    import json
    with open(os.path.join(save_path, "data_info.json"), 'w') as f:
        json.dump(info, f, indent=4)

    print(f"数据信息文件已创建: {os.path.join(save_path, 'data_info.json')}")


# 创建数据信息文件
create_data_info_file(save_path, X_ts.shape[1], X_ts.shape[2], len(class_names))