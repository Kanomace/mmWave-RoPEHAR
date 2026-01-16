# %%
import sys
import os

# 设置路径
model_path = 'D:/Ti/Py_mmWave_Roformer/rope_informer'
checkpoints = 'D:/Ti/Py_mmWave_Roformer/checkpoints/'
output_path = 'D:/Ti/Py_mmWave_Roformer/test_results/'

# 确保目录存在
os.makedirs(checkpoints, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

# 将模型路径添加到sys.path
sys.path.append(os.path.dirname(model_path))

# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import argparse

# 数据路径定义
data_paths = {
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

# 行为类别映射
behavior_mapping = {
    "stationary": 0,
    "run": 1,
    "squat": 2,
    "stand": 3,
    "walk": 4
}


# 自定义数据集类
class BehaviorDataset(Dataset):
    def __init__(self, data_paths, seq_len=15000, transform=None, is_train=True):
        self.data = []
        self.labels = []
        self.seq_len = seq_len
        self.transform = transform

        # 收集所有图像路径和标签
        for behavior, paths in data_paths.items():
            label = behavior_mapping[behavior]

            # 处理XOZ视角
            xoz_path = paths["xoz"]
            if os.path.exists(xoz_path):
                for img_name in os.listdir(xoz_path):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        self.data.append((os.path.join(xoz_path, img_name), label, "xoz"))

            # 处理YOZ视角
            yoz_path = paths["yoz"]
            if os.path.exists(yoz_path):
                for img_name in os.listdir(yoz_path):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        self.data.append((os.path.join(yoz_path, img_name), label, "yoz"))

        # 划分训练集和测试集
        train_data, test_data = train_test_split(
            self.data, test_size=0.2, random_state=42, stratify=[d[1] for d in self.data]
        )

        self.data = train_data if is_train else test_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label, view = self.data[idx]

        # 加载图像
        img = Image.open(img_path).convert('L')  # 转换为灰度图

        if self.transform:
            img = self.transform(img)

        # 将图像转换为序列
        img_array = np.array(img).flatten()

        # 如果序列长度不够，进行填充
        if len(img_array) < self.seq_len:
            pad_width = self.seq_len - len(img_array)
            img_array = np.pad(img_array, (0, pad_width), mode='constant')
        # 如果序列过长，进行截断
        elif len(img_array) > self.seq_len:
            img_array = img_array[:self.seq_len]

        return torch.FloatTensor(img_array), torch.tensor(label, dtype=torch.long)


# 创建数据加载器
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 调整图像大小
    transforms.ToTensor(),
])

train_dataset = BehaviorDataset(data_paths, seq_len=15000, transform=transform, is_train=True)
test_dataset = BehaviorDataset(data_paths, seq_len=15000, transform=transform, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# %%
# 参数设置
parser = argparse.ArgumentParser(description='[Informer] Human Activity Recognition')

parser.add_argument('--model', type=str, required=False, default='informer',
                    help='model of experiment, options: [informer]')
parser.add_argument('--data', type=str, required=False, default='Classification', help='data')
parser.add_argument('--root_path', type=str, default='', help='root path of the data file')
parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=8, help='batch size of train input data')
parser.add_argument('--seq_len', type=int, default=15000, help='input sequence length of Informer encoder')
parser.add_argument('--output_path', type=str, default=output_path, help='test_results')
parser.add_argument('--checkpoints', type=str, default=checkpoints, help='location of model checkpoints')
parser.add_argument('--test_ratio', type=float, default=0.2, help='')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--has_rope', type=bool, default=True, help='')

# 其他参数保持不变
# 其他参数
parser.add_argument('--data_path', type=str, default='', help='data file')
parser.add_argument('--features', type=str, default='M', help='forecasting task')
parser.add_argument('--target', type=str, default='', help='target feature')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
parser.add_argument('--label_len', type=int, default=3, help='start token length')  # 调整为适合分类任务
parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')  # 分类任务预测长度为1
parser.add_argument('--dec_in', type=int, default=1250, help='decoder input size')
parser.add_argument('--c_out', type=int, default=5, help='output size')  # 5个类别
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', default=True, help='whether to use distilling in encoder')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder')
parser.add_argument('--embed', type=str, default='fixed', help='time features encoding')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', default=True, help='use mix attention in generative decoder')
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=2, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='cross_entropy', help='loss function')  # 改为交叉熵损失
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', default=False, help='use automatic mixed precision training')
parser.add_argument('--inverse', action='store_true', default=False, help='inverse output data')
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='use multiple gpus')
parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

args = parser.parse_known_args()[0]

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

# 导入模型
from rope_informer import Exp_Informer

Exp = Exp_Informer
torch.manual_seed(111)


# 修改Exp_Informer类以使用新的数据加载器
class CustomExp_Informer(Exp):
    def _get_data(self, flag):
        if flag == 'test':
            return test_dataset, test_loader
        else:
            return train_dataset, train_loader


# 设置实验
setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(
    args.model, args.data, args.features,
    args.seq_len, args.label_len, args.pred_len,
    args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor,
    args.embed, args.distil, args.mix, args.des, 1)

exp = CustomExp_Informer(args)  # 使用自定义的实验类

print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
exp.train(setting)

print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
exp.test(setting)

torch.cuda.empty_cache()

# %%
# 结果可视化
import matplotlib.pyplot as plt
import numpy as np

truth = np.load(os.path.join(args.output_path, setting, "true.npy"))
preds = np.load(os.path.join(args.output_path, setting, "pred.npy"))

# 计算每个类别的总数和预测正确的数量
bin_counts = np.bincount(truth, minlength=5)
correct_counts = np.bincount(truth[preds == truth], minlength=5)

# 绘制直方图
categories = ['stationary', 'run', 'squat', 'stand', 'walk']
plt.figure(figsize=(10, 6))
plt.bar(categories, bin_counts, width=0.5, align='center', alpha=0.8, label='truth', color='purple')
plt.bar(categories, correct_counts, width=0.5, align='edge', alpha=1.0, label='correct', color='lightgreen')
plt.xlabel('Behavior Categories')
plt.ylabel('Count')
plt.title('Classification Results for 5 Human Behaviors')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('D:/Ti/Py_mmWave_Roformer/behavior_classification_results.png')
plt.show()