import os
import sys
import glob
import json
import time
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

# ============ 路径配置 ============
# 数据源（仅用于推理）
XOZ_DIR = r"D:\Ti\Py_mmWave_Roformer\Dataset\squat.test\pHistBytes_clustered_voxel\pHistBytes_clustered_voxel_XOZ"
YOZ_DIR = r"D:\Ti\Py_mmWave_Roformer\Dataset\squat.test\pHistBytes_clustered_voxel\pHistBytes_clustered_voxel_YOZ"

# 模型与代码路径
ROPE_INFORMER_DIR = r"D:\Ti\Py_mmWave_Roformer\rope_informer"  # 应该包含 Exp_Informer 与模型代码
CHECKPOINT_PATH = r"D:\Ti\Py_mmWave_Roformer\model_checkpoint\checkpoint0914.pth"

# 输出
OUTPUT_DIR = r"D:\Ti\Py_mmWave_Roformer\inference_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ 参数与类别映射 ============
behavior_mapping = {
    "stationary": 0,
    "run": 1,
    "squat": 2,
    "stand": 3,
    "walk": 4,
}
id2label = {v: k for k, v in behavior_mapping.items()}

# 视角图像尺寸
image_sizes = {
    "xoz": (25, 25),  # W, H
    "yoz": (25, 15),
}

# 滑窗与序列长度，与 Kaggle 训练保持一致
WINDOW_SIZE = 3
SEQ_LEN = 3000  # 3 * (25*25 + 25*15) = 3 * (625 + 375) = 3000
BATCH_SIZE = 32

# 是否有标签（你的 run.test 没有多类标注，若你确认都为 run，可设为 True）
has_labels = True
default_label_idx = behavior_mapping["squat"]

# ============ 导入 rope_informer 代码 ============
if ROPE_INFORMER_DIR not in sys.path:
    sys.path.append(ROPE_INFORMER_DIR)

# 这里假设你的项目中有与 Kaggle 相同接口的 Exp_Informer
try:
    from rope_informer import Exp_Informer
except Exception as e:
    raise ImportError(f"无法从 {ROPE_INFORMER_DIR} 导入 Exp_Informer，请检查路径与模块。原始错误：{e}")

# ============ 数据集定义（仅推理） ============
class SlidingWindowFolderDataset(Dataset):
    """
    从两个文件夹(XOZ/YOZ)读取图像，按文件名排序，各自做滑窗；
    每个样本仅来自单一视角（与 Kaggle 逻辑一致：分别为 xoz 或 yoz 的窗口样本）。
    若需要将同一时间戳的 XOZ 与 YOZ 拼接为单一样本，请提供对齐规则后再改。
    """
    def __init__(self, xoz_dir: str, yoz_dir: str, window_size: int = 3,
                 seq_len: int = 3000, label_idx: int = None):
        self.window_size = window_size
        self.seq_len = seq_len
        self.samples: List[Tuple[List[str], str]] = []  # (paths, view)
        self.labels: List[int] = []
        self._collect(xoz_dir, "xoz", label_idx)
        self._collect(yoz_dir, "yoz", label_idx)

    def _collect(self, folder: str, view: str, label_idx: int):
        if not folder or not os.path.isdir(folder):
            return
        exts = (".png", ".jpg", ".jpeg", ".bmp")
        imgs = [p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(exts)]
        imgs.sort()
        if len(imgs) < self.window_size:
            return
        for i in range(len(imgs) - self.window_size + 1):
            win = imgs[i:i + self.window_size]
            self.samples.append((win, view))
            self.labels.append(label_idx if label_idx is not None else -1)

    def __len__(self):
        return len(self.samples)

    def _view_resize(self, img: Image.Image, view: str) -> Image.Image:
        size = image_sizes["xoz"] if view == "xoz" else image_sizes["yoz"]
        return img.resize(size)

    def __getitem__(self, idx):
        paths, view = self.samples[idx]
        window_arrays = []
        for p in paths:
            img = Image.open(p).convert("L")
            img = self._view_resize(img, view)
            arr = np.array(img, dtype=np.float32).flatten()
            window_arrays.append(arr)
        seq = np.concatenate(window_arrays, axis=0)
        # pad/trunc to seq_len
        if len(seq) < self.seq_len:
            pad = self.seq_len - len(seq)
            seq = np.pad(seq, (0, pad), mode="constant")
        elif len(seq) > self.seq_len:
            seq = seq[:self.seq_len]
        x = torch.from_numpy(seq).float()  # shape [seq_len]
        y = torch.tensor(self.labels[idx], dtype=torch.long) if self.labels[idx] >= 0 else torch.tensor(-1)
        meta = {
            "paths": paths,
            "view": view
        }
        return x, y, meta

# ============ args 定义（需与训练一致的关键字段） ============
class Args:
    def __init__(self):
        self.model = 'informer'
        self.data = 'Classification'
        self.root_path = ''
        self.enc_in = 1
        self.d_model = 64
        self.d_ff = 256
        self.train_epochs = 1
        self.batch_size = BATCH_SIZE
        self.seq_len = SEQ_LEN
        self.output_path = OUTPUT_DIR
        self.checkpoints = OUTPUT_DIR  # 推理时不写入checkpoint
        self.test_ratio = 0.2
        self.n_heads = 8
        self.has_rope = True
        self.features = 'M'
        self.target = 'OT'
        self.freq = 'h'
        self.label_len = 48
        self.pred_len = 24
        self.dec_in = 1
        self.c_out = 5
        self.e_layers = 2
        self.d_layers = 1
        self.s_layers = '3,2,1'
        self.factor = 5
        self.padding = 0
        self.distil = True
        self.dropout = 0.05
        self.attn = 'prob'
        self.embed = 'timeF'
        self.activation = 'gelu'
        self.output_attention = False
        self.do_predict = True
        self.mix = True
        self.cols = None
        self.num_workers = 0
        self.itr = 1
        self.patience = 2
        self.learning_rate = 1e-4
        self.des = 'inference'
        self.loss = 'mse'
        self.lradj = 'type1'
        self.use_amp = False
        self.inverse = False
        self.use_gpu = torch.cuda.is_available()
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0'

args = Args()

# ============ 准备数据 ============
label_idx = default_label_idx if has_labels else None
dataset = SlidingWindowFolderDataset(XOZ_DIR, YOZ_DIR, WINDOW_SIZE, SEQ_LEN, label_idx)
if len(dataset) == 0:
    raise RuntimeError("未找到足够的图像构成滑窗样本，请检查目录与文件后缀。")

from torch.utils.data import DataLoader

def collate_with_meta(batch):
    # batch: List[ (x, y, meta_dict) ]
    xs, ys, metas = [], [], []
    for x, y, meta in batch:
        xs.append(x)
        ys.append(y)
        metas.append(meta)  # 保留为字典列表
    xs = torch.stack(xs, dim=0)  # [B, seq_len]
    ys = torch.stack(ys, dim=0) if isinstance(ys[0], torch.Tensor) else torch.tensor(ys, dtype=torch.long)
    return xs, ys, metas

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_with_meta
)

# ============ 自定义实验类，只做推理 ============
class InferenceExp(Exp_Informer):
    def _get_data(self, flag):
        # 兼容 Exp_Informer 接口，不用于训练，仅返回我们自建的数据与loader
        return dataset, loader

# ============ 创建实验、构建模型并加载权重 ============
exp = InferenceExp(args)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Exp_Informer 一般在构造时会build模型；若需要主动拿到模型对象，可从 exp.model
model = getattr(exp, "model", None)
if model is None:
    # 若你的 Exp_Informer 没有在 __init__ 中构建模型，可尝试调用其内部构建方法或自行实例化
    try:
        model = exp._build_model()
        exp.model = model
    except Exception as e:
        raise RuntimeError(f"无法构建模型，请检查 Exp_Informer 实现。错误：{e}")

model = model.to(device)
model.eval()

def load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: str):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    # 兼容保存 dict 和直接保存 state_dict 两种格式
    if isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    elif isinstance(state, dict):
        # 可能本身就是 state_dict
        state_dict = state
    else:
        raise ValueError("未知的 checkpoint 格式，需包含 state_dict 或 model_state_dict。")
    # 处理可能的 DataParallel 前缀
    new_state = {}
    for k, v in state_dict.items():
        nk = k.replace("module.", "") if k.startswith("module.") else k
        new_state[nk] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"Warning: 缺失权重: {missing}")
    if unexpected:
        print(f"Warning: 多余权重: {unexpected}")

load_checkpoint_into_model(model, CHECKPOINT_PATH)

# ============ 推理 ============
all_preds = []
all_probs = []
all_labels = []
all_views = []
all_first_paths = []

softmax = torch.nn.Softmax(dim=-1)

start = time.time()
with torch.no_grad():
    for batch in loader:
        x, y, meta = batch  # x: [B, seq_len]
        x = x.to(device)
        # 根据你的 Exp_Informer 前向接口调整：
        # 常见形式：model(x) 或 model(x, ...)。这里假定分类输出 logits: [B, num_classes]
        logits = model(x) if callable(model) else model.forward(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = softmax(logits)
        preds = torch.argmax(probs, dim=-1)

        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        if has_labels:
            all_labels.append(y.numpy())
        else:
            all_labels.append(np.full((x.size(0),), -1, dtype=np.int64))
        # 记录元信息
        # meta 是一个字典列表（DataLoader collate 会将字典聚合为列表）
        # 这里取每个样本的第一个文件路径作为 ID
        for sample_meta in meta:
            all_views.append(sample_meta["view"])
            all_first_paths.append(sample_meta["paths"][0])

elapsed = time.time() - start
all_preds = np.concatenate(all_preds, axis=0)
all_probs = np.concatenate(all_probs, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# ============ 计算准确率（仅在 has_labels=True 时） ============
acc = None
if has_labels and np.all(all_labels >= 0):
    acc = float((all_preds == all_labels).mean())
    print(f"Inference done. Samples: {len(dataset)}, Accuracy: {acc:.4f}, Time: {elapsed:.2f}s")
else:
    print(f"Inference done. Samples: {len(dataset)}, Time: {elapsed:.2f}s")

# ============ 保存结果 ============
pred_labels = [id2label[int(i)] for i in all_preds]
max_probs = all_probs.max(axis=1)

df = pd.DataFrame({
    "sample_id": list(range(len(pred_labels))),
    "first_frame_path": all_first_paths,
    "view": all_views,
    "pred_label": pred_labels,
    "pred_index": all_preds.astype(int),
    "pred_confidence": max_probs
})
if has_labels:
    df["true_index"] = all_labels.astype(int)
    df["true_label"] = [id2label[i] if i in id2label else "unknown" for i in df["true_index"]]

csv_path = os.path.join(OUTPUT_DIR, "inference_results.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

summary = {
    "samples": int(len(dataset)),
    "has_labels": bool(has_labels),
    "accuracy": float(acc) if acc is not None else None,
    "elapsed_sec": float(elapsed),
    "classes": id2label
}
with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"结果已保存至:\n- {csv_path}\n- {os.path.join(OUTPUT_DIR, 'summary.json')}")