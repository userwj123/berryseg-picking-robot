# 蓝莓采摘机器人视觉系统

基于 YOLO11n-seg 的实例分割 + RealSense D435 深度定位 + 采摘策略。

## 功能

- **实例分割**：YOLO11n-seg 检测蓝莓，输出实例级 mask
- **3D 定位**：RealSense D435 `.db3` 深度图 → 3D 坐标 + 外法线
- **采摘策略**：距离过滤 → 生熟过滤 → TSP 顺序 → 6DOF 输出
- **枝干分割**：DeepLabV3-MobileNetV3（辅助避障）

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 下载预训练权重
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt -O weights/yolo11n-seg.pt

# 训练（BSAIL 数据集）
python -m blueberry_seg.train --config configs/gpu_train.yaml

# 3D 定位回放
python -m blueberry_seg.localize_recording --recording path/to/video.db3 --weights runs/segment/weights/best.pt

# 采摘策略
python -m blueberry_seg.picking_strategy
```

## 数据来源

- **BSAIL**：Kaggle `zhengkunli3969/blueberry-segmentation-with-segment-anything-model`
- **DB-1**：AgriVision 蓝莓数据集（师兄提供）
- **D435 录像**：自采 `.db3` 文件

## 文件说明

- `blueberry_seg/`：实例分割 + 3D 定位 + 采摘策略
- `branch_seg/`：枝干分割（辅助）
- `configs/`：训练配置
- `weights/`：预训练权重（Git LFS 或手动下载）

## 参考

- AgriVision: https://www.nature.com/articles/s41597-025-06263-3
- BerryNet: https://www.kaggle.com/datasets/zhengkunli3969/blueberry-segmentation-with-segment-anything-model
