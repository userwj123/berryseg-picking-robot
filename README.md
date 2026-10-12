<<<<<<< HEAD
# 蓝莓实例分割工程

2026-10-04 新增基于 GPU 50 轮权重和自采 D435 原生 `.db3` 录像的离线三维定位回放。实际数据检查、相机坐标结果、FR5 模拟格式与下一步验证步骤见 [蓝莓三维定位与 FR5 方案](reports/localization_20261004_final/蓝莓三维定位与FR5下一步方案.md)。入口为 `python -m blueberry_seg.localize_recording`；只导出候选与明确不可执行的模拟数据，不连接机械臂。

使用 YOLO11n-seg 构建蓝莓实例分割，支持 CPU 初步训练、CUDA 正式训练、验证及图像/视频推理。当前新增公开 COCO 单果数据和 BSAIL/SAM 实例数据导入；原 DB-1 前景派生标签保留作工程验证。

已在指定 ai 环境完成两个公开实例数据集的下载、整理、全量校验，以及合并数据的 CPU 五轮训练和分来源评估。结果见 [公开数据训练与评估报告](reports/public_training/训练与评估报告.md)。本次权重输出单类别 blueberry，成熟度分类数据另行保留。

最初 DB-1 的两轮工程验证记录保留在 [初步训练结果](reports/初步训练结果.md)；该旧权重 mask mAP50=0，与本次公开实例数据训练结果分开记录。

## 枝干语义分割第一版

新增独立 `branch_seg` 模块：DeepLabV3-MobileNetV3，枝干/背景二分类，支持 CPU 和单张 CUDA GPU。该模型与原蓝莓实例模型分别训练；输出二维枝干 mask 与骨架，果簇连接关系、三维夹持位置及机械臂动作需后续接入。

数据在 `data/processed/branch_pilot_v1/`：从原公开训练图片挑选 30 张，20 train / 5 val / 5 test。使用 AI 视觉路径提示和 SAM2.1 tiny 辅助生成标签，保留原始 SAM 结果与逐张预览；**尚未经过专家人工复核**。标签为 0=局部背景、1=可见标注枝段、255=未标注/遮挡/不确定边界/填充。局部监督并非全图完整枝干真值，指标只能衡量对当前辅助标签的拟合。

```powershell
$py = 'D:\Program Files (x86)\Anaconda\envs\ai\python.exe'
& $py scripts/download_branch_weights.py
& $py scripts/run_branch_pilot.py --config configs/branch_cpu.yaml
```

CPU 配置为 512 输入、batch=2、40 轮上限、验证集连续 12 轮不改善提前停止。前 8 轮冻结骨干，随后微调；通过验证集 Dice 选择最佳权重，再只用验证集选择阈值，最后评估 test。输出在 `runs/branch/cpu_pilot_v1/`，状态、日志、完整预测对照和自动报告在 `reports/branch_pilot/`。已有输出时拒绝重复启动覆盖。

GPU 初训可使用 `python -m branch_seg.train --config configs/branch_gpu.yaml`，需提前安装 CUDA PyTorch；GPU 路径未在本机验证。中断恢复：

```powershell
& $py -m branch_seg.train --config configs/branch_cpu.yaml --resume runs/branch/cpu_pilot_v1/weights/last.pt
```

单独评估和新 RGB 图片推理：

```powershell
& $py -m branch_seg.evaluate --weights runs/branch/cpu_pilot_v1/weights/best.pt --data data/processed/branch_pilot_v1 --output reports/branch_pilot/evaluation
& $py -m branch_seg.predict --weights runs/branch/cpu_pilot_v1/weights/best.pt --source path/to/photo.jpg --output runs/branch/new_photo --threshold 0.5
```

推理阈值可按 `reports/branch_pilot/evaluation/evaluation.json` 的 val 所选值调整；推理保存原图像素坐标下的概率、mask、骨架和叠加图。骨架不直接作为机械臂轨迹。训练入口要求显式允许辅助标签，并检查分组、重复图、掩膜编码、视觉复核与哈希；标签修改后需重新复核并更新清单。

新增工程直接使用 PyTorch/torchvision、NumPy、Pillow、OpenCV、scikit-image、PyYAML、matplotlib，已有 ai 环境可运行；`requirements.txt` 已显式列出 matplotlib，其余 CPU/CUDA 安装规则与下文一致。官方权重与来源/哈希存于 `weights/`，数据许可沿用原来源记录。

## 枝干新增样本与后续训练

2026-10-08 首版枝干模型已完成 CPU 40 轮训练（512 输入，10.7 分钟），5 张 test 的局部辅助标签 IoU 44.8%、Dice 61.9%。完整结果与全部测试预测见 [枝干训练评估报告](reports/branch_pilot/枝干分割训练与评估报告.md)。模型推理已验证可输出原图尺寸的 mask、概率和骨架。

随后新增 **40 张 train、30 张 test**，使用真实公开照片，排除与首批的已识别来源组、相同图和感知哈希距离不超过 8 的近似图。标签仍为 AI/SAM 局部辅助草稿，未经过专家确认。新增 70 张在 `data/processed/branch_extension_v1`；合并可迁移数据在 `data/processed/branch_combined_v2`，共 **60 train / 5 val / 35 test**。新增 test 尚未训练、调参或推理计分。制作和核验记录见 [新增样本说明](reports/branch_extension/新增70张枝干样本说明.md)，逐张预览见 [标签复核索引](reports/branch_extension/标签复核索引.md)。

下一次扩充训练可用 `python -m branch_seg.train --config configs/branch_cpu_extended.yaml`；GPU 用 `configs/branch_gpu_extended.yaml`。这两份配置从官方预训练初始化，使用全新结果目录。本次没有启动扩充训练，也没有改写首批训练结果。训练前建议人工修正并冻结测试标签；改标签需同步清单哈希与复核记录。

## 数据检查结论

- 源数据有 **1,195 对** BMP 图像与 PNG mask。文件名含 Camera、Video 和帧编号。
- 抽查 PNG 为黑白 RGB 前景图；全量转换逐张检查是否为二值格式。**发布 mask 没有每颗蓝莓的实例 ID。**
- 默认以连通域生成派生实例轮廓：分开的果实可生成独立实例，相互接触的果实可能合并。
- 可选 `watershed` 利用距离变换拆分相邻区域，但边界仍是伪标签，可能过分割/漏分割，应人工复核。
- 只有 `blueberry` 一个类别；目前没有成熟度标签，模型不输出成熟/未成熟。
- 按“相机+视频”分组划分 train/val/test，避免同一视频的相邻帧跨集合；不代表已消除不同视频拍摄同一植株的相似性。现场效果仍需独立自采数据验证。

转换统计见 `data/processed/db1/summary.json`；每张图的来源、标注文件校验和及转换统计见 `manifest.json`；标注预览在 `review/`。基于派生标签得到的实例 mAP 不能当作真实单果分离精度。

相邻果实合并问题的替代数据与辅助标注路线见 [实例分割数据与 SAM 方案调研](reports/实例分割数据与SAM方案调研.md)。后续正式训练优先使用下述公开实例数据入口。

## 公开实例数据：下载、整理、训练与评估

```powershell
$py = 'D:\Program Files (x86)\Anaconda\envs\ai\python.exe'
& $py scripts/download_instance_datasets.py
& $py -m blueberry_seg.import_instances
& $py scripts/check_dataset.py --data data/processed/public_combined
& $py -m blueberry_seg.train --config configs/cpu_public_train.yaml
```

下载器支持保留部分文件后续传；校验 ZIP CRC，Ni 镜像另外核对源 SHA256。原包、解压数据与下载清单保存在 `data/raw/ni2020/` 和 `data/raw/bsail_sam/`。不执行下载包中的脚本或使用其缓存文件。转换输出目录必须为空；重新转换请指定新的 `--output`。

转换产物：

| 目录 | 类别 | 数据来源 |
| --- | --- | --- |
| `data/processed/ni2020` | blueberry | COCO 单果多边形，类别合并但实例保留 |
| `data/processed/ni2020_maturity` | Immature / Mature | COCO 原成熟度分类 |
| `data/processed/bsail_sam` | blueberry | SAM 训练/验证标签 + 包内标为 manual 的测试标签 |
| `data/processed/bsail_sam_maturity` | immature / mature / semi-mature | 同上，保留成熟度分类 |
| `data/processed/public_combined` | blueberry | 合并两个来源；人工测试集隔离 |

每个目录均有可移植的 `dataset.yaml`、`dataset_smoke.yaml`、`manifest.json`、`summary.json` 和轮廓预览。只改变分类与文件布局，不通过连通域重新推导实例。多部件/RLE、坏图、越界/退化轮廓等不能无损转换的图像会整体隔离，避免删掉一个标签后让未标果实成为背景。记录见 `reports/dataset_download/import_audit.json`。

重复检查采用解码后的 RGB 哈希与文件名来源分组：Roboflow 增强版本、切片及显式命名同一植株的不同视角优先留在 test，其次 val；跨集合的训练版本隔离。此检查不能证明所有不同文件名视频/植株都独立，最终仍需 D435 独立人工标注集。Ni 包中无标签的 55 张图像保存在 `images/unlabeled/`，不出现在训练/评估 YAML。BSAIL 原 YAML 将 valid 同时设为 test，适配器使用包内独立的人工测试目录修正这一设置。

CPU 基线配置使用完整整理后的合并训练集，5 epochs、384 输入、batch=4、4 线程。一次完成训练及分别评估各来源：

```powershell
& $py scripts/run_public_cpu_baseline.py
```

全量 50 轮 CPU 实验使用独立配置，从官方预训练权重开始，384 输入、batch=4、4 线程。关闭提前停止以完成 50 轮，保留 best/last 权重并每 5 轮保存快照。按五轮实测预留 8–10 小时。训练完成后自动进行四组独立评估，并生成与五轮基线的对比报告及预测图：

```powershell
& $py scripts/run_public_cpu_baseline.py --config configs/cpu_public_50epochs.yaml --report-directory reports/public_training_50epochs
```

新权重在 `runs/segment/cpu_public_50epochs/weights/`；训练日志、状态和最终报告在 `reports/public_training_50epochs/`。`experiment.json` 的 `status` 区分 training/evaluating/reporting/complete/failed；只有 complete 表示训练、评估及报告均完成。可随时查看当前进度：

```powershell
& $py scripts/public_training_status.py --report-directory reports/public_training_50epochs --run-directory runs/segment/cpu_public_50epochs
```

中途退出时可恢复训练状态（此命令仅恢复训练，完成后需另行评估）：

```powershell
& $py -m blueberry_seg.train --config configs/cpu_public_50epochs.yaml --resume runs/segment/cpu_public_50epochs/weights/last.pt
```

结果与运行日志在 `reports/public_training/`，权重在 `runs/segment/cpu_public_baseline/weights/`。评估单个来源：

```powershell
& $py -m blueberry_seg.evaluate --weights runs/segment/cpu_public_baseline/weights/best.pt --data data/processed/ni2020/dataset.yaml --split val --device cpu --imgsz 384 --name ni2020_val
& $py -m blueberry_seg.evaluate --weights runs/segment/cpu_public_baseline/weights/best.pt --data data/processed/bsail_sam/dataset.yaml --split test --device cpu --imgsz 384 --name bsail_manual_test
```

GPU 迁移后使用 `configs/gpu_public_train.yaml`；继续微调可加 `--model runs/segment/cpu_public_baseline/weights/best.pt`。迁移训练数据时复制对应 `data/processed/` 目录即可，不依赖原始数据盘符。成熟度训练用 `--data data/processed/bsail_sam_maturity/dataset.yaml` 等覆盖；本次合并基线只识别单类别蓝莓。

新增导入器只使用基础 requirements 中已有依赖。Ni 镜像声明 CC-BY-NC-SA-4.0；BSAIL 的 Kaggle 元数据标为 Unknown，包内 Roboflow 字段为 Private；记录来源声明，商业授权尚未确认。

## 环境安装与迁移

本机使用 Python 3.12，解释器：`D:\Program Files (x86)\Anaconda\envs\ai\python.exe`。

- `requirements.txt`：固定工程主要库版本，不固定 CPU/CUDA 的 PyTorch 构建。**先安装合适的 PyTorch。**
- `requirements-cpu.txt`：CPU PyTorch、torchvision 与工程依赖。
- `requirements-lock-cpu.txt`：本机验证环境的完整工程依赖闭包，不包含 ai 环境中的无关库。该文件适用于同类 Windows/Python 3.12 CPU 环境；GPU 迁移请勿使用它安装 CPU torch。
- `reports/environment.json`：实际 Python、平台和版本记录。

新机器建议创建专用 Python 3.12 环境，例如：

```powershell
conda create -n blueberry python=3.12 -y
conda activate blueberry
```

CPU 安装：

```powershell
python -m pip install -r requirements-cpu.txt
```

相同 CPU 平台需要完整锁定版本时：

```powershell
python -m pip install -r requirements-lock-cpu.txt
```

GPU 安装：先按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)选择实际平台和 CUDA 版本，运行其 torch/torchvision 安装命令，再安装：

```powershell
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

最后一项必须为 `True` 才可使用 GPU 配置。复制 CPU 环境后仅改 `device=0` 不会获得 CUDA；要换装支持 CUDA 的 PyTorch。项目指定 GPU 时发现不可用会报错，不会悄悄退回 CPU。

## DB-1 旧数据的工程验证入口

在本项目根目录打开 PowerShell：

```powershell
$py = 'D:\Program Files (x86)\Anaconda\envs\ai\python.exe'
```

后续命令均在项目根目录执行。新环境可把 `& $py` 换成 `python`。

### 1. 检查与转换数据

```powershell
& $py scripts/inspect_db1.py --source 'E:\蓝莓数据\DB-1'
& $py -m blueberry_seg.prepare --source 'E:\蓝莓数据\DB-1' --output data/processed/db1 --method components
& $py scripts/check_dataset.py --data data/processed/db1
```

已生成的输出文件夹不能覆盖。需要调整算法或重新划分时，使用另一个输出目录，保留转换来源：

```powershell
& $py -m blueberry_seg.prepare --source 'E:\蓝莓数据\DB-1' --output data/processed/db1_watershed --method watershed
```

分水岭训练时显式指定 `--data data/processed/db1_watershed/dataset.yaml`。`--min-area` 是原图上的像素面积阈值；`--max-side` 默认 1280，设为 0 可保留原始分辨率。归一化多边形随图像缩放保持位置一致，极小果实仍可能因输入缩小而丢失。需要高质量实例训练时，应复核轮廓，补充真实实例标注；不要只扩大伪标签数量。

可先对同一张源图比较两种转换：

```powershell
& $py -m scripts.preview_labels --source 'E:\蓝莓数据\DB-1' --stem 'Camera 3 Video (121)_1201'
```

输出位于 `reports/label_comparison/comparison.jpg`。

### 2. 下载官方预训练权重

```powershell
& $py scripts/download_weights.py
```

权重位于 `weights/yolo11n-seg.pt`，官方来源与下载文件 SHA-256 记录在 `weights/pretrained_source.json`。下载完成后可复制权重到离线训练机器。缺失权重时训练会明确报错，不会默认从随机权重开始。

### 3. CPU 初步训练

```powershell
& $py -m blueberry_seg.train --config configs/cpu_smoke.yaml
```

配置为 16 张 train、4 张 val、2 epochs、320 输入、batch=2、4 个计算线程、workers=0。该运行验证加载数据、反向传播、mask loss、验证和 best/last 权重保存，不用于评估最终效果。

结果通常位于 `runs/segment/cpu_smoke/`。已存在同名训练目录时框架自动递增目录名，查看命令最后输出的实际路径。

正式 CPU 训练（计算量明显增加，建议在 GPU 上完成）：

```powershell
& $py -m blueberry_seg.train --config configs/cpu_train.yaml
```

### 4. GPU 训练与继续微调

确认 CUDA 环境后使用：

```powershell
python -m blueberry_seg.train --config configs/gpu_train.yaml --device 0
```

默认 100 epochs、960 输入、batch=4。显存不足减小 batch/imgsz；性能足够再增大。CPU 与 GPU 共用数据、模型和执行代码。

使用 CPU 已训练权重开始新的 GPU 微调训练：

```powershell
python -m blueberry_seg.train --config configs/gpu_train.yaml --model runs/segment/cpu_smoke/weights/best.pt --name gpu_finetune
```

要恢复**中断中的**训练状态使用：

```powershell
python -m blueberry_seg.train --config configs/gpu_train.yaml --resume runs/segment/gpu_train/weights/last.pt --device 0
```

`--resume` 由框架恢复原训练参数和优化器；不用于给已完成的 2 轮训练追加一个新的 100 轮任务。已完成权重继续训练用 `--model`，如上例。

### 5. 推理与验证

```powershell
& $py -m blueberry_seg.predict --weights runs/segment/cpu_smoke/weights/best.pt --source data/processed/db1/images/val --device cpu --output runs/predict/val
& $py -m blueberry_seg.evaluate --weights runs/segment/cpu_smoke/weights/best.pt --data data/processed/db1/dataset_smoke.yaml --device cpu --imgsz 320 --batch 2
```

推理输出叠加可视化图片及 `instances.jsonl`，包括每颗预测果的置信度、框和多边形。坐标属于**原始 RGB 图像像素坐标**，便于后续关联 D435 深度；当前工程不连接机械臂。

视频可直接作为 `--source`，相机索引可用 `--source 0`。推理目录不能覆盖已有输出，请换名称。初步模型可能漏检或输出空列表，不能仅凭低置信度可视化判断可用于采摘。

独立 test 集评估：

```powershell
python -m blueberry_seg.evaluate --weights runs/segment/gpu_train/weights/best.pt --split test --device 0 --imgsz 960
```

### 6. 自检与环境记录

```powershell
& $py -m unittest discover -s tests -v
& $py scripts/record_environment.py
```

## 目录

```text
blueberry_seg/       标注转换、数据划分、训练、推理、验证、设备与路径管理
configs/            CPU smoke、CPU 全量、GPU 全量配置
scripts/            源数据检查、官方权重下载、数据验证、环境记录
tests/              标签几何、分水岭、边界与视频分组测试
data/processed/     转换数据、可移植 YAML、清单、标签预览
weights/            官方预训练权重
runs/               训练权重、指标、推理结果
reports/            源数据检查、环境信息与本次执行报告
```

迁移到 GPU 机器时复制工程、`data/processed/db1/`、`weights/`，需要微调则带上对应 `runs/.../weights/`。运行时重新生成当前机器路径绑定，无需修改数据 YAML 中的旧盘符。源数据转换清单仍保留原始来源路径，属于溯源信息，不用于训练读取。

模型和框架的使用方式见 [YOLO11 官方说明](https://docs.ultralytics.com/models/yolo11/)，数据格式见 [实例分割标签规范](https://docs.ultralytics.com/datasets/segment/)。框架与预训练模型许可为 AGPL-3.0/Enterprise，数据许可按 AgriVision 发布记录另行保留。
=======
# BerrySeg

>>>>>>> e87d256c29a048fc086cf24e777386123f9d694f
