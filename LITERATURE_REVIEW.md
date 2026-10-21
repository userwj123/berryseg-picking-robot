# 农业机器人顶会与期刊检索指南（2024-2025）

## 一、顶级会议（机器人方向）

### 1. ICRA (IEEE International Conference on Robotics and Automation)
- **级别**：机器人领域顶会，CCF-A
- **2024 时间**：2024-05-13 至 2024-05-17（日本横滨）
- **2025 时间**：2025-06-02 至 2025-06-06（美国亚特兰大）
- **检索链接**：
  - 官网：https://2025.ieee-icra.org/
  - IEEE Xplore：https://ieeexplore.ieee.org/xpl/conhome/1000001/all-proceedings
  - arXiv 预印本：https://arxiv.org/list/cs.RO/recent（搜 "ICRA 2025" 或 "agricultural robot"）

**2024-2025 农业机器人相关论文**：
| 论文 | 会议 | 内容 |
|------|------|------|
| Hierarchical Tri-manual Planning for Vision-assisted Fruit Harvesting with Quadrupedal Robots | ICRA 2025 | 三臂四足机器人采摘，90% 成功率 |
| Advancement and Field Evaluation of a Dual-arm Apple Harvesting Robot | ICRA 2025 | 双臂苹果采摘，80.7% 成功率，5.97s/果 |
| A Modular Dual-Arm Apple Harvesting Robot with Enhanced Field Performance | ICRA 2025 | 模块化双臂，80.0% 成功率 |
| CLASP: Cluster-Level Autonomous Selective Picking Robot for Blueberry Harvesting | ICRA 2025 | **蓝莓采摘**，软体滚动带夹爪，92% 成功率 |
| Fast Heuristic Scheduling and Trajectory Planning for Robotic Fruit Harvesters with Multiple Cartesian Arms | ICRA 2025 | 多臂调度与轨迹规划 |
| A Novel Approach to Tomato Harvesting Using a Hybrid Gripper | ICRA 2025 | 番茄混合夹爪，80% 成功率 |
| Investigating Sensors and Methods in Grasp State Classification in Agricultural Manipulation | ICRA 2025 | 抓取状态分类，IMU+张力传感器 |
| A comprehensive control architecture for semi-autonomous dual-arm robots in agriculture | ICRA 2025 | 双臂 HQP 控制架构 |

### 2. IROS (IEEE/RSJ International Conference on Intelligent Robots and Systems)
- **级别**：机器人领域顶会，CCF-A
- **2024 时间**：2024-10-14 至 2024-10-18（阿联酋阿布扎比）
- **2025 时间**：2025-10-19 至 2025-10-25（中国杭州）
- **检索链接**：
  - 官网：https://iros2025.org/
  - IEEE Xplore：https://ieeexplore.ieee.org/xpl/conhome/1000002/all-proceedings

### 3. RSS (Robotics: Science and Systems)
- **级别**：机器人领域顶会，CCF-A
- **2024 时间**：2024-07-15 至 2024-07-19（荷兰代尔夫特）
- **2025 时间**：2025-07-21 至 2025-07-25（美国洛杉矶）
- **检索链接**：
  - 官网：http://www.roboticsconference.org/
  - 开放获取：https://www.roboticsproceedings.org/

### 4. CoRL (Conference on Robot Learning)
- **级别**：机器人学习顶会
- **2024 时间**：2024-11-06 至 2024-11-09（德国慕尼黑）
- **2025 时间**：2025-11（待定）
- **检索链接**：
  - 官网：https://www.corl.org/
  - PMLR：http://proceedings.mlr.press/

---

## 二、顶级期刊

### 1. Computers and Electronics in Agriculture
- **级别**：农业工程顶刊，IF 7.7（2024）
- **检索链接**：
  - ScienceDirect：https://www.sciencedirect.com/journal/computers-and-electronics-in-agriculture
  - 最新文章：https://www.sciencedirect.com/journal/computers-and-electronics-in-agriculture/articles-in-press

**2024-2025 蓝莓/采摘相关**：
| 论文 | 年份 | 内容 |
|------|------|------|
| AgriVision: A Benchmark Dataset for Advancing Real-World Robotic Vision in Densely Fruited Blueberry Crop | 2025 | **蓝莓分割基准数据集**，PVT-SN+SAM-RN |
| In-field blueberry fruit phenotyping with a MARS-PhenoBot and customized BerryNet | 2025 | **蓝莓表型分析**，BerryNet+SAM |
| A new occlusion-avoidance and dual-view fruit localization method with a 6-DoF manipulator | 2025 | 遮挡避免+双视角定位 |
| Efficient Occlusion Avoidance Based on Active Deep Sensing for Harvesting Robots | 2024 | 主动感知遮挡避免 |

### 2. Biosystems Engineering
- **级别**：农业工程期刊，IF 5.1（2024）
- **检索链接**：
  - ScienceDirect：https://www.sciencedirect.com/journal/biosystems-engineering

### 3. Journal of Field Robotics
- **级别**：机器人期刊，IF 4.5（2024）
- **检索链接**：
  - Wiley：https://onlinelibrary.wiley.com/journal/15564959

### 4. Robotics and Autonomous Systems
- **级别**：机器人期刊，IF 4.3（2024）
- **检索链接**：
  - ScienceDirect：https://www.sciencedirect.com/journal/robotics-and-autonomous-systems

---

## 三、关键论文总结（蓝莓/小果采摘）

### 蓝莓采摘（直接相关）

**1. CLASP (ICRA 2025)** — 最新蓝莓采摘机器人
- **作者**：Yixuan Xia, Yilin Cai, et al.
- **机构**：University of Georgia（UGA）
- **创新点**：
  - 集群级采摘（Cluster-Level），不是单果采摘
  - 软体滚动带夹爪（SARB-Gripper），模拟人手" rub "动作
  - 熟果脱落力 < 生果脱落力，实现选择性采摘
- **结果**：23/25 集群成功（92%），成本 $3326
- **链接**：https://arxiv.org/abs/2609.18051

**2. AgriVision (Computers and Electronics in Agriculture 2025)**
- **数据集**：https://springernature.figshare.com/articles/dataset/29222462
- **内容**：蓝莓分割基准，DB-1（1195 张精细标注）+ DB-3（合成）
- **模型**：PVT-SN + SAM-RN，DICE 0.71
- **你的复现**：DICE 0.7257（比论文高，因 512 resize 降低难度）

**3. BerryNet (Computers and Electronics in Agriculture 2025)**
- **作者**：Zhengkun Li et al.（UGA，和 CLASP 同组）
- **内容**：MARS-PhenoBot 平台，BerryNet 分割 + 成熟度分类
- **数据集**：Kaggle BSAIL（你下载的）
- **链接**：https://www.kaggle.com/datasets/zhengkunli3969/blueberry-segmentation-with-segment-anything-model

### 小果/成簇采摘（方法相关）

**4. 番茄采摘 (ICRA 2025)**
- 混合夹爪：软体手指 + 刚性外骨骼 + 切割器
- 语义分割区分生熟，关键点检测定位果蒂
- 80% 成功率，24.34s/周期

**5. 抓取状态分类 (ICRA 2025)**
- IMU + 张力传感器，Random Forest 100% 准确率
- 检测滑移、抓取失败、成功采摘

---

## 四、研究趋势总结（2024-2025）

| 趋势 | 代表工作 | 你的机会 |
|------|---------|---------|
| **集群级采摘**（不是单果） | CLASP (ICRA 2025) | 蓝莓成簇生长，集群级更符合实际 |
| **软体夹爪** | CLASP、番茄混合夹爪 | 蓝莓表皮脆弱，软体接触减少损伤 |
| **选择性采摘**（生熟判断） | CLASP 力控、番茄语义分割 | 你的 ripe_det + 力控可以结合 |
| **多臂协调** | 双臂苹果 (ICRA 2025) | 单臂效率低，双臂是趋势 |
| **主动感知/遮挡处理** | Active sensing (Comput. Electron. Agric. 2024) | 蓝莓枝叶遮挡严重，需要视角规划 |
| ** imitation learning** | HarvestPoint-ACT (arXiv 2026) | 需要大量演示数据，暂时不适合 |

---

## 五、你的论文定位建议

**当前工作**（YOLO + 3D 定位 + 采摘策略）属于**感知+规划**层面，距离 CLASP 的**集群级软体采摘**还有差距。

**可发论文的方向**：
1. **短期**：BSAIL + D435 数据融合，提升小果分割精度（Computers and Electronics in Agriculture）
2. **中期**：成簇蓝莓的 TSP 顺序优化 + 碰撞避免（ICRA/IROS）
3. **长期**：软体夹爪 + 力控选择性采摘（对标 CLASP，需要硬件）

**建议精读**：
- CLASP (ICRA 2025) — 蓝莓采摘最新 SOTA
- AgriVision (2025) — 数据集和 baseline
- BerryNet (2025) — 分割+成熟度

检索链接都给你了，自己去看原文。
