# 基于机器学习的CT图像超分技术 (CT Image Super-Resolution)

本项目旨在开发和研究基于深度学习（特别是 ResNet 架构）的 CT 图像超分辨率增强算法。

## 项目结构

- `ct-superRes/`: 核心代码库，包含模型定义、训练策略、数据处理及评估工具。
- `finalThesis/`: 毕业论文及相关文档。
- `relevant_essays/`: 技术调研与临床应用分析的相关文献。

## 数据集信息 (Datasets)

本项目支持并建议使用以下高质量公开 CT 数据集进行训练与验证：

1. **RPLHR-CT Dataset**
   - **来源**: [GitHub - smilenaxx/RPLHR-CT](https://github.com/smilenaxx/RPLHR-CT)
   - **下载**: [Zenodo Records](https://zenodo.org/records/17239183)
   - **特点**: 包含真实的配对薄层 (1mm) 与厚层 (5mm) CT 扫描数据，是评估三维超分辨率算法的基准数据集。

2. **CT-Super-Resolution Dataset**
   - **来源**: [GitHub - labcisne/CT-Super-Resolution](https://github.com/labcisne/CT-Super-Resolution)
   - **特点**: 提供了用于超分辨率研究的观测数据与地面真值 (Ground Truth) 图像。

3. **Mayo Clinic Low Dose CT (LDCT)**
   - **来源**: [The Cancer Imaging Archive (TCIA)](https://wiki.cancerimagingarchive.net/display/Public/LDCT-and-Projection-data)
   - **特点**: 包含 299 名患者的临床 CT 检查数据，提供常规剂量与低剂量配对图像。

## 快速开始

请参考 `ct-superRes/README.md` 获取详细的安装、数据准备、训练及实验运行说明。
