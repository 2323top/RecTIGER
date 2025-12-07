

# RecTIGER: ReChorus 框架下的 TIGER 模型复现

[![Python Version](https://img.shields.io/badge/python-3.10-blue)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Framework: ReChorus](https://img.shields.io/badge/Framework-ReChorus_2.0-red)](https://github.com/THUwangcy/ReChorus)
[![Paper: NeurIPS 2023](https://img.shields.io/badge/Paper-TIGER-green)](https://arxiv.org/abs/2305.05065)

## 📖 简介

**RecTIGER** 是基于清华大学 [ReChorus 2.0](https://github.com/THUwangcy/ReChorus) 框架扩展的推荐系统研究项目。本项目的主要目标是复现 NeurIPS 2023 论文 **"Recommender Systems with Generative Retrieval" (TIGER)**，将生成式检索范式（Generative Retrieval）引入到通用的序列推荐框架中。

不同于传统的“召回+排序”或基于 MIPS 的向量检索，TIGER 将推荐任务重构为 **Seq2Seq 的生成任务**。模型通过学习物品的 **Semantic ID（语义 ID）**，利用 Transformer 解码器自回归地生成用户下一时刻可能交互物品的 ID 序列。

本项目保留了 TIGER 语义索引的核心特性，同时无缝接入 ReChorus 的数据管道、训练器和评估模块，并添加了 **Label Smoothing** 等优化策略。

## ✨ 核心亮点

*   **无缝集成**：基于 ReChorus 的 `Reader/Runner/Model` 模块化架构重构，复用框架的高效数据加载与 Top-K 评估流程。
*   **语义 ID 适配**：实现了 `TIGERReader`，支持自动加载离线生成的 RQ-VAE 编码文件（`.npy`），并支持缺失编码时的确定性 ID 降级策略。
*   **生成式检索实现**：基于 HuggingFace `T5ForConditionalGeneration` 实现了 `TIGER` 模型类，支持 Encoder-Decoder 架构训练与 Beam Search 推理。
*   **性能优化**：
    *   在解码阶段引入 **Label Smoothing** 正则化，缓解在稀疏数据集上的过拟合问题。
    *   支持 `corpus.item_codes` 的延迟加载机制，避免修改原始数据管道。

## 📂 项目结构

```text
src/
├── models/
│   └── TIGER.py             # TIGER 模型核心实现 (继承自 SequentialModel)
├── helpers/
│   ├── TIGERReader.py       # 专用数据读取器，处理语义 ID (.npy) 加载与映射
│   ├── TIGERRunner.py       # 专用运行器，保留生成任务扩展接口
│   └── ... (ReChorus base files)
├── main.py                  # 统一入口
└── ...
data/
└── Grocery_and_Gourmet_Food/
    ├── Grocery_and_Gourmet_Food.inter  # 原始交互数据
    └── item_codes.npy                  # (可选) RQ-VAE 生成的物品语义编码
```

## 🚀 快速开始

### 1. 环境准备

本项目依赖 PyTorch 与 Transformers 库。

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
# 额外确保安装了 transformers
pip install transformers
```

### 2. 数据准备

请遵循 ReChorus 的标准数据格式（csv/txt），将数据放置在 `data/<DatasetName>/` 目录下。

**关于语义 ID (Semantic IDs):**
TIGER 依赖 RQ-VAE 生成的离散码本作为物品 ID。
*   **推荐方式**：将离线生成的 RQ-VAE 编码保存为 `.npy` 文件（例如 `item_codes.npy`），放置在数据集目录下。`TIGERReader` 会自动发现并加载。
*   **降级方式**：如果未提供 `.npy` 文件，`TIGERReader` 将根据 `codebook_k` 和 `num_codebooks` 自动生成确定性的伪语义编码，以保证代码可运行（仅用于调试流程，无语义增益）。

### 3. 运行训练

使用命令行参数直接启动训练，以下是在 Amazon Grocery 数据集上的示例命令：

```bash
python main.py \
    --model_name TIGER \
    --dataset Grocery_and_Gourmet_Food \
    --lr 1e-3 \
    --l2 1e-6 \
    --emb_size 64 \
    --num_layers 2 \
    --history_max 20 \
    --codebook_k 256 \
    --num_codebooks 4 \
    --tiger_code_path item_codes.npy \
    --label_smoothing 0.1 \
    --gpu 0
```

**关键参数说明：**
*   `--tiger_code_path`: 语义编码文件路径（可选，若不传则自动搜索）。
*   `--codebook_k`: 码本大小（默认 256）。
*   `--num_codebooks`: 每个物品对应的 Token 数量（语义 ID 长度，默认 4）。
*   `--label_smoothing`: 解码器训练时的标签平滑系数（我们的改进点，建议设为 0.1）。
*   `--beam_size`: 推理时的集束搜索宽度（默认 30）。

## 🧠 模型详解

### TIGER (Transformer Index for GEnerative Recommenders)

TIGER 的工作流程分为两阶段：

1.  **Semantic ID 生成 (离线)**:
    利用 **RQ-VAE (Residual-Quantized Variational AutoEncoder)** 对物品的文本内容（标题、描述等）进行编码。通过残差量化，每个物品被映射为一个由 $m$ 个离散码字组成的元组 $(c_1, c_2, ..., c_m)$。
    > *注：本项目主要关注第二阶段，即在线推荐部分，默认假设语义 ID 已通过 RQ-VAE 生成。*

2.  **生成式检索 (在线)**:
    *   **Encoder**: 将用户交互历史中的物品替换为对应的 Semantic ID 序列，输入 Transformer 编码器。
    *   **Decoder**: 基于编码器的上下文，预测用户下一个交互物品的 Semantic ID。
    *   **Inference**: 使用 Beam Search 生成概率最高的 $K$ 个语义 ID 序列，并通过前缀匹配检索回原始物品。

### 改进点：Label Smoothing

针对生成式检索在稀疏数据上容易“过度自信”导致过拟合的问题，我们在计算交叉熵损失时引入了 Label Smoothing：

$$ \mathcal{L} = (1 - \epsilon) \cdot \text{CE}(p, y) + \epsilon \cdot \text{Uniform}(K) $$

实验表明，设置 `epsilon=0.1` 能在 Grocery 数据集上带来显著的性能提升。

## 📊 实验结果

我们在 **Amazon Grocery** (稀疏) 和 **MovieLens-1M** (稠密) 数据集上进行了对比实验。

| Dataset | Model | HR@5 | NDCG@5 | HR@10 | NDCG@10 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Grocery** | GRU4Rec | 0.3710 | 0.2655 | 0.4763 | 0.2995 |
| | SASRec | 0.3729 | 0.2726 | 0.4684 | 0.3032 |
| | **TIGER** | **0.3934** | **0.2973** | **0.4855** | **0.3270** |
| **ML-1M** | **SASRec** | **0.5177** | **0.3834** | **0.6705** | **0.4326** |
| | TIGER | 0.1371 | 0.0816 | 0.2265 | 0.1101 |

**结论：** TIGER 在具有丰富语义且交互稀疏的场景下（Grocery）表现出 SOTA 性能，验证了语义索引的有效性；但在纯协同过滤信号主导的稠密场景下（ML-1M），单纯基于内容的生成式检索仍存在局限。

## 🤝 开发与贡献

欢迎提交 Issue 或 Pull Request：
*   **Bug 反馈**：请附带完整的参数设置和错误日志。
*   **功能扩展**：欢迎贡献针对 RQ-VAE 的在线训练模块或新的解码策略（如 Trie 约束解码）。

## 🔗 引用

如果本项目对您的研究有帮助，请引用原始论文与 ReChorus 框架：

```bibtex
@inproceedings{rajput2023recommender,
  title={Recommender Systems with Generative Retrieval},
  author={Rajput, Shashank and Mehta, Nikhil and Singh, Anima and others},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}

@article{wang2020rechorus,
  title={ReChorus: A Comprehensive Learning Framework for Recommendation},
  author={Wang, Chenyang and others},
  journal={arXiv preprint arXiv:2005.13602},
  year={2020}
}
```

## 📄 许可证

本项目遵循 MIT License。详情请参阅 [LICENSE](./LICENSE) 文件。
