```markdown
# RecTIGER

[![Python Version](https://img.shields.io/badge/python-3.10-blue)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![arXiv: ReChorus2.0](https://img.shields.io/badge/arXiv-ReChorus-%23B21B1B)](https://arxiv.org/abs/2405.18058)

简介
--
RecTIGER 是在 ReChorus2.0 框架基础上扩展的推荐研究库。本仓库的主要工作是将新提出的 TIGER（Targeted/Temporal/Transformer-oriented/Graph/ENhanced/Ranker —— 根据实际含义替换）模型接入到 ReChorus 中，提供模型实现、训练/评估脚本、示例配置与基准实验记录，便于研究人员在统一框架下比较与复现 TIGER 与其他模型的表现。

核心亮点
--
- 基于 ReChorus2.0 的模块化架构（Reader / Runner / Model），方便复用已有数据预处理与评测流程。
- 新增 TIGER 模型实现（src/models/tiger/TIGER.py）并与 ReChorus 的训练/评估流水线无缝对接。
- 示例配置与 demo 脚本：configs/tiger/*.yaml, scripts/run_tiger.sh（占位，请填入实际文件）。
- 支持的任务：Top-k 推荐、CTR 预测、Impression-based reranking（与 ReChorus 原任务对齐）。
- 高效训练：多线程数据准备、GPU 加速兼容、可接入已有评估优化器与损失函数。

示意结构
--
- src/
  - models/
    - tiger/               # TIGER 模型实现与辅助模块
      - TIGER.py
      - tiger_utils.py
  - helpers/
    - BaseReader.py
    - BaseRunner.py
    - BaseModel.py
  - configs/
    - tiger/               # TIGER 的默认配置（超参 / 网络结构 / 优化器等）
- data/                    # 数据与预处理示例
- docs/                    # 文档与 demo 结果
- scripts/                 # 快速运行脚本（demo / 训练 / 评估）

快速开始
--
1. 环境（示例）
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. 下载并准备数据（示例）
- 请参考 data/README.md 以准备 Top-k / CTR / Impression 数据。
- 若使用示例 Amazon/MIND/MovieLens 数据，请把预处理后的文件放到 data/<DATASET>/ 下。

3. 运行 TIGER demo（占位命令，请根据实际脚本替换）
```bash
# 训练示例
python main.py --config configs/tiger/tiger_topk.yaml --task topk --model TIGER

# 评估示例
python main.py --config configs/tiger/tiger_topk.yaml --task topk --model TIGER --eval_only 1 --checkpoint path/to/checkpoint.pt
```

配置说明
--
- configs/tiger/*.yaml 包含：
  - model.architecture: 模型结构相关字段（层数、维度、注意力 heads 等）
  - training.optimizer: 优化器与学习率
  - training.batch_size, training.epochs
  - data.reader: 指定使用的 Reader（BaseReader / SeqReader / ContextReader 等）
  - task.mode: topk / ctr / rerank
- 在 README 中保留常用参数表与推荐默认值（请将实验中使用的最终参数填入下表）。

TIGER 模型要点（占位，替换为模型具体描述）
--
- 模型核心思想：说明 TIGER 的组成（例如：Transformer 编码器 + 时序增强模块 + 图关系融合 + 目标蒸馏/对比损失）
- 输入/输出格式：模型期望的特征（user_id, item_seq, item_meta, timestamps, candidate_list 等）
- 训练损失：List-wise / BPR / BCE / Softmax 等（写明默认组合）
- 已实现文件：src/models/tiger/TIGER.py（主类）、src/models/tiger/README.md（实现细节与数学式）

数据格式（沿用 ReChorus 规范）
--
见 data/README.md（项目已保留原有数据格式说明）。简要：
- Top-k train.csv: user_id \t item_id \t time
- test/dev: user_id \t item_id \t time \t neg_items（或无 neg_items 则 test_all）
- CTR: user_id \t item_id \t time \t label
- Impression: user_id \t item_id \t time \t label \t impression_id
- 可选：item_meta.csv, user_meta.csv, situation features（c_ 开头）

示例与实验结果（占位）
--
- 在 docs/demo_scripts_results/ 中加入针对 TIGER 的训练曲线、性能表格（HR@K、NDCG@K、AUC、Logloss 等）。
- 请将 demo 脚本与运行日志上传到 docs/ 并在此处引用。

开发与贡献
--
欢迎以 Issue / PR 方式贡献：
- 报告 bug（请附带复现步骤与最小可运行脚本）
- 提交新的数据 reader、runner 或优化器
- 补充 TIGER 的 ablation 配置与基线对比

引用
--
如果本仓库或 TIGER 的实现对你的工作有帮助，请引用：
- ReChorus2.0 论文：Li et al., ReChorus2.0 (arXiv)
- 请在此补充 TIGER 原始论文引用（占位）

许可证与联系
--
- 许可证: MIT（详见 LICENSE）
- 联系: 项目维护者（占位：请填入邮箱或 GitHub 用户名）

附录 — 待补充项（请在合并前完成）
--
- 填写 configs/tiger 下的完整 YAML 示例
- 补充 scripts/run_tiger.sh、demo 数据与复现实验命令
- 在 docs/ 中加入 TIGER 设计文档与超参数敏感性分析
- 将模型实现路径、API 调用示例和 checkpoint 下载链接写入 README

```
