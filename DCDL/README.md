# DCDL 论文代码重建

依据本目录的 [25-DCDL.pdf](25-DCDL.pdf)：*Dual-view collaboration fusion on diffusion learning for bundle recommendation*，Information Fusion 122 (2025), 103154，[DOI](https://doi.org/10.1016/j.inffus.2025.103154)。

这是从论文重建的独立实现。本地没有作者发布的 DCDL 源码及最终配置，因此不能把这里的实现、初始参数或实验结果称为作者原版或作者最优结果。当前只提供模型、配置、单次训练和评测；尚未开展调参、正式训练或五次独立实验。

## 文件与入口

| 文件 | 作用 |
| --- | --- |
| `model.py` | 双视图编码器、CFD、用户边界及联合损失 |
| `data.py` | 原始 ID 数据加载、负采样、稀疏图预处理 |
| `configuration.py` | 默认值、参数和路径校验 |
| `config.yaml` | Youshu、NetEase、iFashion 工程初始配置 |
| `train.py` | 单次训练、实验记录、checkpoint 保存及只读重评 |
| `test_reconstruction.py` | 公式数值核对和隔离的合成数据集成测试 |
| `requirements.txt` | Python 依赖；CUDA 对应的 PyTorch 请单独选择 |

从项目根目录执行；也支持进入 `DCDL` 后执行 `python train.py ...`。`data_path`、`--data-path` 和输出路径中的相对路径统一相对**项目根目录**解析；`--config` 和 `--evaluate` 按命令行当前目录解析。

依赖文件对 Python 3.13 以下环境约束 `numpy<2`，兼容论文使用的旧版 PyTorch 二进制；Python 3.13 及以上应使用支持当前 Python 的较新 PyTorch，并允许相应 NumPy 版本。本次实际验证环境为 Python 3.13、PyTorch 2.11.0+cpu、NumPy 2.4.3。版本约束依据：[PyTorch NumPy 2 支持说明](https://github.com/pytorch/pytorch/issues/107302)、[NumPy 二进制兼容说明](https://numpy.org/doc/2.0/user/troubleshooting-importerror.html)。本次未安装或改动现有 Python 环境。

```powershell
# 仅检查配置和六份数据，不创建训练输出
python -B DCDL/train.py -d Youshu --check-config
python -B DCDL/train.py -d NetEase --check-config
python -B DCDL/train.py -d iFashion --check-config

# 单次训练示例；开发验证阶段没有执行这些正式训练命令
python DCDL/train.py -d Youshu -g 0 --device cuda --note "工程初始配置，单次运行"
python DCDL/train.py -d NetEase -g 0 --device cuda
python DCDL/train.py -d iFashion -g 0 --device cuda

# 重评示例：替换为实际 checkpoint 路径；不追加实验记录
python -B DCDL/train.py --evaluate DCDL/log/Youshu/EXECUTION_ID/best.pt --device cpu

# 小规模开发验证；临时数据、日志和记录均自动清理
python -B -m unittest DCDL.test_reconstruction -v
```

常用覆盖项：`--seed`、`--epochs`、`--data-path`、`--num-workers`、`--batch-size-test`、`--output-dir`、`--record-file`、`--result-json`、`--note`、`--quiet`。GPU 可见性在导入 PyTorch 前设置，`-g 2` 表示物理 GPU 2，程序内使用逻辑 `cuda:0`；`--device cuda` 在没有 CUDA 时直接报错。只读重评使用 checkpoint 的模型配置，可覆盖数据根目录和评测 batch size。

## 公式与实现对应

设训练 UB 矩阵为 `X`，UI 为 `Y`，BI 为 `Z`，维度分别为 U×B、U×I、B×I。三类实体及用户边界向量均参与联合优化。

| 论文 | 实现 |
| --- | --- |
| 式 (1)–(3) | 原始 UB 双向边加自环；单层单头 GAT；逐目标节点进行注意力 softmax |
| 式 (4)–(5) | `P=1[XXᵀ>k1]`、`Q=1[XᵀX>k1]`；保留满足条件的对角元素；使用全局 Frobenius 范数 |
| 式 (6)–(7) | 共现矩阵按行归一化，再加 ReLU 残差；不建立 I×I 稠密矩阵 |
| 式 (8) | UI、BI 邻居均值聚合；无邻居时输出零 |
| 式 (13) | 两视图相加得到最终用户、bundle 表示 |
| 式 (14)、(17)、算法 1 | 从 bundle-view 用户表示加噪，以融合用户表示为条件，输出中间表示并计算加权重建损失 |
| 图 3 | `[time(d), noisy(d), fused(d)] → 4d → GELU → Dropout → 2d → GELU → d` |
| 式 (18)–(20) | 融合表示内积用于排序；中间表示只生成训练边界；UIB + CFD + 显式 L2 联合优化 |

**关系增强的等价稀疏实现。** 令 `A=[[P,X],[Xᵀ,Q]]`，论文的 `R` 是 `A` 的列置换，所以 `RRᵀ=A²`。前向计算为 `A(AE_g)/||A²||_F`。范数按行分块计算 `A²` 并累加平方和，分块大小只控制预处理内存。这里没有用度归一化、行归一化或 LightGCN 替换论文的全局范数。

**物品共现的等价分解。** `C_u E = Yᵀ(YE) / [Yᵀ(Y1)]`，`C_b` 同理。分母是共现矩阵行和，不是原始物品度数；ReLU 放在完整共现乘积之后。零分母对应零共现行，以 1 作安全除数，保留原始物品残差。

**静态稀疏乘法。** 对固定图矩阵 `A`，`AE` 对 `E` 的反向传播直接使用已保存的转置 `Aᵀ`。CPU 运行时缓存 CSR 格式，减少重复的 COO 计算与转置格式转换；CUDA 路径使用 COO。图本身不学习，全部可学习参数的梯度继续传播。缓存不写入 checkpoint；改变模型 dtype 或设备后会按当前图重新生成 CPU 缓存。该计算路径已通过稠密公式对照和数值梯度检查。

**CFD 只在训练使用。** 每轮为全部声明用户采样 `t∈{1,…,T}` 与高斯噪声，同轮各更新共享这组采样；每次更新都用当前编码器表示重新计算 CFD。包括没有 UB 训练历史的用户。目标表示和条件表示均不 detach。没有扩散预训练、交互图重建或反向多步生成；评测只运行双视图编码器。

## 论文不完整或有歧义之处的固定约定

这些约定属于工程重建，不是从作者源码核实的设置。它们固定在本实现中，并由源码 SHA256 和实现版本 `dcdl-paper-reconstruction-v1` 标识。

1. GAT 保留式 (1) 的 d×d 特征投影；激活选择 ELU，LeakyReLU 斜率 0.2；对输入特征和注意力系数应用 dropout 0.2。论文没有完整指定这些细节。CFD dropout 按图 3 放在第一层 GELU 后。线性层和 embedding 使用 He 正态初始化，线性偏置置零，注意力参数使用 Xavier 初始化。
2. 边界采用式 (18) 的字面用户索引：每个用户一个可学习 `w_u∈R^d`，`s_u=<e_mid_u,w_u>`，无偏置。论文没有公开该层源码；没有把它暗中替换成共享线性层。
3. 式 (19) 把 `φ` 写为 LnSigmoid，但直接最小化印刷形式会给出错误的梯度方向。这里按其引用的 [UIB 原论文](https://arxiv.org/abs/2111.11026) 使用 `softplus(s-y_pos)+alpha_1*softplus(y_neg-s)`；正样本分数梯度为负、负样本为正。没有把 `alpha_1` 改成归一化的负例比例。
4. 论文写 beta 从 0 到 0.03；包含零端点会使前两步出现未定义权重。这里使用正端点 `linspace(0.0001,0.03,T)`。即使端点为正，首步后验方差仍为零，因此首步单独采用 `variance_1=beta_1`，得到有限的 `w_1=1/(2 beta_1)`；其余步骤严格使用 `posterior_variance_t=beta_t*(1-alpha_bar_(t-1))/(1-alpha_bar_t)`，`w_t=alpha_bar_(t-1)/(2 posterior_variance_t)`。误差项保留论文的 `||x0-sqrt(alpha_t)*f_theta||²`，没有替换成无权 MSE 或噪声预测损失。这一首步处理是工程约定，不能称为论文唯一解。
5. 为明确损失尺度，令 `N` 为训练 UB 边数，将整条式 (20) 除以 `N`：训练 batch 使用 `mean(UIB) + lambda_1*sum_all_users(diffusion)/N + lambda_2*sum_all_parameters(theta²)/N`。这是完整数据目标的常数缩放，保留各项相对系数；不是分别把三项任意取均值。扩散平方误差按特征求和；L2 包括全部可学习权重和偏置。AdamW 的额外 weight decay 默认设为 0，避免未说明的双重正则。
6. 每轮随机打乱全部训练边，保留不足 batch size 的尾 batch，避免丢失训练边或小数据集无法验证。此处与现有 DCBR 的 `drop_last=True` 不同，属于训练迭代约定；评测和选模规则保持统一。没有增加早停、学习率调度或自动恢复。

## 初始参数与实验协议

`config.yaml` 的 d=64、batch=2048、dropout=0.2 来自论文实验设置。各数据集的 k1/T/alpha_1/lambda_1 参考敏感性图得到工程起点；将它们组合起来并不代表作者使用过这个联合配置。lr=0.001、lambda_2=0.0001 在论文候选范围内；seed=0、100 轮、评测 batch=256、每 5 轮评测是工程设置。配置尚未经过调参验证。

| 数据集 | k1 | T | alpha_1 | lambda_1 | 当前 UB train / tune / test |
| --- | ---: | ---: | ---: | ---: | --- |
| Youshu | 4 | 10 | 4 | 0.15 | 34416 / 5189 / 9746 |
| NetEase | 6 | 15 | 4 | 0.15 | 217864 / 27857 / 56555 |
| iFashion | 8 | 20 | 8 | 0.25 | 1213676 / 149007 / 317025 |

读取根目录 `datasets/<dataset>` 的原有六份文件，不重新编号、不重新划分。现有文件统计与论文宣称的 8:1:1 及部分总数不完全一致，所以后续实验属于当前统一数据协议下的复现，不能直接宣称严格重现论文的数据协议。

每个 UB 正例只采一个均匀负例，拒绝原始训练正例。验证、测试对全部 bundle 排序，仅屏蔽**原始 UB 训练边**，测试不屏蔽验证正例。每个 K 独立调用 `topk`，Recall/NDCG 按有该划分正例的用户作宏平均，算术方式对齐根目录评测；支持 K=[10,20,40,80]。

每完成 5 轮评测一次。最佳初值为验证 `(Recall@20,NDCG@20)=(0,0)`；只有两者**相对同一已保存 checkpoint 同时严格提高**才保存。测试从不决定保存。最终全部 K 的 val/test 指标都来自同一个最佳 checkpoint，不拼接各指标历史最高值。若整个预算内没有合格 checkpoint，运行记为 FAILED 并明确报告。轮数不是 5 的倍数时，尾部不足 5 轮不追加评测。

固定 Python、NumPy、PyTorch 和 DataLoader 的种子；关闭 cuDNN benchmark。CUDA 稀疏归约可能仍有非确定性，固定 seed 不保证不同设备或库版本逐位一致。正式结果仍须后续独立多次实验；本次没有生成多 seed 驱动或调参计划。

## 输出与已完成验证

默认每次运行独立写入 `DCDL/log/<dataset>/<UTC时间_UUID>/`：`config.yaml`、`training.log`、TensorBoard、`best.pt`、`best_config.json`、`result.json`。时间字段采用北京时间 ISO 格式；目录执行 ID 使用 UTC。默认向 `DCDL/实验记录.txt` 追加完整单次实验记录，包含状态、实际配置、预算完成情况、seed、数据和源码指纹、设备、损失、同一最佳轮的全部指标、备注及错误信息；失败和中断同样记录。

Checkpoint 包含模型、优化器、实际配置、静态图及范数、数据指纹、源码指纹、最佳轮次和对应指标。`best_epoch` 从 0 开始，同时保存从 1 开始的轮数。重评通过数据 SHA256 检查确保内容一致，从 checkpoint 恢复图，不重新生成关系图，不追加正式实验记录。实现没有提供断点续训入口；再次训练会新建执行目录。`source_files_match` 报告当前代码/基础配置是否与保存时一致。

2026-10-28 检查结果：13 项测试全部通过，覆盖公式输出及梯度、静态稀疏乘法数值梯度、缓存 dtype 转换、扩散首步、联合损失、项目评测一致性（含并列分数和无正例用户）、选模、负采样及训练保存重评。Windows 下两个 DataLoader worker 的训练也已通过。

三个真实数据集均通过原始文件、ID、重复边和划分检查。Youshu 完整训练 1 轮（17 个 batch），NetEase 完整训练 1 轮（107 个 batch），均完成全量验证/测试、checkpoint 保存和独立进程重评，全部指标完全一致；优化器更新次数与 batch 数一致，TensorBoard 和结果文件完整。iFashion 完整图已构建，至少 300 个 batch 的训练保持有限损失及梯度；按用户加快检查的要求停止长时间 CPU 整轮验证，并另外完成真实数据单步更新、保存和独立进程重载，4 个用户对全部 27694 个 bundle 的分数逐位一致。iFashion 未完成整轮训练或全量验证/测试，不将此次单步快照记为最佳实验 checkpoint。

验证现场位于本地 `DCDL/log/validation/20261027_234928/`，与正式实验记录分开；三个默认完整配置保持不变。当前验证环境仅有 CPU，尚未验证 CUDA、完整 100 轮预算或论文复现成绩。本次没有开展调参。
