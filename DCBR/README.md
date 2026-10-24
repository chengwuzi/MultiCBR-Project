# DCBR：iFashion 统一协议复现

本目录外层是独立移植实现；`raw/` 保留 Jiuqiang Li 的 DCBR 作者开源项目，仅供对照。当前只接入第一种情况的 iFashion，尚未开发其他数据集的搜参脚本。项目根目录的 ReCoBR 文件不改动。

## 文件与依据

- `train.py`：独立包含 DCBR、DNN、GaussianDiffusion、数据加载、训练、评测和结果保存，不在运行时导入 `raw` 或根目录训练器。
- `config.yaml`：合并作者 `raw/configs/overall.yaml` 和 `raw/configs/models/DCBR.yaml` 的 iFashion 参数。
- `实验记录.txt`：每次训练结束（成功、失败或正常处理的中断）后追加一条第一种情况的实验记录，初始为空。
- `test_reproduction.py`：与原始实现及 ReCoBR 评测的数值对照、小数据训练与检查点重载测试。

## 保留与适配

| 内容 | 处理 |
|---|---|
| DCBR 推荐模型、扩散 DNN、扩散方程、ELBO/BLCC、对比损失 | 保留作者实现的计算与权重含义 |
| 每轮顺序 | 先训练扩散模块一遍，再重建 UB 图，再训练推荐模型一遍 |
| 扩散输入及候选 | 原始训练交互向量；包含全部用户，重建 Top-K 在全部组合中选择 |
| 重建图 | 保留作者的自环与对称度归一化 |
| 最终评分 | 融合 UI、BI 表示；UB 表示参与训练对比，保持作者设计 |
| 正则及优化器 | 模型内部显式 L2；推荐和去噪 Adam 均 `weight_decay=0`，防止重复施加正则 |
| 模型参数及训练轮数 | 保留作者 iFashion 设置，包括扩散 140 步、K=1、tau=0.08、训练 50 轮 |
| 随机种子 | 默认 0，沿用作者训练入口；可用 `--seed` 显式覆盖并记录。根目录 ReCoBR 的 iFashion 配置为 2026，并未声称两者默认种子相同 |
| 数据 | 读取根目录 `datasets/iFashion`；训练样本、负采样和评测屏蔽均依据原始训练交互 |
| 评测 | 对齐 ReCoBR 的全量排序、Recall/NDCG、有效用户口径、训练交互屏蔽及 Top-K `[10,20,40,80]`；仍每 5 轮评测 |
| 最佳检查点 | 统一采用作者原规则：验证 Recall@20、NDCG@20 同时严格超过此前保存检查点的对应值才更新；相等不更新，Top-40 和测试结果不参与选模 |
| 时间及产物 | 北京时间精确到秒；每次运行独立 ID、配置快照、代码和数据摘要、日志、TensorBoard、检查点、JSON 和 TXT 汇总 |

选模规则现与 DCBR、MultiCBR 作者原代码一致，ReCoBR 训练入口也统一为该规则。两项初始最佳验证值均为 0，每次接受更新后，两项基准一起取自该检查点；不维护各自独立的历史最大值，不进行指标相加。JSON 和 TXT 保存两项验证选模指标，不再使用相加的 selection_score。统一协议下的实测结果不保证逐位重现作者论文报告值。ReCoBR 现有 iFashion 配置训练 30 轮，本接入保留 DCBR 作者的 50 轮；比较报告需保留这些实际设置，不能写成相同训练轮数。

设备改为显式传递，不再硬编码 `.cuda()`；稀疏张量使用当前构造接口。扩散数据逐行转换为相同的 float32 稠密交互向量，避免先分配完整 U×B 矩阵及其张量副本；不改变用户集合、向量内容、批大小或两遍随机遍历方式。Windows 默认数据加载进程数为 0，其他系统为 10，可显式覆盖；实际值保存在配置中。重放实际配置快照时沿用其中的进程数，命令行 `--num-workers` 优先于快照设置。扩散加载仍使用 0 个工作进程。

## 运行

在项目根目录中检查配置和路径（不训练、不写实验产物）：

```bash
python DCBR/train.py --check-config
```

在装有可用 CUDA 版 PyTorch 的环境运行完整作者配置：

```bash
python -u DCBR/train.py -d iFashion -g 0 --device cuda --note "DCBR iFashion 作者配置，ReCoBR统一评测与选模"
```

也可以进入 `DCBR` 后运行 `python -u train.py ...`；默认配置位置以及数据、产物路径不依赖当前工作目录。显式传入的 `--config`、`--evaluate`、`--result-json` 按当前目录解释；配置中的 `data_path` 和 `--output-dir`、`--record-file` 的相对路径按项目根目录解释。

可用参数包括 `--seed`、`--epochs`、`--num-workers`、`--quiet`、`--config`、`--data-path`、`--output-dir`、`--record-file` 和 `--result-json`。任何实验参数覆盖都会写入实际配置。单独指定 `--epochs` 时，仍需达到至少一次评测间隔；不要把不同训练预算下的结果当作完全相同条件的实验。

## 结果与复核

每次运行产生 `DCBR/log/iFashion/<执行ID>/`，包含：

- `config.yaml`：完整实际配置快照，可作为 `--config` 重放，实体数量和设备等运行字段会重新计算。
- `training.log`、`tensorboard/`：训练与评测信息。控制台进度由 tqdm 显示，`--quiet` 可关闭批次进度，逐轮摘要仍写入日志。
- `best.pt`、`best_config.json`：同一最佳轮次的推荐参数、去噪参数、扩散状态、重建图、优化器状态、配置及数据指纹。当前入口支持重新评测，尚不提供断点续训命令。
- `result.json`：执行状态、实际配置、数据指纹、最佳轮次与指标、产物路径。失败或中断后的已有最佳结果不代表完整训练成功。

`DCBR/实验记录.txt` 保存追加式可读汇总，指标均对应验证集选中的同一检查点，明确记录从 0 计数的 best_epoch 和从 1 计数的训练轮次。根目录 `工程记录.txt` 不由程序自动修改。`--check-config` 和 `--evaluate` 不新增训练实验记录。

重算已保存检查点的验证和测试结果：

```bash
python DCBR/train.py --evaluate DCBR/log/iFashion/<执行ID>/best.pt -g 0 --device cuda
```

移动到另一台机器后可用 `--data-path ./datasets` 指定本机数据位置；数据文件指纹必须与训练时一致。评测会根据原始 UI/BI 图重算最终表示，遵循作者推理逻辑。重建 UB 图已保存用于追溯，但作者最终评分不直接使用 UB 视图。

## 开发验证

```bash
python -m unittest discover -s DCBR -p test_reproduction.py -v
```

测试验证原始参数保留、两轮扩散和推荐更新的损失/梯度/参数/评分一致性、不同扩散采样模式、逐行输入与重建图一致性、ReCoBR 指标及负采样一致性、完整小数据训练/追加记录/检查点重载，以及运行失败记录。

原始扩散实现硬编码 CUDA。CPU 对照测试仅将原始张量的 `.cuda()` 调用临时映射为恒等操作，不改变原始公式、数据类型或磁盘文件。这些测试证明所覆盖路径的 CPU 数值与流程一致性；真实 iFashion 的 50 轮训练及 GPU 数值行为仍需在目标环境运行验证。
