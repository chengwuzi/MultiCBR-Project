# MultiCBR：统一框架下的作者配置复现

本目录外层独立移植 `raw/models/MultiCBR.py`，保留作者在 Youshu、NetEase、iFashion 上的模型与超参数。三者均属于“情况一”：作者已有目标数据集配置，先运行原配置，不开展参数搜索。`raw/` 仅供对照，正式训练不导入其中的代码，也不依赖 ReCoBR 或 DCBR 的运行模块。

对应论文为本目录 `24-MultiCBR.pdf`：*MultiCBR: Multi-view Contrastive Learning for Bundle Recommendation*，ACM TOIS 42(4)，2024，DOI:10.1145/3640810。

## 文件

| 文件 | 职责 |
|---|---|
| `model.py` | 原作者模型的独立移植 |
| `data.py` | 原始 ID 数据读取、输入检查、负采样与 DataLoader |
| `configuration.py` | 单实验配置校验、路径解析、源码／配置摘要 |
| `config.yaml` | 三个数据集各自完整的作者配置及工程运行设置 |
| `train.py` | 单次训练、统一评测、验证选模、检查点重评和运行记录 |
| `run_repeated.py` | 指定数据集及次数，以不同随机种子串行训练，保存计划、支持恢复并汇总均值／标准差 |
| `实验记录.txt` | 正式情况一实验的追加式汇总，开发验证不写入此文件 |
| `test_reproduction.py` | 作者数值对照、公共协议和小数据训练／保存／重评检查 |
| `requirements.txt` | 依赖列表；CUDA 对应的 PyTorch 单独选择 |

## 保留的模型与训练机制

- 三组 Xavier 正态初始化的用户、组合、物品嵌入；UB/UI/BI 三个视图采用作者的双向传播与对应邻居均值聚合。
- 保留传播中的逐层 L2 归一化及配置化层融合；第零层仍使用原始嵌入。归一化后的层表示只用于汇总，后续传播的中间量按原代码保留。
- 按 `[UB,UI,BI]` 加权融合用户和组合表示，再计算内积评分；不采用三个分支分别打分再求和的替代方案。
- 保留 BPR、融合后用户与正组合自对比损失的平均值，以及 `loss = BPR + c_lambda * contrastive`。批内重复 ID 不去重，不增加融合前对比、扩散或残差聚合。
- 正则由作者的 `Adam(weight_decay=l2_reg)` 实现，不额外添加显式 L2，不改成 AdamW。
- 默认 Noise；ED、MD、OP 分支保留原实现。Noise 的 ratio 是扰动幅度，不是删边概率；ED 在原代码的批次位置更新图，不擅自移至每轮开始。
- 均匀采样一个负组合，仅拒绝原始 UB 训练正例；训练按交互行打乱，`drop_last=True`。不使用验证／测试交互排除训练负例。

必须区分论文与开源实现：论文式(11)描述两份增强表示，但作者代码的 `cal_loss` 实际将同一份融合表示同时作为对比两端。本移植忠实保留该代码，不自行改成两次独立增强。论文描述简单层平均，开源配置实际提供不相等的逐层系数，且传播实现包含逐层归一化；这里沿用作者代码与配置。论文正文候选集合也没有包含开源 Youshu 最终配置中的 `c_lambda=0.04`、`c_temp=0.23`，二者按开源配置保留，不用正文网格替换。

模型文件的兼容调整仅为 `torch.sparse_coo_tensor` 构造和 `np.asarray` 替代旧 `.A` 转换，不改变数学公式、参数、随机采样或训练机制。图和融合系数在构造时按指定设备生成；重评根据已核对的原始数据与保存的实际配置重新构建确定性的推理图。

## 三个作者配置与工程设置

| 数据集 | 训练轮数 | `c_lambda` | `c_temp` | 模态融合 `[UB,UI,BI]` | Noise 幅度 `[UB,UI,BI]` |
|---|---:|---:|---:|---|---|
| Youshu | 100 | 0.04 | 0.23 | `[0.35,0.15,0.5]` | `[0.2,0.1,0.1]` |
| NetEase | 50 | 0.05 | 0.2 | `[0.5,0.2,0.3]` | `[0,0,0.2]` |
| iFashion | 50 | 0.1 | 0.2 | `[0.1,0.2,0.7]` | `[0.05,0,0.15]` |

共同保留作者的64维、两层传播、训练与评测 batch=2048、lr=0.001、l2_reg=1e-5、单负样本、每5轮评测，以及各数据集全部逐层权重。不能用根目录 ReCoBR 配置代替本目录 MultiCBR 配置。

作者入口未固定随机种子。本工程增加默认 `seed=0` 和 `num_workers=0`，分别用于可追溯运行和跨平台数据加载；它们不是作者提供的最优设置。支持显式覆盖种子、预算、workers和评测 batch，全部实际值进入快照及记录。原作者训练 workers=10、评测 workers=20，这里不作为模型机制固定。

论文报告三次训练均值。本工程提供单次实验入口及多次随机种子批量入口，实际重复次数由命令行指定；不搜索最佳种子，不按成绩筛选运行。训练轮数、重复次数等与已有 ReCoBR 不同的事实要如实说明。固定种子不保证不同硬件／CUDA／库版本逐位一致。

列表参数每项必须仅有一个值，每次命令仅产生一个实验；多值直接报错，不隐式展开未登记的笛卡尔积。配置可重放保存的实际快照，重复的 YAML 键、未知参数及冲突的列表／标量值均会被拒绝。

## 公共数据与评测选模

读取根目录 `datasets/<dataset>` 的原六份文件，保留全部声明 ID，不删除无 UB 交互用户，不重新划分。检查实体范围、重复交互、UB 划分重叠、负采样可行性及完整训练批次。

| 数据集 | 用户 / 组合 / 物品 | UB 训练 / 验证 / 测试 |
|---|---|---|
| Youshu | 8039 / 4771 / 32770 | 34416 / 5189 / 9746 |
| NetEase | 18528 / 22864 / 123628 | 217864 / 27857 / 56555 |
| iFashion | 53897 / 27694 / 42563 | 1213676 / 149007 / 317025 |

部分交互总量与论文表1不同，不能仅凭同名或相同实体数宣称划分完全相同。后续结果属于本项目统一数据协议下的作者代码复现。Youshu 三个 UB 划分实际涉及2959个用户，模型仍保留8039个用户。

评测对所有组合排序，按原项目减去原始训练 mask×1e8，仅屏蔽 UB 训练正例；测试不屏蔽验证正例。Recall/NDCG 按对应划分有正例用户作宏平均。TopK 固定 `[10,20,40,80]`，每个 K 独立调用 `topk`，保留并列分数时与 ReCoBR 一致的计算口径。

单次训练的初始最佳验证指标为 `(Recall@20,NDCG@20)=(0,0)`。两项必须同时严格超过同一已保存检查点才更新，相等不更新；Top40及测试不选epoch。所有K的验证／测试结果均来自同一个检查点。每5轮评测，尾部不足5轮不额外评测。训练预算内没有合格检查点则记录FAILED。这里没有跨配置选参或搜参控制器。

## 运行

推荐从项目根目录执行；也支持进入 `MultiCBR` 后执行 `python train.py ...`。数据、输出、记录及额外结果 JSON 的相对路径按项目根目录解释；显式 `--config`、`--evaluate` 按当前工作目录解释。默认配置通过脚本位置定位。

```powershell
# 只读检查配置和完整数据：不建训练目录，不写实验记录
python -B MultiCBR/train.py -d Youshu --check-config
python -B MultiCBR/train.py -d NetEase --check-config
python -B MultiCBR/train.py -d iFashion --check-config

# 三条独立的正式训练命令；按服务器实际设备选择物理GPU号
python -u MultiCBR/train.py -d Youshu -g 0 --device cuda
python -u MultiCBR/train.py -d NetEase -g 0 --device cuda
python -u MultiCBR/train.py -d iFashion -g 0 --device cuda

# 显式运行种子／备注示例；不将此例解释为已确定重复实验方案
python -u MultiCBR/train.py -d Youshu -g 0 --device cuda --seed 0 --note "作者配置，统一评测，单次运行"

# 用实际config.yaml快照重放：会新建执行，保留旧产物
python MultiCBR/train.py -d Youshu --config MultiCBR/log/Youshu/EXECUTION_ID/config.yaml -g 0 --device cuda

# 只读重评：不创建训练输出，也不追加实验记录
python -B MultiCBR/train.py --evaluate MultiCBR/log/Youshu/EXECUTION_ID/best.pt -g 0 --device cuda
```

`-g` 指定物理GPU，导入PyTorch前设置可见性，程序内部使用逻辑 `cuda:0`。显式 `--device cuda` 时没有CUDA会直接报错，不静默用CPU训练。常用参数还有 `--epochs`、`--num-workers`、`--batch-size-test`、`--data-path`、`--quiet`、`--output-dir`、`--record-file`、`--result-json`。不要同时启动重复的正式实验；不同数据集若并行运行，默认TXT通过文件锁串行追加完整记录。

检查点重评以保存的配置为准，不读取当前基础配置替换模型参数；允许设备、数据目录及评测 batch／workers变化，要求原始数据SHA256和训练源码一致。旧PyTorch不支持`weights_only`参数时兼容加载；仅重评本入口生成的可信检查点。没有断点续训、自动重试、自动早停或后台训练任务。

## 多次随机种子运行并取平均

用 `-d` 指定数据集、`-n` 指定独立训练次数。默认通过系统随机源在32位种子空间中无放回抽样，各次使用不同种子；模型参数、训练预算与公共协议保持相同。脚本启动独立Python子进程串行训练，一次完成后再开始下一次。以下5次只是使用示例，可改为所需次数：

```powershell
python -u MultiCBR/run_repeated.py -d Youshu -n 5 -g 0
python -u MultiCBR/run_repeated.py -d NetEase -n 5 -g 0
python -u MultiCBR/run_repeated.py -d iFashion -n 5 -g 0
```

批量入口默认要求CUDA，`-g 0`指定物理GPU 0。三个命令分别创建自己的批次，每个数据集内串行。运行中每30秒报告当前次数、种子、耗时及日志路径，详细训练／评测输出保存到各次`stdout.txt`。支持`--config`、`--epochs`、`--data-path`、`--num-workers`、`--batch-size-test`等显式覆盖，整批固定，不在各次之间调参。

随机种子在任何正式训练前一次性生成，并保存到 `MultiCBR/log/repeated/<dataset>/<批次ID>/manifest.json`，不是训练过程中按结果决定。该目录另保存冻结的`config.yaml`、每次独立执行及尝试的日志／结果／模型产物、`summary.json`和`summary.txt`。基础配置中的seed在此仅是占位值，每次实际训练均由已保存的随机种子覆盖；不修改`MultiCBR/config.yaml`。单次运行仍向`MultiCBR/实验记录.txt`追加完整记录。

全部指定次数有效成功后，对各次**验证集选中检查点**的完整精度验证／测试Recall与NDCG分别等权平均，并报告样本标准差（ddof=1）。每次全部TopK指标来自同一检查点，不将各指标的历史最高值拼接，也不选择测试最好的若干次。n=1时均值等于该次结果，样本标准差留空。任一次失败、中断或产物不完整，汇总标记未完成，正式均值与标准差均留空，避免把部分成功运行冒充N次结果。

同一命令再次执行会创建**新批次和新随机种子**。若要继续原批次，必须使用首次输出的批次路径：

```powershell
python -u MultiCBR/run_repeated.py --resume MultiCBR/log/repeated/Youshu/批次ID
```

恢复会复核并跳过完整成功的运行，未完成的运行从第1轮使用**原种子**重新训练，保留旧尝试，不新增或替换种子。恢复不允许改变次数、模型配置、数据、训练源码和GPU／设备。成功运行的结果与检查点缺失或被修改时停止，不能只凭SUCCESS标志汇总。强制结束控制器后，若旧训练PID可能仍在运行，会拒绝重复启动，需要先核查原主机；正常中断只停止本控制器启动的子进程。失败后不会自行无限重试。

可先保存随机计划再运行，或明确重放已保存的种子序列：

```powershell
# 检查输入并保存随机种子计划，不启动训练；之后用输出路径--resume执行
python -u MultiCBR/run_repeated.py -d Youshu -n 5 -g 0 --dry-run

# 明确预先指定各次种子，用于重放／跨模型采用同一序列；无需寻找最佳种子
python -u MultiCBR/run_repeated.py -d Youshu --seeds 17 53 2026 -g 0
```

`--dry-run`会新建批次计划及未完成汇总，不回写基础配置、正式结果或启动训练。显式种子数量须与`-n`一致（省略`-n`时由序列长度决定），不允许重复种子。后续论文应保留实际随机种子序列、次数、各次成绩、均值和标准差，不能只报告一次随机优胜值。

批量入口的必要检查：`python -B -m unittest MultiCBR.test_repeated -v`。仅覆盖种子计划、均值／标准差、两次小数据独立子进程训练与成功复用，不执行真实数据完整训练或重复全量回归。

## 产物与必要检查

每次训练写入 `MultiCBR/log/<dataset>/<UTC时间_UUID>/`：`training.log`、`tensorboard/`、`config.yaml`、`best.pt`、`best_config.json`、`result.json` 和 `sources/`。执行目录ID使用UTC，记录起止时间采用北京时间UTC+08:00、精确到秒，耗时使用单调时钟。

运行记录包括独立执行ID、状态、备注、完整实际配置、种子、预算及完成轮数、代码提交、训练源码快照及SHA256、配置摘要、六份数据摘要、Python／PyTorch／NumPy／SciPy／CUDA和GPU环境、产物路径及关键产物SHA256。best.pt保存选中轮次的模型、Adam状态、实际配置、来源摘要及完整指标。`best_epoch`从0计数，同时记录从1计数的训练轮次。`实验记录.txt`含易读指标及完整精度JSON，失败与正常处理的中断也追加记录；中断时已有best不算完整训练成功。强制杀进程或掉电无法保证完成finally记录。

```powershell
python -B -m unittest MultiCBR.test_reproduction -v
```

仅做必要检查：作者配置逐项一致；小图上作者与移植实现的损失、梯度、两次Adam更新及推理评分对照；公共指标／训练屏蔽／有效用户／严格选模；小数据训练、快照重放、追加记录及独立进程检查点重评。开发测试使用临时目录，正式实验记录不混入模拟结果。

本地是CPU版PyTorch。开发检查不代表CUDA验证、完整作者预算训练或取得论文复现成绩。正式结果须由服务器实跑产生；当前不改动ReCoBR、DCBR、DCDL、任何raw文件或工程记录，不代用户提交。
