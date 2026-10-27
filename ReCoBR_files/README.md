# ReCoBR 实验与参数搜索

本目录复用根目录 `models/model.py` 的模型及 `train.py`、`utility.py` 的训练、重建和评测函数，新增独立的配置读取、实验调度与结果管理。模型计算不作改动。DCBR 和各模型的 `raw/` 不参与写入。

已完成的第一轮先关闭潜空间扩散重建，运行初始对照和 `c_lambda` 搜索；固定优胜 `c_lambda` 后开启扩散搜索 K，此后保持开启。Meal-H 新增第二轮，以下单列说明。全程关闭融合前 UB 锚定对比（`anchor_cl.enabled=false`、`anchor_cl_lambda=0`），保留融合后对比学习。模块状态由实际配置和各阶段覆盖项明确保存，未来开启融合前对比须先讨论并修改阶段约束。

## 文件和结果

- `config.yaml`：Meal-H、Meal-L 各自首轮验证优胜配置，扩散开启；已登记轮次在计划 `overrides` 中冻结完整初值，历史含义不随基础配置更新而改变。
- `Meal-H/搜参配置.yaml`、`Meal-L/搜参配置.yaml`：各数据集的多轮计划，按轮次 ID 选择；每项有 ID、备注、覆盖配置，运行时回写状态、是否成功、执行 ID、实际配置和依赖选择信息。运行中不要手工编辑正在使用的计划；控制器会检测计划变化并停止覆盖。
- 各数据集目录中的 `搜参结果.txt`：按执行 ID 追加实验结果，包含北京时间（精确到秒）、耗时、配置、备注、重试、best epoch、验证/测试全部 topk 指标、指纹、日志和产物路径；还追加轮次和阶段汇总。
- 各数据集目录中的 `log/tuning/history.json`：机器可读的完整执行历史。每次执行另存 `execution.json`、实际输入配置、源码快照、各次尝试的 stdout 和结果 JSON。训练产物中有 `best.pt`、配置、TensorBoard 和训练日志。
- `train.py`：执行一个配置；列表字段必须只有一个值，禁止隐式展开成未登记的多次训练。
- `run_experiments.py`：独立子进程串行调度，不导入 torch，不预占 CUDA。
- `recobr_config.py`：共享的校验、递归配置覆盖、数据/预训练嵌入/源码摘要。

第一轮计划为 `meal_h_round_01`、`meal_l_round_01`，每轮 26 个实验，均为 150 轮、seed=2026。每个轮次在 `overrides` 中冻结完整初值，后续修改基础配置不会悄悄改写既有实验含义。

| 阶段 | 新实验数 | 扩散 | 选优与继承 |
|---|---:|---|---|
| 初始对照 | 1 | 关闭 | 当前初始 `c_lambda`、原融合比例 |
| `c_lambda` | 6 | 关闭 | 初始对照及六个候选共同选优 |
| K | 6 | 开启 | 完整继承前阶段优胜 `c_lambda`；六个 K 全部重新训练 |
| 融合粗搜 | 7 | 开启 | 固定优胜 `c_lambda`、K，七组比例与 K 阶段原比例共同选优 |
| 融合细搜 | 6 | 开启 | 围绕同一个粗搜优胜，六种方向各转移 0.05；此前对照继续参与选优 |

Meal-H 的 `c_lambda` 初值为 0.05，新增 `[0.01,0.02,0.04,0.10,0.20,0.40]`，K 为 `[1,3,7,13,19,35]`。Meal-L 的初值为 0.04，新增 `[0.005,0.01,0.02,0.05,0.10,0.20]`，K 为 `[1,3,5,7,11,19]`。0.005 及 K=35 是工程补充点，其他值为论文敏感性分析范围内的取值；K 网格是按数据特点选取的粗搜子集。

融合权重顺序为 `[UB,UI,BI]`。七组粗搜比例为均衡 `[1/3,1/3,1/3]`、三组单视图偏重 `[0.6,0.2,0.2]` 的排列、三组双视图偏重 `[0.4,0.4,0.2]` 的排列。这套粗搜及步长 0.05 的细搜属于工程设计。

细搜用 `fusion_transfer: {from: UB, to: UI, amount: 0.05}` 等六种有向转移定义；待前置候选全部成功后，脚本选择同一个粗搜优胜，完整继承再调整权重，并在依赖记录中保存原比例、转移方向和实际比例。六个细搜实验互不继承，不因前一个细搜结果改变搜索中心。

`stages` 分别报告各阶段优胜；`final_candidates` 只包含 K、融合粗搜和融合细搜的 19 个开启扩散配置。无扩散结果保留作对照，不会被汇总为这轮最终优胜，即使其验证指标更高。无扩散对照中的 K 字段没有实际参与重建，不能作为开启扩散的 K 对照复用。

## 运行

Meal-H 第二轮使用同一个数据集计划，按新轮次 ID 运行；具体候选与继承安排见后文“Meal-H 第二轮”。

以下命令在项目根目录运行。相对配置路径均相对于项目根目录解析，不依赖服务器上的用户名或盘符。必须通过 `--plan` 明确指定数据集计划，避免选错文件。

```powershell
# 列出轮次；预览不修改计划、结果和历史，也不启动训练
python ReCoBR_files/run_experiments.py --plan ReCoBR_files/Meal-H/搜参配置.yaml --list
python ReCoBR_files/run_experiments.py --plan ReCoBR_files/Meal-L/搜参配置.yaml --list
python ReCoBR_files/run_experiments.py --plan ReCoBR_files/Meal-H/搜参配置.yaml --round meal_h_round_01 --dry-run
python ReCoBR_files/run_experiments.py --plan ReCoBR_files/Meal-L/搜参配置.yaml --round meal_l_round_01 --dry-run

# 正式运行指定轮次，物理 GPU 0 在独立子进程中映射为 cuda:0
python -u ReCoBR_files/run_experiments.py --plan ReCoBR_files/Meal-H/搜参配置.yaml --round meal_h_round_01 -g 0
python -u ReCoBR_files/run_experiments.py --plan ReCoBR_files/Meal-L/搜参配置.yaml --round meal_l_round_01 -g 0
```

每次命令执行指定计划中的完整轮次：首轮各 26 个实验，Meal-H 第二轮为 34 个新实验及 1 个历史引用。可在两个 IDE 窗口的控制台分别运行不同数据集；每个数据集内部仍串行运行实验并按依赖继承优胜配置。两个控制台可以共用同一份工程代码和 GPU 0，也可通过 `-g` 分别指定 GPU。

两套计划的 `work_dir`、`result_file` 和计划文件路径均不同，因此配置回写、history、执行锁、结果文档及训练产物相互独立。启动不同轮次也不能同时占用同一套计划或工作目录。模型、基础配置、数据及预训练嵌入只读共用，运行期间不要修改正在使用的公共文件。

默认 `resume` 跳过经复核的成功实验；失败或中断实验从第 1 个 epoch 重新训练，不从中途检查点接着优化。

```powershell
# 指定实验重跑：保留旧记录、创建新的执行 ID
python -u ReCoBR_files/run_experiments.py --plan ReCoBR_files/Meal-H/搜参配置.yaml --round meal_h_round_01 --mode selected --only initial -g 0

# 只重跑失败项；全轮重跑（历史引用对照始终不重训）
python -u ReCoBR_files/run_experiments.py --plan ReCoBR_files/Meal-H/搜参配置.yaml --round meal_h_round_01 --mode failed -g 0
python -u ReCoBR_files/run_experiments.py --plan ReCoBR_files/Meal-H/搜参配置.yaml --round meal_h_round_01 --mode all -g 0

# 单次配置检查和单次训练
python ReCoBR_files/train.py -d Meal-H --check-config
python -u ReCoBR_files/train.py -d Meal-H -g 0 --device cuda --note "当前基础配置"

# 对保存的 best.pt 重新评测（用相应产物目录中的 config.yaml）
python ReCoBR_files/train.py -d Meal-H --config "实际产物目录/config.yaml" --evaluate "实际产物目录/best.pt" -g 0
```

重新评测会另建输出目录，检查数据、嵌入、训练源码和方法配置，并恢复 best epoch 当时的重建 UB 图。配置快照包含运行字段，评测入口会按模型配置字段读取。

## 已确定的协议

每次训练内部：只有本次验证 Recall@20 和 NDCG@20 **同时严格高于同一个已保存检查点**时才更新 best。按配置中的 `test_interval` 评测，未到评测间隔的最后一轮不额外选模。

不同配置之间：使用未舍入的验证 Recall@20，完全相同再比较验证 NDCG@20；两者完全相同保留计划声明顺序在前者。测试指标仅报告，不能参与自动选参。结果中 `best_epoch` 从 0 计数，`best_epoch_1based` 是实际训练轮次。

固定每轮 seed 和预算，单实验不能覆盖 seed、epochs、数据、嵌入输入及公共评测设置。默认数据加载 workers=0，独立训练进程启动前设 `PYTHONHASHSEED` 和 GPU 环境，训练时复用根目录 `set_seed`。固定种子不承诺不同硬件、驱动或 PyTorch 版本逐位一致。

执行前后校验模型、训练辅助函数、入口、控制器、数据和两份预训练嵌入摘要。成功标志还必须通过训练预算、完整指标、实际配置、执行身份和检查点文件摘要复核；损坏或不匹配的产物不会作为成功对照继续使用。计划和工作历史均有互斥锁；中断只终止本控制器持有的训练子进程。

## 后续添加一轮搜索

先按用户当次约定、模型机制及验证结果讨论参数和网格，明确固定条件及候选依据，再将完整轮次追加到对应数据集的搜参计划。旧论文图表不能替代当前新数据集的搜索设计；Meal-H 第二轮明确自主设计网格。后续新增数据集也采用独立目录、计划、`work_dir` 和 `result_file`。**这里的示意结构不是已决定的搜参安排，也不要照抄示意值当正式实验。**

```yaml
id: reviewed_round_id
dataset: Meal-H
note: 经讨论确认的参数、范围及依据
selection_rule: validation_recall20_then_ndcg20
overrides: {}  # 正式计划应在此冻结完整初始配置及固定 seed/预算
experiments:
  - id: initial
    note: 初始对照
    overrides: {}
  - id: k_candidate
    note: 经讨论确认的 K 候选
    overrides:
      latent_rebuild:
        enabled: true
        rebuild_k: 3
    inherit_best:
      candidates: [initial]
      rule: validation_recall20_then_ndcg20
  - id: next_parameter
    note: 在完整继承前阶段优胜配置后仅改变当前参数
    overrides:
      latent_rebuild:
        enabled: true
      c_temps: [0.3]
    inherit_best:
      candidates: [initial, k_candidate]
      rule: validation_recall20_then_ndcg20
stages:
  - id: k_stage
    candidates: [initial, k_candidate]
```

嵌套覆盖采用递归合并，修改 K 不会清掉扩散模块的其他字段。`inherit_best` 必须引用本轮前面声明的实验；所有候选都有效成功后才选优，完整继承获胜配置，再覆盖当前参数。对照应加入每个阶段候选，防止强行采用更差的新值；后续阶段也应保留此前累计优胜对照。前置失败或结果无效会阻塞依赖项。

跨轮复用使用显式 `reference` 固定 `round_id`、`experiment_id`、`execution_id`、`fingerprint`，绝不仅凭相同 ID 或手工 `success=true` 信任结果。默认核验 history 及产物；确有迁移需求时可显式指定历史 `result_file` 引用完整 TXT 执行块，来源及未复核旧产物的事实会保留在依赖记录中。优先保留完整历史和产物。首次产生 ReCoBR 历史后，再按实际执行 ID 编写引用项。

TXT 引用自动兼容 LF/CRLF：当前文本转换后的 SHA256 必须与历史摘要精确一致，真实源码或数据内容变化仍拒绝；转换证据保留两端摘要。预训练 `.pt` 按二进制摘要严格核对。显式 `allow_data_path_relocation=true` 时，数据及嵌入摘要核对通过后允许数据目录、两份嵌入路径迁移，其他配置必须一致；不会改写原始历史配置和指标。

## Meal-H 第二轮

轮次 `meal_h_round_02`，34 个新实验、1 个 TXT 历史对照，共35个候选。历史对照固定首轮 `fusion_fine_ub_to_ui`，执行 ID `20261026T172912_81c8dcda`，验证 Recall@20=0.3785844258140611、NDCG@20=0.24390730448890638；不重训、不要求旧 history 或检查点仍位于原路径。仅验证@20使用未舍入选模值，其他历史指标沿用文档显示精度。

以当前首轮优胜为完整起点：`c_lambda=0.1`、K=13、融合比例 `[0.05,0.35,0.60]`。固定150轮、seed=2026、扩散开启、融合前对比关闭、两个额外聚合系数为0、SetLoss权重0.05。每阶段新增4—6个实验，全部前置候选有效成功后，完整继承累计验证优胜，再仅覆盖当前参数；旧对照始终参与，平局保留较早者。阶段内各候选共用同一个前阶段优胜，不逐个追逐同阶段临时优胜。

| 顺序 | 参数 | 新增候选 | 数量 |
|---|---|---|---:|
| 1 | 推荐模型 `l2_reg` | `1e-6,5e-6,5e-5,1e-4` | 4 |
| 2 | 扩散噪声上限 | `0.002,0.008,0.05,0.10` | 4 |
| 3 | 三视图融合 `[UB,UI,BI]` | `[.025,.375,.60]`、`[0,.40,.60]`、`[.025,.325,.65]`、`[0,.35,.65]`、`[.025,.30,.675]`、`[0,.30,.70]` | 6 |
| 4 | `BI_layer` | `[.7,.3,0]`、`[.3,.7,0]`、`[.3,.4,.3]`、`[.1,.3,.6]` | 4 |
| 5 | `BI_ratio` | `0,0.1,0.3,0.5` | 4 |
| 6 | `UI_layer` | `[.6,.3,.1]`、`[.2,.6,.2]`、`[.2,.3,.5]`、`[1/3,1/3,1/3]` | 4 |
| 7 | `UI_ratio` | `0.025,0.05,0.1,0.2` | 4 |
| 8 | 融合后对比温度 `c_temp` | `0.05,0.1,0.3,0.5` | 4 |

这些网格按当前实现、Meal-H首轮趋势及参数作用自主设计，属于工程补充方案，不受旧论文实验或敏感性图表网格约束。L2检查弱/强约束；层融合分别覆盖初始、一阶、二阶偏重；BI扰动覆盖无、弱、强扰动，UI从零对照开始检查正扰动；温度覆盖较尖锐和较平滑的分布。噪声上限固定12步及起点1e-4，覆盖弱至较强的累计扰动，候选均合法。

噪声上限安排在融合前：此时UB权重固定0.05，扩散仍有作用路径。若融合随后选中UB=0，当前关闭融合前对比且两个额外聚合系数为0，重建UB图不会影响最终推荐表示，必须将它如实解释为边界对照，不能把该结果归因为扩散收益；后续没有再安排失去作用的扩散参数搜索。模块开关仍按计划保持开启。

```powershell
# 预览单个首阶段候选，只作配置检查，不启动训练、不改计划和结果
python -B ReCoBR_files/run_experiments.py --plan ReCoBR_files/Meal-H/搜参配置.yaml --round meal_h_round_02 --mode selected --only l2_1e6 --dry-run

# 正式执行完整第二轮，可按服务器实际情况修改物理GPU号
python -u ReCoBR_files/run_experiments.py --plan ReCoBR_files/Meal-H/搜参配置.yaml --round meal_h_round_02 -g 0
```

计划、结果、history 和产物继续使用 Meal-H 独立目录；首轮已有内容保留，Meal-L计划不变。基础配置不由搜索自动覆盖，下一次写回优胜仍按用户指令执行。关键检查：`python -B -m unittest discover -s ReCoBR_files -p test_meal_h_round2.py -v`，覆盖真实TXT引用、候选与协议、完整继承、对照保留及迁移校验，不启动正式训练。

## 本地检查

```powershell
python -B -m unittest discover -s ReCoBR_files -p test_search.py -v
```

日常修改只需运行相关的关键检查。本轮可用 `python -B -m unittest discover -s ReCoBR_files -p test_round_plan.py -v` 核对计划、阶段继承、动态细搜及无扩散训练；测试只使用临时小型 CPU 数据或合成指标，不启动真实 Meal-H/L 正式训练，也不写工程记录。
