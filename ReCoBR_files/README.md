# ReCoBR 实验与参数搜索

本目录复用根目录 `models/model.py` 的模型及 `train.py`、`utility.py` 的训练、重建和评测函数，新增独立的配置读取、实验调度与结果管理。模型计算不作改动。DCBR 和各模型的 `raw/` 不参与写入。

当前阶段固定开启潜空间扩散重建，关闭融合前 UB 锚定对比（`anchor_cl.enabled=false`、`anchor_cl_lambda=0`）；保留原有融合后对比学习。单次实验、后续搜参及配置检查均执行这些约束。未来开启融合前对比须先讨论并修改阶段约束。

## 文件和结果

- `config.yaml`：Meal-H、Meal-L 已讨论的工程初始配置，尚非优胜配置。
- `Meal-H/搜参配置.yaml`、`Meal-L/搜参配置.yaml`：各数据集的多轮计划，按轮次 ID 选择；每项有 ID、备注、覆盖配置，运行时回写状态、是否成功、执行 ID、实际配置和依赖选择信息。运行中不要手工编辑正在使用的计划；控制器会检测计划变化并停止覆盖。
- 各数据集目录中的 `搜参结果.txt`：按执行 ID 追加实验结果，包含北京时间（精确到秒）、耗时、配置、备注、重试、best epoch、验证/测试全部 topk 指标、指纹、日志和产物路径；还追加轮次和阶段汇总。
- 各数据集目录中的 `log/tuning/history.json`：机器可读的完整执行历史。每次执行另存 `execution.json`、实际输入配置、源码快照、各次尝试的 stdout 和结果 JSON。训练产物中有 `best.pt`、配置、TensorBoard 和训练日志。
- `train.py`：执行一个配置；列表字段必须只有一个值，禁止隐式展开成未登记的多次训练。
- `run_experiments.py`：独立子进程串行调度，不导入 torch，不预占 CUDA。
- `recobr_config.py`：共享的校验、递归配置覆盖、数据/预训练嵌入/源码摘要。

目前只登记 `meal_h_initial`、`meal_l_initial` 两个初始对照轮次，每轮一个实验；均为 150 轮、seed=2026。每个轮次在 `overrides` 中冻结完整初值，后续修改基础配置不会悄悄改写既有实验含义。尚未自行制定任何参数搜索网格。

## 运行

以下命令在项目根目录运行。相对配置路径均相对于项目根目录解析，不依赖服务器上的用户名或盘符。必须通过 `--plan` 明确指定数据集计划，避免选错文件。

```powershell
# 列出轮次；预览不修改计划、结果和历史，也不启动训练
python ReCoBR_files/run_experiments.py --plan ReCoBR_files/Meal-H/搜参配置.yaml --list
python ReCoBR_files/run_experiments.py --plan ReCoBR_files/Meal-L/搜参配置.yaml --list
python ReCoBR_files/run_experiments.py --plan ReCoBR_files/Meal-H/搜参配置.yaml --round meal_h_initial --dry-run
python ReCoBR_files/run_experiments.py --plan ReCoBR_files/Meal-L/搜参配置.yaml --round meal_l_initial --dry-run

# 正式运行指定轮次，物理 GPU 0 在独立子进程中映射为 cuda:0
python -u ReCoBR_files/run_experiments.py --plan ReCoBR_files/Meal-H/搜参配置.yaml --round meal_h_initial -g 0
python -u ReCoBR_files/run_experiments.py --plan ReCoBR_files/Meal-L/搜参配置.yaml --round meal_l_initial -g 0
```

每次命令只执行指定计划中的指定轮次。可在两个 IDE 窗口的控制台分别运行上面两条正式命令，同时执行 Meal-H、Meal-L；每个数据集内部仍串行运行实验并按依赖继承优胜配置。两个控制台可以共用同一份工程代码和 GPU 0，也可通过 `-g` 分别指定 GPU。

两套计划的 `work_dir`、`result_file` 和计划文件路径均不同，因此配置回写、history、执行锁、结果文档及训练产物相互独立。启动不同轮次也不能同时占用同一套计划或工作目录。模型、基础配置、数据及预训练嵌入只读共用，运行期间不要修改正在使用的公共文件。

默认 `resume` 跳过经复核的成功实验；失败或中断实验从第 1 个 epoch 重新训练，不从中途检查点接着优化。

```powershell
# 指定实验重跑：保留旧记录、创建新的执行 ID
python -u ReCoBR_files/run_experiments.py --plan ReCoBR_files/Meal-H/搜参配置.yaml --round meal_h_initial --mode selected --only initial -g 0

# 只重跑失败项；全轮重跑（历史引用对照始终不重训）
python -u ReCoBR_files/run_experiments.py --plan ReCoBR_files/Meal-H/搜参配置.yaml --round meal_h_initial --mode failed -g 0
python -u ReCoBR_files/run_experiments.py --plan ReCoBR_files/Meal-H/搜参配置.yaml --round meal_h_initial --mode all -g 0

# 单次配置检查和单次训练
python ReCoBR_files/train.py -d Meal-H --check-config
python -u ReCoBR_files/train.py -d Meal-H -g 0 --device cuda --note "初始配置对照"

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

先结合论文、模型机制及验证结果讨论参数和网格，明确固定条件及工程补充点，再将完整轮次追加到对应数据集的搜参计划。后续新增数据集也采用独立目录、计划、`work_dir` 和 `result_file`。**这里的示意结构不是已决定的搜参安排，也不要照抄示意值当正式实验。**

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
        rebuild_k: 3
    inherit_best:
      candidates: [initial]
      rule: validation_recall20_then_ndcg20
  - id: next_parameter
    note: 在完整继承前阶段优胜配置后仅改变当前参数
    overrides:
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

## 本地检查

```powershell
python -B -m unittest discover -s ReCoBR_files -p test_search.py -v
```

测试仅使用临时小型 CPU 数据，涵盖原训练数值一致性、重建图恢复、真实子进程连续继承、重试、跳过、重跑、失效检测与阶段约束。不会在真实 Meal-H/L 上启动正式训练，也不会写工程记录。
