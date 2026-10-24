# ReCoBR

ReCoBR 是一个多视图组合推荐模型，联合建模用户–组合（UB）、用户–物品（UI）和组合–物品（BI）关系，结合跨视图残差聚合、融合前 UB 锚定对比学习，以及可选的潜空间扩散交互筛选。

模型实现在 `models/model.py` 的 `ReCoBR` 类中，训练入口为 `train.py`，各数据集的参数配置见 `config.yaml`。`--model` 默认值为 `ReCoBR`，也可通过 `-m ReCoBR` 显式指定。

训练产生的日志、TensorBoard 数据和模型检查点分别存放在 `log/<数据集>/ReCoBR/`、`runs/<数据集>/ReCoBR/` 和 `checkpoints/<数据集>/ReCoBR/` 下。

以下部署命令中的 `MultiCBR` 为现有服务器目录名。

清空目录   &&   上传本地文件     密码：ssh2025

ssh -p 1771 ssh_user@210.45.212.126 "rm -rf /home/ssh_user/code/25-chenglong/MultiCBR/*"

scp -P 1771 -r * ssh_user@210.45.212.126:/home/ssh_user/code/25-chenglong/MultiCBR

登录服务器

ssh -p 1771 ssh_user@210.45.212.126

cd ~/code/25-chenglong/MultiCBR

tmux new -s recobr

conda activate recbole

python -u train.py -d iFashion -m ReCoBR 2>&1 | tee logs_run.txt

python -u train.py -d Youshu -m ReCoBR 2>&1 | tee logs_run.txt

python -u train.py -d NetEase -m ReCoBR 2>&1 | tee logs_run.txt

tmux attach -t recobr


————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

环境依赖配置
torch>=1.9.0
numpy
scipy
tqdm
pyyaml
tensorboard

————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————








