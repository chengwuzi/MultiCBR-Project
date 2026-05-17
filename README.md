
清空目录   &&   上传本地文件     密码：ssh2025

ssh -p 1771 ssh_user@210.45.212.126 "rm -rf /home/ssh_user/code/25-chenglong/MultiCBR/*"

scp -P 1771 -r * ssh_user@210.45.212.126:/home/ssh_user/code/25-chenglong/MultiCBR

登录服务器

ssh -p 1771 ssh_user@210.45.212.126

cd ~/code/25-chenglong/MultiCBR

tmux new -s multicbr

conda activate recbole

python -u train.py -d iFashion 2>&1 | tee logs_run.txt

python -u train.py -d Youshu 2>&1 | tee logs_run.txt

python -u train.py -d NetEase 2>&1 | tee logs_run.txt

tmux attach -t multicbr


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

工作计划

1.模块2消融实验探究工作继续

2.模型代码整理
进展：初步完成代码整理工作；后续：train.py,utils.py,model.py三文件待细致打磨

3.总体对比实验（两模块结合）
进展：初步完成，沿用当前各数据集最佳结果

4.两模块机制研究实验（需精心设计）

5.PPT画图准备

6.表格 & 公式 & 论文 准备







