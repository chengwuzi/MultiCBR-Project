
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


版本更新记录

DWT-1.0：Youshu & NetEase 的DWT扩散模型模块参数搜索工作已经完成，两个数据集DWT模块最佳参数已确定







