# MultiCBR
Pytorch implementation for "MultiCBR: Multi-view Contrastive Learning for Bundle Recommendation"

### Environment

- OS: Ubuntu 18.04 or higher version
- python == 3.7.11 or above
- supported(tested) CUDA versions: 10.2
- Pytorch == 1.9.0 or above

### Run the code
To train MultiCBR on dataset NetEase with GPU 0, simply run:

    python train.py -g 0 -m MultiCBR -d NetEase
You can indicate GPU id or dataset with cmd line arguments, and the hyper-parameters are recorded in config.yaml. 





分支说明

MultiCBR-Raw: MultiCBR 的原始版本，未加任何改动；

MultiCBR-Base: 经过我处理的版本，加了一些便携性改动（环境配置，便携启动等），没有加框架优化的改动；后续CBR模型的改造均以此为基础



MultiCBR改成去除验证集后的实验结果

Top-3 Avg, TOP 10: REC_T=0.21116, NDCG_T=0.15720
Top-3 Avg, TOP 20: REC_T=0.29961, NDCG_T=0.18161
Top-3 Avg, TOP 40: REC_T=0.40549, NDCG_T=0.21028
Top-3 Avg, TOP 80: REC_T=0.52184, NDCG_T=0.23703