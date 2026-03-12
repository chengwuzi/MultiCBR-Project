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



实验结果

baseline(0+0)
2026-03-12 12:59:29, Top_10, Test: recall: 0.058615, ndcg: 0.039228
2026-03-12 12:59:29, Top_20, Val:  recall: 0.086959, ndcg: 0.038399
2026-03-12 12:59:29, Top_20, Test: recall: 0.090363, ndcg: 0.049101
2026-03-12 12:59:29, Top_40, Val:  recall: 0.129508, ndcg: 0.047699
2026-03-12 12:59:29, Top_40, Test: recall: 0.135898, ndcg: 0.061065
2026-03-12 12:59:29, Top_80, Val:  recall: 0.184241, ndcg: 0.057796
2026-03-12 12:59:29, Top_80, Test: recall: 0.191687, ndcg: 0.073654


0.1 + 0.1 
2026-03-12 12:00:07, Top_10, Test: recall: 0.058387, ndcg: 0.038874
2026-03-12 12:00:07, Top_20, Val:  recall: 0.085304, ndcg: 0.037803
2026-03-12 12:00:07, Top_20, Test: recall: 0.092211, ndcg: 0.049358
2026-03-12 12:00:07, Top_40, Val:  recall: 0.128731, ndcg: 0.047308
2026-03-12 12:00:07, Top_40, Test: recall: 0.135617, ndcg: 0.060867
2026-03-12 12:00:07, Top_80, Val:  recall: 0.184726, ndcg: 0.057566
2026-03-12 12:00:07, Top_80, Test: recall: 0.193831, ndcg: 0.073879