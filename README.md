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



实验结果记录


user_mode + bundle_mode

static + static
2026-03-12 07:45:22, Best in epoch 29, TOP 10: REC_V=0.05374, NDCG_V=0.02973
2026-03-12 07:45:22, Best in epoch 29, TOP 10: REC_T=0.05913, NDCG_T=0.03950
2026-03-12 07:45:22, Best in epoch 29, TOP 20: REC_V=0.08576, NDCG_V=0.03829
2026-03-12 07:45:22, Best in epoch 29, TOP 20: REC_T=0.09045, NDCG_T=0.04918
2026-03-12 07:45:22, Best in epoch 29, TOP 40: REC_V=0.12759, NDCG_V=0.04742
2026-03-12 07:45:22, Best in epoch 29, TOP 40: REC_T=0.13457, NDCG_T=0.06086
2026-03-12 07:45:22, Best in epoch 29, TOP 80: REC_V=0.18355, NDCG_V=0.05769
2026-03-12 07:45:22, Best in epoch 29, TOP 80: REC_T=0.18769, NDCG_T=0.07283


adaptive + static
2026-03-12 09:12:23, Top_10, Test: recall: 0.055665, ndcg: 0.037627
2026-03-12 09:12:23, Top_20, Val:  recall: 0.086267, ndcg: 0.038012
2026-03-12 09:12:23, Top_20, Test: recall: 0.089292, ndcg: 0.048203
2026-03-12 09:12:23, Top_40, Val:  recall: 0.130138, ndcg: 0.047630
2026-03-12 09:12:23, Top_40, Test: recall: 0.134515, ndcg: 0.060265
2026-03-12 09:12:23, Top_80, Val:  recall: 0.189502, ndcg: 0.058591
2026-03-12 09:12:23, Top_80, Test: recall: 0.196270, ndcg: 0.074169


static + adaptive
2026-03-12 08:46:26, Best in epoch 29, TOP 10: REC_V=0.05116, NDCG_V=0.02759
2026-03-12 08:46:26, Best in epoch 29, TOP 10: REC_T=0.05158, NDCG_T=0.03467
2026-03-12 08:46:26, Best in epoch 29, TOP 20: REC_V=0.08060, NDCG_V=0.03545
2026-03-12 08:46:26, Best in epoch 29, TOP 20: REC_T=0.08164, NDCG_T=0.04404
2026-03-12 08:46:26, Best in epoch 29, TOP 40: REC_V=0.12222, NDCG_V=0.04453
2026-03-12 08:46:26, Best in epoch 29, TOP 40: REC_T=0.12572, NDCG_T=0.05565
2026-03-12 08:46:26, Best in epoch 29, TOP 80: REC_V=0.17550, NDCG_V=0.05440
2026-03-12 08:46:26, Best in epoch 29, TOP 80: REC_T=0.18066, NDCG_T=0.06804


adaptive + adaptive
效果很差，比baseline差很多，我就没等他跑完了

