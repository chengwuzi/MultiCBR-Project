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

0 + 0.05
2026-03-12 14:13:46, Best in epoch 29, TOP 10: REC_V=0.05357, NDCG_V=0.02958
2026-03-12 14:13:46, Best in epoch 29, TOP 10: REC_T=0.05751, NDCG_T=0.03855
2026-03-12 14:13:46, Best in epoch 29, TOP 20: REC_V=0.08681, NDCG_V=0.03841
2026-03-12 14:13:46, Best in epoch 29, TOP 20: REC_T=0.08915, NDCG_T=0.04853
2026-03-12 14:13:46, Best in epoch 29, TOP 40: REC_V=0.12921, NDCG_V=0.04770
2026-03-12 14:13:46, Best in epoch 29, TOP 40: REC_T=0.13314, NDCG_T=0.06017
2026-03-12 14:13:46, Best in epoch 29, TOP 80: REC_V=0.18510, NDCG_V=0.05797
2026-03-12 14:13:46, Best in epoch 29, TOP 80: REC_T=0.18823, NDCG_T=0.07258

0 + 0.1
2026-03-12 14:39:20, Best in epoch 29, TOP 10: REC_V=0.05596, NDCG_V=0.03070
2026-03-12 14:39:20, Best in epoch 29, TOP 10: REC_T=0.05869, NDCG_T=0.03968
2026-03-12 14:39:20, Best in epoch 29, TOP 20: REC_V=0.08683, NDCG_V=0.03900
2026-03-12 14:39:20, Best in epoch 29, TOP 20: REC_T=0.09148, NDCG_T=0.04984
2026-03-12 14:39:20, Best in epoch 29, TOP 40: REC_V=0.13164, NDCG_V=0.04876
2026-03-12 14:39:20, Best in epoch 29, TOP 40: REC_T=0.13488, NDCG_T=0.06137
2026-03-12 14:39:20, Best in epoch 29, TOP 80: REC_V=0.18747, NDCG_V=0.05899
2026-03-12 14:39:20, Best in epoch 29, TOP 80: REC_T=0.19201, NDCG_T=0.07412

0 + 0.2
2026-03-12 15:08:08, Best in epoch 34, TOP 10: REC_V=0.05538, NDCG_V=0.03029
2026-03-12 15:08:08, Best in epoch 34, TOP 10: REC_T=0.05894, NDCG_T=0.03944
2026-03-12 15:08:08, Best in epoch 34, TOP 20: REC_V=0.08874, NDCG_V=0.03923
2026-03-12 15:08:08, Best in epoch 34, TOP 20: REC_T=0.09318, NDCG_T=0.05012
2026-03-12 15:08:08, Best in epoch 34, TOP 40: REC_V=0.13379, NDCG_V=0.04898
2026-03-12 15:08:08, Best in epoch 34, TOP 40: REC_T=0.13739, NDCG_T=0.06177
2026-03-12 15:08:08, Best in epoch 34, TOP 80: REC_V=0.18935, NDCG_V=0.05924
2026-03-12 15:08:08, Best in epoch 34, TOP 80: REC_T=0.19585, NDCG_T=0.07491

0.05 + 0
2026-03-12 15:33:29, Top_20, Val:  recall: 0.084913, ndcg: 0.038305
2026-03-12 15:33:29, Top_20, Test: recall: 0.090111, ndcg: 0.048804
--------------------
2026-03-12 15:33:29, Top_40, Val:  recall: 0.128173, ndcg: 0.047726
2026-03-12 15:33:29, Top_40, Test: recall: 0.134630, ndcg: 0.060552
--------------------
2026-03-12 15:33:29, Top_80, Val:  recall: 0.185203, ndcg: 0.058228
2026-03-12 15:33:29, Top_80, Test: recall: 0.188961, ndcg: 0.072761

0.05 + 0.05
2026-03-12 15:58:36, Top_20, Val:  recall: 0.086453, ndcg: 0.038290
2026-03-12 15:58:36, Top_20, Test: recall: 0.089941, ndcg: 0.049270
--------------------
2026-03-12 15:58:36, Top_40, Val:  recall: 0.129533, ndcg: 0.047698
2026-03-12 15:58:36, Top_40, Test: recall: 0.133416, ndcg: 0.060715
--------------------
2026-03-12 15:58:36, Top_80, Val:  recall: 0.186131, ndcg: 0.058153
2026-03-12 15:58:36, Top_80, Test: recall: 0.190854, ndcg: 0.073608

0.05 + 0.1
2026-03-12 16:24:21, Top_20, Val:  recall: 0.086279, ndcg: 0.038414
2026-03-12 16:24:21, Top_20, Test: recall: 0.091289, ndcg: 0.049401
--------------------
2026-03-12 16:24:21, Top_40, Val:  recall: 0.129871, ndcg: 0.047936
2026-03-12 16:24:21, Top_40, Test: recall: 0.136332, ndcg: 0.061316
--------------------
2026-03-12 16:24:21, Top_80, Val:  recall: 0.186467, ndcg: 0.058345
2026-03-12 16:24:21, Top_80, Test: recall: 0.192701, ndcg: 0.073961

0 + 0.3 
2026-03-12 17:18:30, Top_10, Val:  recall: 0.054219, ndcg: 0.029168
2026-03-12 17:18:30, Top_10, Test: recall: 0.059017, ndcg: 0.039391
--------------------
2026-03-12 17:18:30, Top_20, Val:  recall: 0.087110, ndcg: 0.037948
2026-03-12 17:18:30, Top_20, Test: recall: 0.092482, ndcg: 0.049796
--------------------
2026-03-12 17:18:30, Top_40, Val:  recall: 0.132921, ndcg: 0.047937
2026-03-12 17:18:30, Top_40, Test: recall: 0.140001, ndcg: 0.062252
--------------------
2026-03-12 17:18:30, Top_80, Val:  recall: 0.189902, ndcg: 0.058444
2026-03-12 17:18:30, Top_80, Test: recall: 0.197797, ndcg: 0.075201


0 + 0.4 
2026-03-12 17:44:19, Top_10, Val:  recall: 0.055452, ndcg: 0.029698
2026-03-12 17:44:19, Top_10, Test: recall: 0.057637, ndcg: 0.037962
--------------------
2026-03-12 17:44:19, Top_20, Val:  recall: 0.088866, ndcg: 0.038609
2026-03-12 17:44:19, Top_20, Test: recall: 0.092042, ndcg: 0.048644
--------------------
2026-03-12 17:44:19, Top_40, Val:  recall: 0.131936, ndcg: 0.048002
2026-03-12 17:44:19, Top_40, Test: recall: 0.138050, ndcg: 0.060793
--------------------
2026-03-12 17:44:19, Top_80, Val:  recall: 0.191936, ndcg: 0.058994
2026-03-12 17:44:19, Top_80, Test: recall: 0.197590, ndcg: 0.074073
