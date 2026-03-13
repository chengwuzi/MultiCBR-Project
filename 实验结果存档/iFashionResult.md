[1/6] Testing ui_beta=0.0, bi_beta=0.0
Running: C:\Users\admin\anaconda3\envs\torch2.4_cuda11.8\python.exe -u train.py --dataset iFashion --ui_bundle_user_agg_beta 0.0 --bi_user_bundle_agg_beta 0.0 --epochs 40
load config file done!
>>>>>>>>>>B-I statistics>>>>>>>>>>
Average interactions 3.86061954498291
Non-zero rows 1.0
Non-zero columns 1.0
Matrix density 9.070365408454437e-05
>>>>>>>>>>U-I statistics>>>>>>>>>>
Average interactions 42.50041580200195
Non-zero rows 0.9989980889474368
Non-zero columns 0.9486643328712732
Matrix density 0.0009985296492948938
>>>>>>>>>>U-B statistics in train>>>>>>>>>>
Average interactions 22.5184326171875
Non-zero rows 1.0
Non-zero columns 1.0
Matrix density 0.0008131159568128954
>>>>>>>>>>U-B statistics in tune>>>>>>>>>>
Average interactions 2.764662265777588
Non-zero rows 1.0
Non-zero columns 0.8580558965840976
Matrix density 9.98289241748367e-05
>>>>>>>>>>U-B statistics in test>>>>>>>>>>
Average interactions 5.882052898406982
Non-zero rows 1.0
Non-zero columns 0.9736043908427818
Matrix density 0.00021239448271911793
{'data_path': './datasets', 'batch_size_train': 2048, 'batch_size_test': 2048, 'topk': [10, 20, 40, 80], 'neg_num': 1, 'aug_type': 'Noise', 'ed_interval': 1, 'embedding_sizes': [64], 'num_layerss': [2], 'ui_bundle_user_agg_beta': 0.0, 'bi_user_bundle_agg_beta': 0.0, 'UB_ratios': [0.05], 'UI_ratios': [0.0], 'BI_ratios': [0.15], 'fusion_weights': {'modal_weight': [0.1, 0.2, 0.7], 'UB_layer': [0.5, 0.5, 0.0], 'UI_layer': [0.3, 0.2, 0.5], 'BI_layer': [0.5, 0.5, 0.0]}, 'lrs': [0.001], 'l2_regs': [1e-05], 'c_lambdas': [0.1], 'c_temps': [0.2], 'epochs': 40, 'test_interval': 5, 'dataset': 'iFashion', 'model': 'MultiCBR', 'gpu': '0', 'info': '', 'num_users': 53897, 'num_bundles': 27694, 'num_items': 42563, 'device': device(type='cuda', index=0)}
C:\Users\admin\PycharmProjects\MultiCBR-Project\models\MultiCBR.py:37: UserWarning: torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated.  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=). (Triggered internally at C:\cb\pytorch_1000000000000\work\torch\csrc\utils\tensor_new.cpp:643.)
  graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))
2026-03-12 19:38:21, Top_10, Val:  recall: 0.103120, ndcg: 0.075638
2026-03-12 19:38:21, Top_10, Test: recall: 0.104703, ndcg: 0.102275
--------------------
2026-03-12 19:38:21, Top_20, Val:  recall: 0.146039, ndcg: 0.089826
2026-03-12 19:38:21, Top_20, Test: recall: 0.147629, ndcg: 0.119563
--------------------
2026-03-12 19:38:21, Top_40, Val:  recall: 0.199347, ndcg: 0.104261
2026-03-12 19:38:21, Top_40, Test: recall: 0.200452, ndcg: 0.138117
--------------------
2026-03-12 19:38:21, Top_80, Val:  recall: 0.262457, ndcg: 0.118583
2026-03-12 19:38:21, Top_80, Test: recall: 0.263428, ndcg: 0.156872
--------------------
top20 as the final evaluation standard
2026-03-12 19:38:21, Best in epoch 4, TOP 10: REC_V=0.10312, NDCG_V=0.07564
2026-03-12 19:38:21, Best in epoch 4, TOP 10: REC_T=0.10470, NDCG_T=0.10227
2026-03-12 19:38:21, Best in epoch 4, TOP 20: REC_V=0.14604, NDCG_V=0.08983
2026-03-12 19:38:21, Best in epoch 4, TOP 20: REC_T=0.14763, NDCG_T=0.11956
2026-03-12 19:38:21, Best in epoch 4, TOP 40: REC_V=0.19935, NDCG_V=0.10426
2026-03-12 19:38:21, Best in epoch 4, TOP 40: REC_T=0.20045, NDCG_T=0.13812
2026-03-12 19:38:21, Best in epoch 4, TOP 80: REC_V=0.26246, NDCG_V=0.11858
2026-03-12 19:38:21, Best in epoch 4, TOP 80: REC_T=0.26343, NDCG_T=0.15687
2026-03-12 19:45:04, Top_10, Val:  recall: 0.103403, ndcg: 0.075535
2026-03-12 19:45:04, Top_10, Test: recall: 0.105222, ndcg: 0.102294
--------------------
2026-03-12 19:45:04, Top_20, Val:  recall: 0.146419, ndcg: 0.089731
2026-03-12 19:45:04, Top_20, Test: recall: 0.148592, ndcg: 0.119746
--------------------
2026-03-12 19:45:04, Top_40, Val:  recall: 0.199319, ndcg: 0.104035
2026-03-12 19:45:04, Top_40, Test: recall: 0.202291, ndcg: 0.138578
--------------------
2026-03-12 19:45:04, Top_80, Val:  recall: 0.263990, ndcg: 0.118701
2026-03-12 19:45:04, Top_80, Test: recall: 0.265062, ndcg: 0.157278
--------------------
top20 as the final evaluation standard
2026-03-12 19:51:51, Top_10, Val:  recall: 0.104237, ndcg: 0.076056
2026-03-12 19:51:51, Top_10, Test: recall: 0.104957, ndcg: 0.102119
--------------------
2026-03-12 19:51:51, Top_20, Val:  recall: 0.146826, ndcg: 0.090122
2026-03-12 19:51:51, Top_20, Test: recall: 0.149223, ndcg: 0.119964
--------------------
2026-03-12 19:51:51, Top_40, Val:  recall: 0.200500, ndcg: 0.104633
2026-03-12 19:51:51, Top_40, Test: recall: 0.202644, ndcg: 0.138710
--------------------
2026-03-12 19:51:51, Top_80, Val:  recall: 0.264303, ndcg: 0.119121
2026-03-12 19:51:51, Top_80, Test: recall: 0.265837, ndcg: 0.157517
--------------------
top20 as the final evaluation standard
2026-03-12 19:51:51, Best in epoch 14, TOP 10: REC_V=0.10424, NDCG_V=0.07606
2026-03-12 19:51:51, Best in epoch 14, TOP 10: REC_T=0.10496, NDCG_T=0.10212
2026-03-12 19:51:51, Best in epoch 14, TOP 20: REC_V=0.14683, NDCG_V=0.09012
2026-03-12 19:51:51, Best in epoch 14, TOP 20: REC_T=0.14922, NDCG_T=0.11996
2026-03-12 19:51:51, Best in epoch 14, TOP 40: REC_V=0.20050, NDCG_V=0.10463
2026-03-12 19:51:51, Best in epoch 14, TOP 40: REC_T=0.20264, NDCG_T=0.13871
2026-03-12 19:51:51, Best in epoch 14, TOP 80: REC_V=0.26430, NDCG_V=0.11912
2026-03-12 19:51:51, Best in epoch 14, TOP 80: REC_T=0.26584, NDCG_T=0.15752
2026-03-12 19:58:34, Top_10, Val:  recall: 0.103814, ndcg: 0.075723
2026-03-12 19:58:34, Top_10, Test: recall: 0.105633, ndcg: 0.102696
--------------------
2026-03-12 19:58:34, Top_20, Val:  recall: 0.147234, ndcg: 0.090041
2026-03-12 19:58:34, Top_20, Test: recall: 0.148535, ndcg: 0.119972
--------------------
2026-03-12 19:58:34, Top_40, Val:  recall: 0.200644, ndcg: 0.104484
2026-03-12 19:58:34, Top_40, Test: recall: 0.201766, ndcg: 0.138658
--------------------
2026-03-12 19:58:34, Top_80, Val:  recall: 0.264564, ndcg: 0.118955
2026-03-12 19:58:34, Top_80, Test: recall: 0.264388, ndcg: 0.157299
--------------------
top20 as the final evaluation standard
2026-03-12 20:05:26, Top_10, Val:  recall: 0.104500, ndcg: 0.076073
2026-03-12 20:05:26, Top_10, Test: recall: 0.105798, ndcg: 0.102787
--------------------
2026-03-12 20:05:26, Top_20, Val:  recall: 0.148504, ndcg: 0.090624
2026-03-12 20:05:26, Top_20, Test: recall: 0.149066, ndcg: 0.120207
--------------------
2026-03-12 20:05:26, Top_40, Val:  recall: 0.202340, ndcg: 0.105200
2026-03-12 20:05:26, Top_40, Test: recall: 0.202799, ndcg: 0.139116
--------------------
2026-03-12 20:05:26, Top_80, Val:  recall: 0.265404, ndcg: 0.119506
2026-03-12 20:05:26, Top_80, Test: recall: 0.266900, ndcg: 0.158171
--------------------
top20 as the final evaluation standard
2026-03-12 20:05:26, Best in epoch 24, TOP 10: REC_V=0.10450, NDCG_V=0.07607
2026-03-12 20:05:26, Best in epoch 24, TOP 10: REC_T=0.10580, NDCG_T=0.10279
2026-03-12 20:05:26, Best in epoch 24, TOP 20: REC_V=0.14850, NDCG_V=0.09062
2026-03-12 20:05:26, Best in epoch 24, TOP 20: REC_T=0.14907, NDCG_T=0.12021
2026-03-12 20:05:26, Best in epoch 24, TOP 40: REC_V=0.20234, NDCG_V=0.10520
2026-03-12 20:05:26, Best in epoch 24, TOP 40: REC_T=0.20280, NDCG_T=0.13912
2026-03-12 20:05:26, Best in epoch 24, TOP 80: REC_V=0.26540, NDCG_V=0.11951
2026-03-12 20:05:26, Best in epoch 24, TOP 80: REC_T=0.26690, NDCG_T=0.15817
2026-03-12 20:12:19, Top_10, Val:  recall: 0.104404, ndcg: 0.075893
2026-03-12 20:12:19, Top_10, Test: recall: 0.105772, ndcg: 0.102603
--------------------
2026-03-12 20:12:19, Top_20, Val:  recall: 0.147865, ndcg: 0.090214
2026-03-12 20:12:19, Top_20, Test: recall: 0.150155, ndcg: 0.120495
--------------------
2026-03-12 20:12:19, Top_40, Val:  recall: 0.201721, ndcg: 0.104787
2026-03-12 20:12:19, Top_40, Test: recall: 0.203512, ndcg: 0.139235
--------------------
2026-03-12 20:12:19, Top_80, Val:  recall: 0.265499, ndcg: 0.119280
2026-03-12 20:12:19, Top_80, Test: recall: 0.266739, ndcg: 0.158051
--------------------
top20 as the final evaluation standard
2026-03-12 20:19:11, Top_10, Val:  recall: 0.103814, ndcg: 0.075762
2026-03-12 20:19:11, Top_10, Test: recall: 0.105900, ndcg: 0.102590
--------------------
2026-03-12 20:19:11, Top_20, Val:  recall: 0.147805, ndcg: 0.090279
2026-03-12 20:19:11, Top_20, Test: recall: 0.149273, ndcg: 0.120069
--------------------
2026-03-12 20:19:11, Top_40, Val:  recall: 0.201358, ndcg: 0.104799
2026-03-12 20:19:11, Top_40, Test: recall: 0.202568, ndcg: 0.138785
--------------------
2026-03-12 20:19:11, Top_80, Val:  recall: 0.265293, ndcg: 0.119302
2026-03-12 20:19:11, Top_80, Test: recall: 0.266248, ndcg: 0.157735
--------------------
top20 as the final evaluation standard
2026-03-12 20:26:02, Top_10, Val:  recall: 0.104073, ndcg: 0.075465
2026-03-12 20:26:02, Top_10, Test: recall: 0.105902, ndcg: 0.102696
--------------------
2026-03-12 20:26:02, Top_20, Val:  recall: 0.147787, ndcg: 0.089904
2026-03-12 20:26:02, Top_20, Test: recall: 0.149975, ndcg: 0.120453
--------------------
2026-03-12 20:26:02, Top_40, Val:  recall: 0.200786, ndcg: 0.104231
2026-03-12 20:26:02, Top_40, Test: recall: 0.203556, ndcg: 0.139269
--------------------
2026-03-12 20:26:02, Top_80, Val:  recall: 0.265148, ndcg: 0.118866
2026-03-12 20:26:02, Top_80, Test: recall: 0.265905, ndcg: 0.157821
--------------------
top20 as the final evaluation standard

[2/6] Testing ui_beta=0.0, bi_beta=0.1
Running: C:\Users\admin\anaconda3\envs\torch2.4_cuda11.8\python.exe -u train.py --dataset iFashion --ui_bundle_user_agg_beta 0.0 --bi_user_bundle_agg_beta 0.1 --epochs 40
load config file done!
>>>>>>>>>>B-I statistics>>>>>>>>>>
Average interactions 3.86061954498291
Non-zero rows 1.0
Non-zero columns 1.0
Matrix density 9.070365408454437e-05
>>>>>>>>>>U-I statistics>>>>>>>>>>
Average interactions 42.50041580200195
Non-zero rows 0.9989980889474368
Non-zero columns 0.9486643328712732
Matrix density 0.0009985296492948938
>>>>>>>>>>U-B statistics in train>>>>>>>>>>
Average interactions 22.5184326171875
Non-zero rows 1.0
Non-zero columns 1.0
Matrix density 0.0008131159568128954
>>>>>>>>>>U-B statistics in tune>>>>>>>>>>
Average interactions 2.764662265777588
Non-zero rows 1.0
Non-zero columns 0.8580558965840976
Matrix density 9.98289241748367e-05
>>>>>>>>>>U-B statistics in test>>>>>>>>>>
Average interactions 5.882052898406982
Non-zero rows 1.0
Non-zero columns 0.9736043908427818
Matrix density 0.00021239448271911793
{'data_path': './datasets', 'batch_size_train': 2048, 'batch_size_test': 2048, 'topk': [10, 20, 40, 80], 'neg_num': 1, 'aug_type': 'Noise', 'ed_interval': 1, 'embedding_sizes': [64], 'num_layerss': [2], 'ui_bundle_user_agg_beta': 0.0, 'bi_user_bundle_agg_beta': 0.1, 'UB_ratios': [0.05], 'UI_ratios': [0.0], 'BI_ratios': [0.15], 'fusion_weights': {'modal_weight': [0.1, 0.2, 0.7], 'UB_layer': [0.5, 0.5, 0.0], 'UI_layer': [0.3, 0.2, 0.5], 'BI_layer': [0.5, 0.5, 0.0]}, 'lrs': [0.001], 'l2_regs': [1e-05], 'c_lambdas': [0.1], 'c_temps': [0.2], 'epochs': 40, 'test_interval': 5, 'dataset': 'iFashion', 'model': 'MultiCBR', 'gpu': '0', 'info': '', 'num_users': 53897, 'num_bundles': 27694, 'num_items': 42563, 'device': device(type='cuda', index=0)}
C:\Users\admin\PycharmProjects\MultiCBR-Project\models\MultiCBR.py:37: UserWarning: torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated.  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=). (Triggered internally at C:\cb\pytorch_1000000000000\work\torch\csrc\utils\tensor_new.cpp:643.)
  graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))
2026-03-12 20:33:21, Top_10, Val:  recall: 0.102679, ndcg: 0.074275
2026-03-12 20:33:21, Top_10, Test: recall: 0.104447, ndcg: 0.100586
--------------------
2026-03-12 20:33:21, Top_20, Val:  recall: 0.146536, ndcg: 0.088749
2026-03-12 20:33:21, Top_20, Test: recall: 0.148776, ndcg: 0.118434
--------------------
2026-03-12 20:33:21, Top_40, Val:  recall: 0.201917, ndcg: 0.103739
2026-03-12 20:33:21, Top_40, Test: recall: 0.203020, ndcg: 0.137521
--------------------
2026-03-12 20:33:21, Top_80, Val:  recall: 0.266191, ndcg: 0.118338
2026-03-12 20:33:21, Top_80, Test: recall: 0.267532, ndcg: 0.156744
--------------------
top20 as the final evaluation standard
2026-03-12 20:33:21, Best in epoch 4, TOP 10: REC_V=0.10268, NDCG_V=0.07427
2026-03-12 20:33:21, Best in epoch 4, TOP 10: REC_T=0.10445, NDCG_T=0.10059
2026-03-12 20:33:21, Best in epoch 4, TOP 20: REC_V=0.14654, NDCG_V=0.08875
2026-03-12 20:33:21, Best in epoch 4, TOP 20: REC_T=0.14878, NDCG_T=0.11843
2026-03-12 20:33:21, Best in epoch 4, TOP 40: REC_V=0.20192, NDCG_V=0.10374
2026-03-12 20:33:21, Best in epoch 4, TOP 40: REC_T=0.20302, NDCG_T=0.13752
2026-03-12 20:33:21, Best in epoch 4, TOP 80: REC_V=0.26619, NDCG_V=0.11834
2026-03-12 20:33:21, Best in epoch 4, TOP 80: REC_T=0.26753, NDCG_T=0.15674
2026-03-12 20:40:14, Top_10, Val:  recall: 0.102693, ndcg: 0.073674
2026-03-12 20:40:14, Top_10, Test: recall: 0.104162, ndcg: 0.099916
--------------------
2026-03-12 20:40:14, Top_20, Val:  recall: 0.147602, ndcg: 0.088527
2026-03-12 20:40:14, Top_20, Test: recall: 0.149057, ndcg: 0.118066
--------------------
2026-03-12 20:40:14, Top_40, Val:  recall: 0.201732, ndcg: 0.103171
2026-03-12 20:40:14, Top_40, Test: recall: 0.203664, ndcg: 0.137235
--------------------
2026-03-12 20:40:14, Top_80, Val:  recall: 0.267230, ndcg: 0.118044
2026-03-12 20:40:14, Top_80, Test: recall: 0.268356, ndcg: 0.156501
--------------------
top20 as the final evaluation standard
2026-03-12 20:47:10, Top_10, Val:  recall: 0.102657, ndcg: 0.073682
2026-03-12 20:47:10, Top_10, Test: recall: 0.104231, ndcg: 0.099773
--------------------
2026-03-12 20:47:10, Top_20, Val:  recall: 0.147938, ndcg: 0.088647
2026-03-12 20:47:10, Top_20, Test: recall: 0.149265, ndcg: 0.117962
--------------------
2026-03-12 20:47:10, Top_40, Val:  recall: 0.203088, ndcg: 0.103596
2026-03-12 20:47:10, Top_40, Test: recall: 0.204415, ndcg: 0.137313
--------------------
2026-03-12 20:47:10, Top_80, Val:  recall: 0.268094, ndcg: 0.118372
2026-03-12 20:47:10, Top_80, Test: recall: 0.269400, ndcg: 0.156670
--------------------
top20 as the final evaluation standard
2026-03-12 20:54:03, Top_10, Val:  recall: 0.102807, ndcg: 0.073794
2026-03-12 20:54:03, Top_10, Test: recall: 0.104808, ndcg: 0.100472
--------------------
2026-03-12 20:54:03, Top_20, Val:  recall: 0.147156, ndcg: 0.088477
2026-03-12 20:54:03, Top_20, Test: recall: 0.149944, ndcg: 0.118672
--------------------
2026-03-12 20:54:03, Top_40, Val:  recall: 0.202717, ndcg: 0.103500
2026-03-12 20:54:03, Top_40, Test: recall: 0.204237, ndcg: 0.137734
--------------------
2026-03-12 20:54:03, Top_80, Val:  recall: 0.267800, ndcg: 0.118275
2026-03-12 20:54:03, Top_80, Test: recall: 0.269722, ndcg: 0.157208
--------------------
top20 as the final evaluation standard
2026-03-12 21:01:00, Top_10, Val:  recall: 0.103056, ndcg: 0.073729
2026-03-12 21:01:00, Top_10, Test: recall: 0.104233, ndcg: 0.100074
--------------------
2026-03-12 21:01:00, Top_20, Val:  recall: 0.147449, ndcg: 0.088412
2026-03-12 21:01:00, Top_20, Test: recall: 0.149345, ndcg: 0.118273
--------------------
2026-03-12 21:01:00, Top_40, Val:  recall: 0.202774, ndcg: 0.103390
2026-03-12 21:01:00, Top_40, Test: recall: 0.204223, ndcg: 0.137545
--------------------
2026-03-12 21:01:00, Top_80, Val:  recall: 0.267562, ndcg: 0.118101
2026-03-12 21:01:00, Top_80, Test: recall: 0.269186, ndcg: 0.156848
--------------------
top20 as the final evaluation standard
2026-03-12 21:07:52, Top_10, Val:  recall: 0.102827, ndcg: 0.073848
2026-03-12 21:07:52, Top_10, Test: recall: 0.104735, ndcg: 0.100175
--------------------
2026-03-12 21:07:52, Top_20, Val:  recall: 0.148005, ndcg: 0.088770
2026-03-12 21:07:52, Top_20, Test: recall: 0.149850, ndcg: 0.118393
--------------------
2026-03-12 21:07:52, Top_40, Val:  recall: 0.203361, ndcg: 0.103762
2026-03-12 21:07:52, Top_40, Test: recall: 0.204827, ndcg: 0.137698
--------------------
2026-03-12 21:07:52, Top_80, Val:  recall: 0.269015, ndcg: 0.118669
2026-03-12 21:07:52, Top_80, Test: recall: 0.269561, ndcg: 0.156956
--------------------
top20 as the final evaluation standard
2026-03-12 21:07:52, Best in epoch 29, TOP 10: REC_V=0.10283, NDCG_V=0.07385
2026-03-12 21:07:52, Best in epoch 29, TOP 10: REC_T=0.10473, NDCG_T=0.10018
2026-03-12 21:07:52, Best in epoch 29, TOP 20: REC_V=0.14800, NDCG_V=0.08877
2026-03-12 21:07:52, Best in epoch 29, TOP 20: REC_T=0.14985, NDCG_T=0.11839
2026-03-12 21:07:52, Best in epoch 29, TOP 40: REC_V=0.20336, NDCG_V=0.10376
2026-03-12 21:07:52, Best in epoch 29, TOP 40: REC_T=0.20483, NDCG_T=0.13770
2026-03-12 21:07:52, Best in epoch 29, TOP 80: REC_V=0.26902, NDCG_V=0.11867
2026-03-12 21:07:52, Best in epoch 29, TOP 80: REC_T=0.26956, NDCG_T=0.15696
2026-03-12 21:14:49, Top_10, Val:  recall: 0.103198, ndcg: 0.073846
2026-03-12 21:14:49, Top_10, Test: recall: 0.103978, ndcg: 0.099910
--------------------
2026-03-12 21:14:49, Top_20, Val:  recall: 0.147651, ndcg: 0.088519
2026-03-12 21:14:49, Top_20, Test: recall: 0.149705, ndcg: 0.118392
--------------------
2026-03-12 21:14:49, Top_40, Val:  recall: 0.202558, ndcg: 0.103379
2026-03-12 21:14:49, Top_40, Test: recall: 0.204998, ndcg: 0.137778
--------------------
2026-03-12 21:14:49, Top_80, Val:  recall: 0.268350, ndcg: 0.118320
2026-03-12 21:14:49, Top_80, Test: recall: 0.269649, ndcg: 0.157012
--------------------
top20 as the final evaluation standard
2026-03-12 21:21:46, Top_10, Val:  recall: 0.103419, ndcg: 0.074287
2026-03-12 21:21:46, Top_10, Test: recall: 0.104865, ndcg: 0.100377
--------------------
2026-03-12 21:21:46, Top_20, Val:  recall: 0.148016, ndcg: 0.089047
2026-03-12 21:21:46, Top_20, Test: recall: 0.149508, ndcg: 0.118414
--------------------
2026-03-12 21:21:46, Top_40, Val:  recall: 0.202971, ndcg: 0.103941
2026-03-12 21:21:46, Top_40, Test: recall: 0.204482, ndcg: 0.137693
--------------------
2026-03-12 21:21:46, Top_80, Val:  recall: 0.269254, ndcg: 0.118973
2026-03-12 21:21:46, Top_80, Test: recall: 0.269347, ndcg: 0.157022
--------------------
top20 as the final evaluation standard
2026-03-12 21:21:46, Best in epoch 39, TOP 10: REC_V=0.10342, NDCG_V=0.07429
2026-03-12 21:21:46, Best in epoch 39, TOP 10: REC_T=0.10487, NDCG_T=0.10038
2026-03-12 21:21:46, Best in epoch 39, TOP 20: REC_V=0.14802, NDCG_V=0.08905
2026-03-12 21:21:46, Best in epoch 39, TOP 20: REC_T=0.14951, NDCG_T=0.11841
2026-03-12 21:21:46, Best in epoch 39, TOP 40: REC_V=0.20297, NDCG_V=0.10394
2026-03-12 21:21:46, Best in epoch 39, TOP 40: REC_T=0.20448, NDCG_T=0.13769
2026-03-12 21:21:46, Best in epoch 39, TOP 80: REC_V=0.26925, NDCG_V=0.11897
2026-03-12 21:21:46, Best in epoch 39, TOP 80: REC_T=0.26935, NDCG_T=0.15702

[3/6] Testing ui_beta=0.0, bi_beta=0.2
Running: C:\Users\admin\anaconda3\envs\torch2.4_cuda11.8\python.exe -u train.py --dataset iFashion --ui_bundle_user_agg_beta 0.0 --bi_user_bundle_agg_beta 0.2 --epochs 40
load config file done!
>>>>>>>>>>B-I statistics>>>>>>>>>>
Average interactions 3.86061954498291
Non-zero rows 1.0
Non-zero columns 1.0
Matrix density 9.070365408454437e-05
>>>>>>>>>>U-I statistics>>>>>>>>>>
Average interactions 42.50041580200195
Non-zero rows 0.9989980889474368
Non-zero columns 0.9486643328712732
Matrix density 0.0009985296492948938
>>>>>>>>>>U-B statistics in train>>>>>>>>>>
Average interactions 22.5184326171875
Non-zero rows 1.0
Non-zero columns 1.0
Matrix density 0.0008131159568128954
>>>>>>>>>>U-B statistics in tune>>>>>>>>>>
Average interactions 2.764662265777588
Non-zero rows 1.0
Non-zero columns 0.8580558965840976
Matrix density 9.98289241748367e-05
>>>>>>>>>>U-B statistics in test>>>>>>>>>>
Average interactions 5.882052898406982
Non-zero rows 1.0
Non-zero columns 0.9736043908427818
Matrix density 0.00021239448271911793
{'data_path': './datasets', 'batch_size_train': 2048, 'batch_size_test': 2048, 'topk': [10, 20, 40, 80], 'neg_num': 1, 'aug_type': 'Noise', 'ed_interval': 1, 'embedding_sizes': [64], 'num_layerss': [2], 'ui_bundle_user_agg_beta': 0.0, 'bi_user_bundle_agg_beta': 0.2, 'UB_ratios': [0.05], 'UI_ratios': [0.0], 'BI_ratios': [0.15], 'fusion_weights': {'modal_weight': [0.1, 0.2, 0.7], 'UB_layer': [0.5, 0.5, 0.0], 'UI_layer': [0.3, 0.2, 0.5], 'BI_layer': [0.5, 0.5, 0.0]}, 'lrs': [0.001], 'l2_regs': [1e-05], 'c_lambdas': [0.1], 'c_temps': [0.2], 'epochs': 40, 'test_interval': 5, 'dataset': 'iFashion', 'model': 'MultiCBR', 'gpu': '0', 'info': '', 'num_users': 53897, 'num_bundles': 27694, 'num_items': 42563, 'device': device(type='cuda', index=0)}
C:\Users\admin\PycharmProjects\MultiCBR-Project\models\MultiCBR.py:37: UserWarning: torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated.  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=). (Triggered internally at C:\cb\pytorch_1000000000000\work\torch\csrc\utils\tensor_new.cpp:643.)
  graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))
2026-03-12 21:29:12, Top_10, Val:  recall: 0.101219, ndcg: 0.072124
2026-03-12 21:29:12, Top_10, Test: recall: 0.102312, ndcg: 0.097513
--------------------
2026-03-12 21:29:12, Top_20, Val:  recall: 0.146049, ndcg: 0.086915
2026-03-12 21:29:12, Top_20, Test: recall: 0.147566, ndcg: 0.115831
--------------------
2026-03-12 21:29:12, Top_40, Val:  recall: 0.201495, ndcg: 0.101910
2026-03-12 21:29:12, Top_40, Test: recall: 0.203753, ndcg: 0.135562
--------------------
2026-03-12 21:29:12, Top_80, Val:  recall: 0.268838, ndcg: 0.117213
2026-03-12 21:29:12, Top_80, Test: recall: 0.269545, ndcg: 0.155120
--------------------
top20 as the final evaluation standard
2026-03-12 21:29:12, Best in epoch 4, TOP 10: REC_V=0.10122, NDCG_V=0.07212
2026-03-12 21:29:12, Best in epoch 4, TOP 10: REC_T=0.10231, NDCG_T=0.09751
2026-03-12 21:29:12, Best in epoch 4, TOP 20: REC_V=0.14605, NDCG_V=0.08692
2026-03-12 21:29:12, Best in epoch 4, TOP 20: REC_T=0.14757, NDCG_T=0.11583
2026-03-12 21:29:12, Best in epoch 4, TOP 40: REC_V=0.20150, NDCG_V=0.10191
2026-03-12 21:29:12, Best in epoch 4, TOP 40: REC_T=0.20375, NDCG_T=0.13556
2026-03-12 21:29:12, Best in epoch 4, TOP 80: REC_V=0.26884, NDCG_V=0.11721
2026-03-12 21:29:12, Best in epoch 4, TOP 80: REC_T=0.26954, NDCG_T=0.15512
2026-03-12 21:36:06, Top_10, Val:  recall: 0.100312, ndcg: 0.071053
2026-03-12 21:36:06, Top_10, Test: recall: 0.101685, ndcg: 0.096237
--------------------
2026-03-12 21:36:06, Top_20, Val:  recall: 0.146058, ndcg: 0.086151
2026-03-12 21:36:06, Top_20, Test: recall: 0.147721, ndcg: 0.114880
--------------------
2026-03-12 21:36:06, Top_40, Val:  recall: 0.202798, ndcg: 0.101484
2026-03-12 21:36:06, Top_40, Test: recall: 0.204481, ndcg: 0.134814
--------------------
2026-03-12 21:36:06, Top_80, Val:  recall: 0.269035, ndcg: 0.116536
2026-03-12 21:36:06, Top_80, Test: recall: 0.271249, ndcg: 0.154684
--------------------
top20 as the final evaluation standard
2026-03-12 21:43:03, Top_10, Val:  recall: 0.100707, ndcg: 0.071390
2026-03-12 21:43:03, Top_10, Test: recall: 0.101142, ndcg: 0.096268
--------------------
2026-03-12 21:43:03, Top_20, Val:  recall: 0.146413, ndcg: 0.086494
2026-03-12 21:43:03, Top_20, Test: recall: 0.147915, ndcg: 0.115152
--------------------
2026-03-12 21:43:03, Top_40, Val:  recall: 0.203077, ndcg: 0.101814
2026-03-12 21:43:03, Top_40, Test: recall: 0.204751, ndcg: 0.135110
--------------------
2026-03-12 21:43:03, Top_80, Val:  recall: 0.270671, ndcg: 0.117186
2026-03-12 21:43:03, Top_80, Test: recall: 0.270915, ndcg: 0.154793


iFashion -- 实验结果


