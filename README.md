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

两模块结合  代码开发 + 实验  +  稳定种子开发 + 种子参数搜索完成 ， 因此确定 新分支 DECBR

DECBR模块一：BI视图 & UI视图 增强

DECBR模块二：对比学习改造，增加融合前两个子视图向UB视图看齐的对比loss





