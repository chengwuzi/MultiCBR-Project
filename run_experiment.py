import train

def run():
    # 这里是你的实验配置
    # 你可以在这里修改参数，直接右键运行此文件即可启动实验
    
    experiment_config = {
        "gpu": "0",
        "dataset": "Youshu",
        "model": "MultiCBR",
        "info": "my_custom_run", # 实验备注信息
        
        # --- 下面可以覆盖 config.yaml 中的参数 ---
        
        # 修改 epoch 数
        "epochs": 110,  # 默认跑 50 轮
        
        # --- Bundle Intent 模块配置 ---
        "use_bundle_intent": False,     # 是否启用 Bundle Intent 模块
        "n_bundle_intents": 16,        # 意图原型数量
        "intent_temp": 0.5,            # 意图 Softmax 温度
        "intent_alpha": 0.2,           # 意图特征融合权重 (Residual)
        "intent_lambda": 0.005,        # 意图对齐损失权重系数
        
        # 修改学习率
        # "lrs": [1e-3],
    }

    print("Starting experiment with config:", experiment_config)
    train.main(experiment_config)

if __name__ == "__main__":
    run()
