import train

def run():
    # 这里是你的实验配置
    # 你可以在这里修改参数，直接右键运行此文件即可启动实验
    
    experiment_config = {
        "gpu": "0",
        "dataset": "NetEase",
        "model": "MultiCBR",
        "info": "my_custom_run", # 实验备注信息
        
        # --- 下面可以覆盖 config.yaml 中的参数 ---

        # 固定随机种子
        # "seed": None,  # <---- 只需要加上这一行，比如填 42 或者 2024
        
        # 修改 epoch 数
        "epochs": 60,  # 测试用，设为 1 跑得快
        
        # 修改学习率
        # "lrs": [1e-3],
    }

    print("Starting experiment with config:", experiment_config)
    train.main(experiment_config)

if __name__ == "__main__":
    run()
