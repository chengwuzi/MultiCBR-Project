import train

def run():
    # 这里是你的实验配置
    # 你可以在这里修改参数，直接右键运行此文件即可启动实验
    
    experiment_config = {
        "gpu": "0",
        "dataset": "Youshu",  # 切换到 Youshu 数据集
        "model": "MultiCBR",
        "info": "diff_ratio05", # 实验备注信息，建议根据模式修改，例如 original, pop_ratio05, diff_ratio05
        
        # --- 下面可以覆盖 config.yaml 中的参数 ---
        
        # 1. 原始 MultiCBR 模式 (关闭 purifier)
        # "bi_purifier": {
        #     "enabled": False
        # },

        # 2. Popularity 对照模式 (开启 purifier, mode 为 popularity)
        # "bi_purifier": {
        #     "enabled": True,
        #     "mode": "popularity",
        #     "keep_ratio": 0.5,
        #     "min_keep": 5,
        #     "cache_dir": "./bi_purifier_cache",
        #     "reuse_cached_graph": True,
        #     "save_purified_npz": True
        # },

        # 3. Diffusion 提纯模式 (开启 purifier, mode 为 diffusion)
        "bi_purifier": {
            "enabled": True,
            "mode": "diffusion",
            "keep_ratio": 0.5,
            "min_keep": 5,
            "epochs": 20,              # purifier 训练轮数
            "batch_size": 256,
            "cache_dir": "./bi_purifier_cache",
            "reuse_cached_graph": False, # 第二次跑会直接读缓存，不再重复训练
            "save_purified_npz": False
        },
        
        # 测试用，设为 1 跑得快；正式实验请注释掉或改回正常 epoch (Youshu 默认是 100)
        # "epochs": 1,  
    }

    print("Starting experiment with config:", experiment_config)
    train.main(experiment_config)

if __name__ == "__main__":
    run()
