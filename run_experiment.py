import train

def run():
    # 这里是你的实验配置
    # 你可以在这里修改参数，直接右键运行此文件即可启动实验
    
    experiment_config = {
        "gpu": "0",
        "dataset": "Youshu",  # 切换回 Youshu 数据集
        "model": "MultiCBR",
        "info": "ub_diff_k5", # 实验备注信息，建议根据模式修改，例如 original, ub_pop_k5, ub_diff_k5
        
        # --- 下面可以覆盖 config.yaml 中的参数 ---
        
        # --- BI Purifier (当前测试 UB，先把它关了) ---
        "bi_purifier": {
            "enabled": False
        },

        # --- UB Purifier 配置 ---
        
        # 1. 原始 MultiCBR 模式 (关闭 UB purifier)
        # "ub_purifier": {
        #     "enabled": False
        # },

        # 2. Popularity 对照模式 (开启 UB purifier, mode 为 popularity)
        # "ub_purifier": {
        #     "enabled": True,
        #     "mode": "popularity",
        #     "keep_k": 5,              # 针对 Youshu，建议先用固定 keep_k=5 测试
        #     "keep_ratio": None,
        #     "min_keep": 1,
        #     "cache_dir": "./ub_purifier_cache",
        #     "reuse_cached_graph": True,
        #     "save_purified_npz": True
        # },

        # 3. Diffusion 提纯模式 (开启 UB purifier, mode 为 diffusion)
        "ub_purifier": {
            "enabled": True,
            "mode": "diffusion",
            "keep_k": 5,              # 针对 Youshu，建议先用固定 keep_k=5 测试 (Youshu 用户平均只有 6 个 bundle)
            "keep_ratio": None,
            "min_keep": 1,
            "epochs": 20,             # UB purifier 训练轮数
            "batch_size": 256,
            "cache_dir": "./ub_purifier_cache",
            "reuse_cached_graph": False, # 如果想每次强制重新训练 purifier，设为 False
            "save_purified_npz": False
        },
        
        # 测试用，设为 1 跑得快；正式实验请注释掉或改回正常 epoch (Youshu 默认是 100)
        # "epochs": 1,  
    }

    print("Starting experiment with config:", experiment_config)
    train.main(experiment_config)

if __name__ == "__main__":
    run()
