import itertools
import subprocess
import sys
import datetime

# --- 配置部分 ---
DATASET = "Youshu"
# 网格参数定义：包含基线 (0.0, 0.0)
UI_BETAS = [0.0]
BI_BETAS = [0.00, 0.00, 0.00, 0.05, 0.05, 0.05]
OUTPUT_FILE = "grid_search_results.txt"
TAIL_LINES = None # 保存最后 N 行输出，设为 None 则保存全部
EPOCHS = 120 # 在这里配置 epoch 数量

def run_experiment(ui_beta, bi_beta):
    cmd = [
        sys.executable, "-u", "train.py", # -u: unbuffered binary stdout/stderr
        "--dataset", DATASET,
        "--ui_bundle_user_agg_beta", str(ui_beta),
        "--bi_user_bundle_agg_beta", str(bi_beta),
        "--epochs", str(EPOCHS)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        # 使用 subprocess.Popen 实时捕获输出
        # stderr=subprocess.STDOUT: 将 stderr 合并到 stdout，这样 tqdm 进度条也能被捕获
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1,  # 行缓冲
            encoding='utf-8', 
            errors='replace'
        )

        stdout_lines = []
        
        while True:
            # read(1) 逐字符读取，确保能捕获 \r 刷新
            char = process.stdout.read(1)
            if not char and process.poll() is not None:
                break
            if char:
                # 实时写入控制台（保留回车符，实现进度条原地刷新）
                sys.stdout.write(char)
                sys.stdout.flush()
                # 累积输出用于保存，但不在此处做任何过滤
                # 这样文件里会保存完整的历史，而控制台视觉上是正常的进度条
                stdout_lines.append(char)
        
        returncode = process.poll()
        full_output = "".join(stdout_lines)
        
        # 构造一个类似 subprocess.CompletedProcess 的对象返回
        class CompletedProcess:
            def __init__(self, args, returncode, stdout, stderr):
                self.args = args
                self.returncode = returncode
                # 过滤掉进度条刷新过程中产生的重复行，只保留最终状态或关键日志
                # tqdm 的进度条通常包含 \r，如果不处理直接写入文件会很乱
                # 这里简单处理：保留所有行，但文件查看器通常能处理回车符
                # 如果觉得文件太大，可以在这里做进一步清洗，例如：
                # self.stdout = re.sub(r'.*\r', '', stdout) 
                self.stdout = stdout
                self.stderr = stderr
        
        # 因为 stderr 已经合并到 stdout，所以 stderr 为空字符串
        return CompletedProcess(cmd, returncode, full_output, "")

    except Exception as e:
        print(f"Error running command: {e}")
        return None

def save_result(file_path, ui_beta, bi_beta, result):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "SUCCESS" if result and result.returncode == 0 else "FAILED"
    
    with open(file_path, "a", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"dataset: {DATASET}\n")
        f.write(f"ui_bundle_user_agg_beta: {ui_beta}\n")
        f.write(f"bi_user_bundle_agg_beta: {bi_beta}\n")
        f.write(f"epochs: {EPOCHS}\n")
        f.write(f"status: {status}\n")
        
        command_str = ' '.join(result.args) if result else 'N/A'
        f.write(f"command: {command_str}\n")
        f.write("-" * 50 + "\n")
        
        if result:
            stdout_lines = result.stdout.splitlines()
            
            if TAIL_LINES is not None and len(stdout_lines) > TAIL_LINES:
                f.write(f"[STDOUT (Last {TAIL_LINES} lines)]\n")
                stdout_content = "\n".join(stdout_lines[-TAIL_LINES:])
            else:
                f.write("[STDOUT]\n")
                stdout_content = result.stdout
                
            f.write(stdout_content + "\n\n")
            
            f.write("[STDERR]\n")
            f.write(result.stderr + "\n")
        else:
             f.write("[ERROR]\nFailed to execute command.\n")
             
        f.write("=" * 50 + "\n\n")

def main():
    # 每次运行前清空结果文件，避免混淆（可选）
    # open(OUTPUT_FILE, 'w').close() 
    
    combinations = list(itertools.product(UI_BETAS, BI_BETAS))
    print(f"Total combinations to run: {len(combinations)}")
    
    for i, (ui_beta, bi_beta) in enumerate(combinations):
        # 跳过已运行的组合 (0.0, 0.0) 和 (0.1, 0.1)
        # if (ui_beta == 0.0 and bi_beta == 0.0) or (ui_beta == 0.1 and bi_beta == 0.1):
        #    print(f"\n[{i+1}/{len(combinations)}] Skipping ui_beta={ui_beta}, bi_beta={bi_beta} (Already run)")
        #    continue

        print(f"\n[{i+1}/{len(combinations)}] Testing ui_beta={ui_beta}, bi_beta={bi_beta}")
        
        result = run_experiment(ui_beta, bi_beta)
        save_result(OUTPUT_FILE, ui_beta, bi_beta, result)
        
        if result and result.returncode != 0:
            print(f"Warning: Experiment failed for ui_beta={ui_beta}, bi_beta={bi_beta}")

    print(f"\nAll experiments finished. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
