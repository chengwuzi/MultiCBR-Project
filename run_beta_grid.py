import itertools
import subprocess
import sys
import datetime

# --- 配置部分 ---
DATASET = "NetEase"
# 网格参数定义：包含基线 (0.0, 0.0)
UI_BETAS = [0.0, 0.05, 0.1, 0.2]
BI_BETAS = [0.0, 0.05, 0.1, 0.2]
OUTPUT_FILE = "grid_search_results.txt"
TAIL_LINES = None # 保存最后 N 行输出，设为 None 则保存全部

def run_experiment(ui_beta, bi_beta):
    cmd = [
        sys.executable, "train.py",
        "--dataset", DATASET,
        "--ui_bundle_user_agg_beta", str(ui_beta),
        "--bi_user_bundle_agg_beta", str(bi_beta)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        # 使用 subprocess.Popen 实时捕获输出并显示
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # 行缓冲
        )

        stdout_lines = []
        stderr_lines = []

        # 实时读取 stdout 并打印
        # 注意：这里会阻塞直到子进程结束，或者 stdout 关闭
        # 为了更完美的实时显示，通常需要多线程或 select，但简单起见，我们优先处理 stdout
        # 另一种简单方法是让 stdout 直接继承父进程，但那样我们就捕获不到内容用于写入文件了
        # 下面采用一种折中方案：逐行读取并 print，同时保存到 list
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line, end='') # 实时打印到控制台
                stdout_lines.append(line)
        
        # 读取剩余的 stderr
        stderr_content = process.stderr.read()
        if stderr_content:
            print(stderr_content, end='', file=sys.stderr)
            stderr_lines = stderr_content.splitlines(keepends=True)

        returncode = process.poll()
        
        # 构造一个类似 subprocess.CompletedProcess 的对象返回
        class CompletedProcess:
            def __init__(self, args, returncode, stdout, stderr):
                self.args = args
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr
        
        return CompletedProcess(cmd, returncode, "".join(stdout_lines), "".join(stderr_lines))

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
        if (ui_beta == 0.0 and bi_beta == 0.0) or (ui_beta == 0.1 and bi_beta == 0.1):
            print(f"\n[{i+1}/{len(combinations)}] Skipping ui_beta={ui_beta}, bi_beta={bi_beta} (Already run)")
            continue

        print(f"\n[{i+1}/{len(combinations)}] Testing ui_beta={ui_beta}, bi_beta={bi_beta}")
        
        result = run_experiment(ui_beta, bi_beta)
        save_result(OUTPUT_FILE, ui_beta, bi_beta, result)
        
        if result and result.returncode != 0:
            print(f"Warning: Experiment failed for ui_beta={ui_beta}, bi_beta={bi_beta}")

    print(f"\nAll experiments finished. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
