import sys
import importlib

def check_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    try:
        importlib.import_module(import_name)
        print(f"[OK] {package_name} is installed.")
        return True
    except ImportError:
        print(f"[FAIL] {package_name} is NOT installed.")
        return False

def main():
    print("Checking environment for MultiCBR Project...")
    print("-" * 40)

    # Check Python version
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}")
    if sys.version_info >= (3, 7):
        print("[OK] Python version is >= 3.7")
    else:
        print("[WARNING] Python version might be too old (README recommends >= 3.7.11)")

    print("-" * 40)

    # Check PyTorch and CUDA
    try:
        import torch
        print(f"[OK] PyTorch is installed (Version: {torch.__version__})")
        
        if torch.cuda.is_available():
            print(f"[OK] CUDA is available.")
            print(f"     CUDA version: {torch.version.cuda}")
            print(f"     Device count: {torch.cuda.device_count()}")
            print(f"     Current device: {torch.cuda.get_device_name(0)}")
        else:
            print("[WARNING] CUDA is NOT available. Training will run on CPU, which might be slow.")
    except ImportError:
        print("[FAIL] PyTorch is NOT installed.")

    print("-" * 40)

    # Check other dependencies
    dependencies = {
        "numpy": "numpy",
        "scipy": "scipy",
        "tqdm": "tqdm",
        "pyyaml": "yaml",
        "tensorboard": "torch.utils.tensorboard"
    }

    all_deps_ok = True
    for package, import_name in dependencies.items():
        if not check_package(package, import_name):
            all_deps_ok = False

    print("-" * 40)
    if all_deps_ok:
        print("All dependencies checked. You seem to be ready to go!")
    else:
        print("Some dependencies are missing. Please install them using 'pip install -r requirements.txt'")

if __name__ == "__main__":
    main()
