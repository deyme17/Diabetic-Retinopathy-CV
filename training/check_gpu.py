import torch

def check_environment():
    print("=" * 60)
    print("ENVIREMENT INFORMATION")
    print("=" * 60)

    # PyTorch verion
    print(f"PyTorch version: {torch.__version__}")
    # CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA: {cuda_available}")

    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPU: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  name: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Memory: {props.total_memory / 1024 ** 3:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
    else:
        print("  GPU is not available. Use CPU.")

    # CPU info
    print(f"\nCPU threads: {torch.get_num_threads()}")

    return cuda_available


# start checking
cuda_available = check_environment()
device = torch.device("cuda" if cuda_available else "cpu")
print(f"\nCurrent device: {device}")