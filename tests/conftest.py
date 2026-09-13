import os
import shutil
import pytest


def pytest_configure(config):
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        import subprocess
        if shutil.which("nvidia-smi") is None:
            raise pytest.UsageError("GPU tests require a Blackwell GPU and nvidia-smi; see RUNNING.md")
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            lines = [line for line in result.stdout.splitlines() if line.strip()]
            if not lines:
                raise pytest.UsageError("nvidia-smi reported no GPUs")
            gpu_id = min(lines, key=lambda l: int(l.split(",")[1])).split(",")[0].strip()
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


@pytest.fixture(scope="session", autouse=True)
def configure_blackwell():
    import torch
    import gemm_kernels

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in {(10, 0), (10, 3)}:
        raise pytest.UsageError("GPU tests require SM100/SM103 (B200/B100/B300) and CUDA-enabled PyTorch")
    gemm_kernels.SM_COUNT = torch.cuda.get_device_properties(0).multi_processor_count


@pytest.fixture(autouse=True)
def seed_inputs():
    import torch

    seed = int(os.environ.get("GEMM_TEST_SEED", "0"))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
