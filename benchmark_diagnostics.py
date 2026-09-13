"""Record the code and actual compiler output used by a benchmark run.

The optional hooks only surround compilation and the first launch. They delegate
to TVM's existing compiler without changing options and are removed before timing.
"""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def run_metadata(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout
        dirty = bool(status)
    except (OSError, subprocess.SubprocessError):
        revision, dirty = "unknown", "unknown"
    metadata = {"git_revision": revision, "git_dirty": dirty}
    for name in ("gemm_kernels", "utils", "benchmark", "benchmark_diagnostics"):
        path = root / f"{name}.py"
        metadata[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "unknown"
    metadata.update(
        compiler=os.environ.get("TVM_CUDA_COMPILE_MODE", "nvrtc").lower(),
        ptxas_reg_level=os.environ.get("TVM_CUDA_PTXAS_REG_LEVEL", "10"),
    )
    # Only record relevant flags, not the entire process environment.
    for key in ("TVM_CUDA_PTXAS_EXTRA_OPTS", "TVM_CUDA_NVRTC_EXTRA_OPTS",
                "TVM_CUDA_NVCC_NO_FAST_MATH", "TVM_KERNEL_DEBUG", "TVM_KERNEL_DUMP",
                "CUDA_LAUNCH_BLOCKING", "CUDA_VISIBLE_DEVICES"):
        metadata[key] = os.environ.get(key, "")
    return metadata


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def dump_binary_resources(binary_path):
    """Read resources from the actual binary when CUDA's cuobjdump is available.

    Some NVRTC versions return frontend warnings without ptxas resource statistics,
    even with -v. Disassembly is outside timing and does not recompile the kernel.
    """
    binary_path = Path(binary_path)
    output_path = binary_path.with_suffix(".resources.txt")
    executable = shutil.which("cuobjdump")
    if executable is None:
        candidate = Path(os.environ.get("CUDA_PATH", "/usr/local/cuda")) / "bin/cuobjdump"
        if candidate.is_file():
            executable = str(candidate)
    if executable is None:
        output_path.write_text("Resource report unavailable: cuobjdump was not found.\n"
                               "Run cuobjdump --dump-resource-usage on the saved binary.\n")
        return
    try:
        result = subprocess.run([executable, "--dump-resource-usage", str(binary_path)],
                                capture_output=True, text=True, timeout=30)
        output_path.write_text(f"cuobjdump exit code: {result.returncode}\n" + result.stdout + result.stderr)
    except (OSError, subprocess.SubprocessError) as error:
        # Resource reporting is optional; retain the saved binary and timing run.
        output_path.write_text(f"Resource report unavailable: {error}\n")


@contextmanager
def capture_nvrtc(nvrtc, directory):
    """Read NVRTC's log; some versions omit ptxas resource statistics."""
    original = nvrtc.nvrtcCompileProgram
    count = 0

    def compile_program(prog, num_options, options):
        nonlocal count
        count += 1
        prefix = directory / f"nvrtc_{count:02d}"
        write_json(prefix.with_suffix(".options.json"), [
            opt.decode() if isinstance(opt, bytes) else str(opt) for opt in options
        ])
        result = original(prog, num_options, options)
        status, size = nvrtc.nvrtcGetProgramLogSize(prog)
        if status == nvrtc.nvrtcResult.NVRTC_SUCCESS and size > 0:
            log = bytearray(size)
            (status,) = nvrtc.nvrtcGetProgramLog(prog, log)
            message = (log.decode("utf-8", errors="replace").rstrip("\0")
                       if status == nvrtc.nvrtcResult.NVRTC_SUCCESS else f"Cannot read NVRTC log: {status}\n")
        else:
            message = f"NVRTC log size={size}, status={status}\n"
        prefix.with_suffix(".log").write_text(message)
        return result

    nvrtc.nvrtcCompileProgram = compile_program
    try:
        yield
    finally:
        nvrtc.nvrtcCompileProgram = original


@contextmanager
def capture_compilation(directory):
    if directory is None:
        yield
        return

    import tvm_ffi
    from tvm.support import nvcc  # Registers TVM's compiler callback if needed.

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    original = tvm_ffi.get_global_func("tvm_callback_cuda_compile")
    compiler = os.environ.get("TVM_CUDA_COMPILE_MODE", "nvrtc").lower()
    count = 0

    def compile_cuda(code):
        nonlocal count
        count += 1
        prefix = directory / f"module_{count:02d}"
        prefix.with_suffix(".cu").write_text(str(code))
        binary = original(code)
        suffix = ".cubin" if compiler == "nvrtc" else ".fatbin"
        binary_path = prefix.with_suffix(suffix)
        binary_path.write_bytes(bytes(binary))
        dump_binary_resources(binary_path)
        return binary

    tvm_ffi.register_global_func("tvm_callback_cuda_compile", compile_cuda, override=True)
    try:
        if compiler == "nvrtc":
            from cuda.bindings import nvrtc

            result, major, minor = nvrtc.nvrtcVersion()
            write_json(directory / "nvrtc_version.json", {
                "status": str(result), "major": major, "minor": minor,
            })
            with capture_nvrtc(nvrtc, directory):
                yield
        else:
            yield
    finally:
        tvm_ffi.register_global_func("tvm_callback_cuda_compile", original, override=True)
        write_json(directory / "capture.json", {"compiler": compiler, "modules": count})
