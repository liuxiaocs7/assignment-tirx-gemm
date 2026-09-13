"""Check diagnostic hooks preserve compiler inputs, outputs, and global state."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from benchmark_diagnostics import capture_compilation, capture_nvrtc, run_metadata


@pytest.mark.parametrize("compile_status", [0, 6])
def test_nvrtc_log_capture_preserves_call_and_restores_hook(tmp_path, compile_status):
    options = [b"--gpu-architecture=sm_103a", b"--ptxas-options=-v"]
    program = object()
    calls = []
    log = b"ptxas info : Used 128 registers, 0 spill stores\n\0"

    def compile_program(prog, count, opts):
        calls.append((prog, count, opts))
        return (compile_status,)

    def get_log(prog, buffer):
        assert prog is program
        buffer[:] = log
        return (0,)

    nvrtc = SimpleNamespace(
        nvrtcCompileProgram=compile_program,
        nvrtcGetProgramLogSize=lambda prog: (0, len(log)),
        nvrtcGetProgramLog=get_log,
        nvrtcResult=SimpleNamespace(NVRTC_SUCCESS=0),
    )
    with pytest.raises(RuntimeError, match="later compilation/launch failure"):
        with capture_nvrtc(nvrtc, tmp_path):
            assert nvrtc.nvrtcCompileProgram(program, 2, options) == (compile_status,)
            raise RuntimeError("later compilation/launch failure")
    assert nvrtc.nvrtcCompileProgram is compile_program
    assert calls == [(program, 2, options)]
    assert calls[0][2] is options
    assert json.loads((tmp_path / "nvrtc_01.options.json").read_text()) == [o.decode() for o in options]
    assert (tmp_path / "nvrtc_01.log").read_bytes() == log.rstrip(b"\0")


@pytest.mark.parametrize("fail", [False, True])
def test_tvm_callback_capture_preserves_binary_and_restores_on_exit(tmp_path, monkeypatch, fail):
    tvm_ffi = pytest.importorskip("tvm_ffi")
    pytest.importorskip("tvm.support.nvcc")
    name = "tvm_callback_cuda_compile"
    original = tvm_ffi.get_global_func(name)
    calls = []
    payload = bytearray(b"\x7fELF\x00\xff\x01")

    def compiler(source):
        calls.append(str(source))
        if fail:
            raise RuntimeError("compiler fixture failed")
        return payload

    monkeypatch.setenv("TVM_CUDA_COMPILE_MODE", "nvcc")
    tvm_ffi.register_global_func(name, compiler, override=True)
    try:
        if fail:
            with pytest.raises(Exception, match="compiler fixture failed"):
                with capture_compilation(tmp_path):
                    tvm_ffi.get_global_func(name)("// actual source")
            assert not (tmp_path / "module_01.fatbin").exists()
        else:
            with capture_compilation(tmp_path):
                assert bytes(tvm_ffi.get_global_func(name)("// actual source")) == payload
            assert (tmp_path / "module_01.fatbin").read_bytes() == payload
        assert calls == ["// actual source"]
        assert (tmp_path / "module_01.cu").read_text() == "// actual source"
        assert json.loads((tmp_path / "capture.json").read_text()) == {"compiler": "nvcc", "modules": 1}
        # A subsequent compile must no longer run through the capture wrapper.
        if fail:
            with pytest.raises(Exception, match="compiler fixture failed"):
                tvm_ffi.get_global_func(name)("// second source")
        else:
            assert bytes(tvm_ffi.get_global_func(name)("// second source")) == payload
        assert calls[-1] == "// second source"
        assert not (tmp_path / "module_02.cu").exists()
    finally:
        tvm_ffi.register_global_func(name, original, override=True)


def test_metadata_identifies_uncommitted_source(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "gemm_kernels.py"
    source.write_text("# original\n")
    subprocess.run(["git", "add", "gemm_kernels.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
                    "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)
    monkeypatch.setenv("TVM_CUDA_PTXAS_REG_LEVEL", "8")
    before = run_metadata(tmp_path)
    source.write_text("# modified\n")
    after = run_metadata(tmp_path)
    assert not before["git_dirty"] and after["git_dirty"]
    assert before["git_revision"] == after["git_revision"]
    assert before["gemm_kernels_sha256"] != after["gemm_kernels_sha256"]
    assert after["gemm_kernels_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert after["ptxas_reg_level"] == "8"


def test_metadata_works_without_git(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "")
    (tmp_path / "gemm_kernels.py").write_text("# archived source\n")
    metadata = run_metadata(tmp_path)
    assert metadata["git_revision"] == metadata["git_dirty"] == "unknown"
    assert len(metadata["gemm_kernels_sha256"]) == 64
