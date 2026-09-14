"""Recorded compiler inputs for experiments predating production adoption."""

from pathlib import Path

import pytest


@pytest.fixture
def pre_bfirst_step10(monkeypatch):
    pytest.importorskip("tvm")
    import gemm_kernels

    path = (Path(__file__).parents[1] / "results_b300/step10_roles.rx5lNL/step10/"
            "step10_4096_baseline/builder.py")
    namespace = dict(vars(gemm_kernels))
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    monkeypatch.setattr(gemm_kernels, "hgemm_v10", namespace["hgemm_v10"])
