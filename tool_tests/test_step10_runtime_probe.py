"""Exercise diagnostic counting, graph verification and failure retention on CPU.

GPU APIs are substituted; these tests do not establish B300 capture support or
performance. The ordinary grading paths are covered by their existing tests.
"""

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("runtime_probe", ROOT / "probe_step10_runtime.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.fixture
def fake_gpu():
    state = SimpleNamespace(capture=None, writes=0, checks=0, drop_capture=False,
                            tvm_stream=False, warmups=0)
    output = SimpleNamespace(value=42)
    output.fill_ = lambda value: setattr(output, "value", value)

    def call():
        if state.capture is not None:
            assert state.tvm_stream
            if not state.drop_capture:
                state.capture.nodes.append(lambda: setattr(output, "value", 42))
        else:
            state.writes += 1
            output.value = 42

    class Graph:
        def __init__(self):
            self.nodes = []
            self.replays = 0

        def replay(self):
            self.replays += 1
            for node in self.nodes:
                node()

    @contextmanager
    def capture(graph, stream):
        state.capture = graph
        try:
            yield
        finally:
            state.capture = None

    @contextmanager
    def stream_context(stream):
        yield

    @contextmanager
    def ffi_stream(context):
        with context:
            state.tvm_stream = True
            try:
                yield
            finally:
                state.tvm_stream = False

    def check(actual, reference, **kwargs):
        assert kwargs == dict(rtol=1e-3, atol=1e-2)
        assert actual.value == reference
        state.checks += 1

    torch = SimpleNamespace(
        cuda=SimpleNamespace(Stream=object, synchronize=lambda: None, CUDAGraph=Graph,
                             graph=capture, stream=stream_context),
        testing=SimpleNamespace(assert_close=check),
    )
    return torch, SimpleNamespace(use_torch_stream=ffi_stream), call, output, state


def test_graph_has_thirty_calls_and_replay_recomputes_poisoned_output(fake_gpu):
    torch, ffi, call, output, state = fake_gpu
    graph = probe.capture_graph(torch, ffi, call, output, 42)
    assert len(graph.nodes) == 30 and graph.replays == 2
    assert state.writes == 10 and state.checks == 2
    assert not state.tvm_stream


def test_empty_graph_cannot_pass_using_stale_correct_output(fake_gpu):
    torch, ffi, call, output, state = fake_gpu
    state.drop_capture = True
    with pytest.raises(AssertionError):
        probe.capture_graph(torch, ffi, call, output, 42)


def test_graph_and_eager_time_thirty_kernels_with_ten_kernel_warmups(fake_gpu):
    torch, ffi, call, output, state = fake_gpu
    graph = probe.capture_graph(torch, ffi, call, output, 42)
    settings = []

    def timer(fn, warmup, repeat):
        settings.append((warmup, repeat))
        for _ in range(warmup + repeat):
            fn()
        return .15 if repeat == 30 else 3.0

    writes, replays = state.writes, graph.replays
    assert probe.measure("eager", call, graph, torch, timer) == .15
    assert state.writes - writes == 40
    writes = state.writes
    assert probe.measure("graph", call, graph, torch, timer) == .1
    assert state.writes - writes == 10 and graph.replays - replays == 1
    assert settings == [(10, 30), (0, 1)]


def test_balanced_comparison_keeps_all_slow_eager_samples(fake_gpu, tmp_path):
    torch, ffi, call, output, state = fake_gpu
    graph = probe.capture_graph(torch, ffi, call, output, 42)

    def timer(fn, warmup, repeat):
        for _ in range(warmup + repeat):
            fn()
        return .14 if repeat == 30 else 3.0

    rows = probe.paired_trials(torch, call, graph, output, 42, timer, tmp_path, "before_load", .1391)
    assert [r["mode"] for r in rows] == ["eager", "graph", "graph", "eager"] * 4
    assert all(r["verified"] for r in rows)
    assert json.loads((tmp_path / "before_load.json").read_text()) == rows
    stats = probe.summary(rows, .1391)
    assert stats["eager_within_limit"] == 0
    assert stats["eager_median_ms"] == .14 and stats["graph_median_ms"] == .1
    assert stats["paired_eager_over_graph"] == pytest.approx(1.4)


def test_numerical_failure_saves_unverified_sample_and_stops(fake_gpu, tmp_path):
    torch, ffi, call, output, state = fake_gpu
    graph = probe.capture_graph(torch, ffi, call, output, 42)

    def timer(fn, warmup, repeat):
        output.value = -1
        return .14

    with pytest.raises(AssertionError):
        probe.paired_trials(torch, call, graph, output, 42, timer, tmp_path, "before_load", .1391)
    row, = json.loads((tmp_path / "before_load.json").read_text())
    assert row["ms"] == .14 and row["verified"] is False


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_timings_are_not_saved_as_valid_samples(fake_gpu, tmp_path, value):
    torch, ffi, call, output, state = fake_gpu
    graph = probe.capture_graph(torch, ffi, call, output, 42)
    with pytest.raises(ValueError, match="Invalid"):
        probe.paired_trials(torch, call, graph, output, 42, lambda *a, **k: value,
                            tmp_path, "before_load", .1391)


@pytest.mark.parametrize("prefix", ["", "GPU-"])
def test_monitor_selects_allocated_uuid(prefix):
    value = "778768b4-6c9e-e483-890e-0812760948ae"
    assert probe.nvidia_uuid(prefix + value) == "GPU-" + value


@pytest.mark.parametrize("mode", ["success", "slow", "capture_error"])
def test_entrypoint_preserves_failure_and_never_overwrites_results(tmp_path, monkeypatch, mode):
    directory = tmp_path / "diagnostic"
    monkeypatch.setattr(probe.shutil, "which", lambda _: "/bin/unused")

    def run(path, seconds):
        (path / "evidence.json").write_text("[]")
        if mode == "capture_error":
            raise RuntimeError("graph capture unsupported")
        return int(mode == "slow")

    monkeypatch.setattr(probe, "run", run)
    expected = {"success": 0, "slow": 1, "capture_error": 2}[mode]
    assert probe.main(["--output", str(directory)]) == expected
    assert (directory / "exitcode.txt").read_text().strip() == str(expected)
    assert (directory / "evidence.json").exists()
    if expected == 2:
        assert "graph capture unsupported" in (directory / "error.txt").read_text()
    with pytest.raises(FileExistsError):
        probe.main(["--output", str(directory)])


def test_monitor_cleanup_after_workload_failure(fake_gpu, tmp_path, monkeypatch):
    torch, ffi, call, output, state = fake_gpu
    graph = probe.capture_graph(torch, ffi, call, output, 42)
    commands, snapshots = [], []
    monitor = SimpleNamespace(stopped=False)
    monitor.poll = lambda: 0 if monitor.stopped else None
    monitor.terminate = lambda: setattr(monitor, "stopped", True)
    monitor.wait = lambda **kw: 0

    def start(command, **kwargs):
        commands.append(command)
        kwargs["stdout"].write("timestamp,uuid\n")
        return monitor

    monkeypatch.setattr(probe.subprocess, "Popen", start)
    monkeypatch.setattr(probe, "snapshot", lambda *args: snapshots.append(args[1]))
    def broken_call():
        raise RuntimeError("GPU execution failure")
    with pytest.raises(RuntimeError, match="GPU execution failure"):
        probe.load_telemetry(torch, broken_call, graph, output, 42, tmp_path, "GPU-test", 1)
    assert monitor.stopped
    assert commands[0][1:3] == ["-i", "GPU-test"]
    assert snapshots == ["gpu_before_load"]


@pytest.mark.parametrize("value", ["0", "11", "nan"])
def test_invalid_load_duration_rejected_before_gpu_imports(value):
    with pytest.raises(SystemExit) as error:
        probe.main(["--load-seconds", value])
    assert error.value.code == 2
