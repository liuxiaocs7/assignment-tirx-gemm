"""Exercise diagnostic shell failure handling without pretending to run a GPU."""

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def diagnostic_workspace(tmp_path):
    for name in ('run_step10_diagnose.sh', 'utils.py'):
        shutil.copy2(ROOT / name, tmp_path / name)
    (tmp_path / 'tools').mkdir()
    shutil.copy2(ROOT / 'tools/summarize_benchmark_stability.py', tmp_path / 'tools')
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    # Only external GPU programs are substituted. Shell control flow, tee,
    # exit-code capture and the CSV stability analysis are real.
    uv = bin_dir / 'uv'
    uv.write_text('''#!/usr/bin/env python3
import csv, json, os, pathlib, sys
args = sys.argv[1:]
with pathlib.Path('calls.jsonl').open('a') as f:
    f.write(json.dumps(args) + '\\n')
if '-m' in args and 'pytest' in args:
    print('simulated targeted pytest')
    sys.exit(int(os.environ.get('TEST_PYTEST_EXIT', '0')))
if 'benchmark.py' in args:
    values = json.loads(os.environ.get('TEST_SAMPLES', '[0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13]'))
    path = pathlib.Path(args[args.index('--csv')+1])
    row = dict(step=10, M=4096, N=4096, K=4096, trials=len(values),
               reference_ms=.107, limit_ms=.1391, samples_ms=json.dumps(values))
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=row); w.writeheader(); w.writerow(row)
    print('simulated benchmark')
    sys.exit(int(os.environ.get('TEST_BENCHMARK_EXIT', '0')))
print(json.dumps(dict(uuid='GPU-test', gpu='test B300', SMs=148)))
sys.exit(int(os.environ.get('TEST_DEVICE_EXIT', '0')))
''')
    uv.chmod(0o755)
    gpu = bin_dir / 'nvidia-smi'
    gpu.write_text('#!/bin/sh\necho "simulated GPU snapshot"\nexit "${TEST_GPU_EXIT:-0}"\n')
    gpu.chmod(0o755)
    env = dict(os.environ, PATH=f'{bin_dir}:{os.environ["PATH"]}')
    return tmp_path, env


@pytest.mark.parametrize('pytest_exit,benchmark_exit,samples,expected', [
    (0, 0, [.13]*8, 0),
    (1, 0, [.13]*8, 1),  # Failure must not prevent the formal benchmark.
    (0, 1, [.14]*8, 1),
    (0, 0, [.13]*7+[.14], 1),  # Median PASS can conceal a slow raw sample.
])
def test_diagnostic_keeps_all_runs_and_failure_evidence(
    diagnostic_workspace, pytest_exit, benchmark_exit, samples, expected,
):
    workspace, env = diagnostic_workspace
    env.update(TEST_PYTEST_EXIT=str(pytest_exit), TEST_BENCHMARK_EXIT=str(benchmark_exit),
               TEST_SAMPLES=json.dumps(samples))
    result = subprocess.run(['bash', 'run_step10_diagnose.sh'], cwd=workspace,
                            env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == expected, result.stdout + result.stderr
    calls = [json.loads(line) for line in (workspace/'calls.jsonl').read_text().splitlines()]
    tests = [c for c in calls if 'pytest' in c]
    benches = [c for c in calls if 'benchmark.py' in c]
    assert len(tests) == 3 and len(benches) == 1
    assert all('tests/test_step10.py::test_multi_consumer[4096]' in c for c in tests)
    assert benches[0][benches[0].index('--trials')+1] == '8'
    assert '--diagnostics-dir' in benches[0]
    run, = (workspace/'results_b300').iterdir()
    for i in range(1,4):
        assert (run/f'pytest_{i}_exitcode.txt').read_text().strip() == str(pytest_exit)
        assert (run/f'pytest_{i}_pipeline.txt').read_text().strip() == f'command={pytest_exit} tee=0'
    assert (run/'benchmark_exitcode.txt').read_text().strip() == str(benchmark_exit)
    assert (run/'summary_exitcode.txt').read_text().strip() == str(int(any(v > .1391 for v in samples)))
    assert (run/'gpu_after.txt').exists()
    assert (run/'overall_exitcode.txt').read_text().strip() == str(expected)
    assert f'{sum(v <= .1391 for v in samples)}/8' in (run/'summary.log').read_text()


def test_device_failure_stops_before_tests(diagnostic_workspace):
    workspace, env = diagnostic_workspace
    env['TEST_DEVICE_EXIT'] = '3'
    result = subprocess.run(['bash','run_step10_diagnose.sh'], cwd=workspace,
                            env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 3
    calls = (workspace/'calls.jsonl').read_text()
    assert 'pytest' not in calls and 'benchmark.py' not in calls


def test_bad_arguments_do_not_create_run(diagnostic_workspace):
    workspace, env = diagnostic_workspace
    result = subprocess.run(['bash','run_step10_diagnose.sh','--unknown'], cwd=workspace,
                            env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 2 and not (workspace/'results_b300').exists()


@pytest.mark.parametrize('broken', ['tee', 'snapshot'])
def test_capture_failure_cannot_report_success(diagnostic_workspace, broken):
    workspace, env = diagnostic_workspace
    if broken == 'tee':
        shim = workspace/'bin/tee'
        shim.write_text('#!/bin/sh\ncat > "$1"\nexit 4\n')
        shim.chmod(0o755)
    else:
        env['TEST_GPU_EXIT'] = '5'
    result = subprocess.run(['bash','run_step10_diagnose.sh'], cwd=workspace,
                            env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 1
    run, = (workspace/'results_b300').iterdir()
    assert (run/'step10.csv').exists() and (run/'gpu_after.txt').exists()
    if broken == 'tee':
        assert (run/'benchmark_pipeline.txt').read_text().strip() == 'command=0 tee=4'
    else:
        assert (run/'gpu_after_exitcode.txt').read_text().strip() == '5'


@pytest.mark.parametrize('changed,value', [
    ('limit_ms', .2), ('samples_ms', '[]'), ('samples_ms', '[0.13, -0.1]'),
    ('samples_ms', '[0.13, 1e999]'), ('trials', 3),
])
def test_invalid_stability_evidence_is_rejected(tmp_path, changed, value):
    row = dict(step=10,M=4096,N=4096,K=4096,trials=2,limit_ms=.1391,samples_ms='[0.13, 0.13]')
    row[changed] = value
    path = tmp_path/'samples.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=row)
        writer.writeheader(); writer.writerow(row)
    result = subprocess.run(['python3',str(ROOT/'tools/summarize_benchmark_stability.py'),
                             str(path),'--expected-trials','2'], capture_output=True, text=True, timeout=5)
    assert result.returncode == 2, result.stdout + result.stderr
