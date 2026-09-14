"""Run the real shell wrapper with external GPU programs substituted."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('runner,variants', [
    ('run_step10_epilogue.sh', ['tmem_epi64', 'tmem_epi128', 'tmem_epi32_double']),
    ('run_step10_ready.sh', ['tmem_split_ready']),
])
@pytest.mark.parametrize('failure', ['', 'device', 'probe', 'tee', 'snapshot'])
def test_epilogue_runner_saves_failures_and_balanced_command(tmp_path, failure, runner, variants):
    shutil.copy2(ROOT / runner, tmp_path)
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    uv = bin_dir / 'uv'
    uv.write_text('''#!/usr/bin/env python3
import json,os,pathlib,sys
args=sys.argv[1:]
with pathlib.Path('calls.jsonl').open('a') as f: f.write(json.dumps(args)+'\\n')
if 'probe_persistent.py' in args:
    print('simulated baseline SLOW; probe completed')
    sys.exit(3 if os.environ['TEST_FAILURE']=='probe' else 0)
print('{}')
sys.exit(4 if os.environ['TEST_FAILURE']=='device' else 0)
''')
    gpu = bin_dir / 'nvidia-smi'
    gpu.write_text('#!/bin/sh\necho snapshot\n[ "$TEST_FAILURE" != snapshot ]\n')
    for executable in (uv, gpu):
        executable.chmod(0o755)
    if failure == 'tee':
        tee = bin_dir / 'tee'
        tee.write_text('#!/bin/sh\ncat > "$1"\nexit 5\n')
        tee.chmod(0o755)
    env = dict(os.environ, PATH=f'{bin_dir}:{os.environ["PATH"]}', TEST_FAILURE=failure)
    result = subprocess.run(['bash', runner], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=15)
    run, = (tmp_path / 'results_b300').iterdir()
    calls = [json.loads(line) for line in (tmp_path / 'calls.jsonl').read_text().splitlines()]
    if failure == 'device':
        assert result.returncode == 4 and len(calls) == 1
        assert (run / 'device_exitcode.txt').read_text().strip() == '4'
        return
    assert result.returncode == int(bool(failure))
    probe, = [c for c in calls if 'probe_persistent.py' in c]
    assert probe[probe.index('--size')+1] == '4096'
    for key, value in [('--trials', '8'), ('--warmup', '10'), ('--repeat', '30')]:
        assert probe[probe.index(key)+1] == value
    assert probe[probe.index('--variants')+1:probe.index('--output')] == variants
    assert (run / 'gpu_after.txt').exists()
    assert (run / 'overall_exitcode.txt').read_text().strip() == str(int(bool(failure)))
    assert (run / 'pipeline.txt').read_text().strip() == (
        f'probe={3 if failure == "probe" else 0} tee={5 if failure == "tee" else 0}')


@pytest.mark.parametrize('runner', ['run_step10_epilogue.sh', 'run_step10_ready.sh'])
def test_bad_arguments_exit_before_any_run(tmp_path, runner):
    shutil.copy2(ROOT / runner, tmp_path)
    result = subprocess.run(['bash', runner, '--unexpected'], cwd=tmp_path,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 2 and not (tmp_path / 'results_b300').exists()
