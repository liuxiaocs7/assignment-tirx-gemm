"""Check every saved timing sample against the unchanged assignment limits.

This diagnostic is deliberately stricter than the benchmark's median status.
It does not change grading or establish stability beyond the observed samples.
"""

import argparse
import ast
import csv
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def assignment_limits():
    values = {}
    for node in ast.parse((ROOT / 'utils.py').read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('REFERENCE_TIMES', 'TIMING_TOLERANCE'):
                    values[target.id] = ast.literal_eval(node.value)
    return {key: value * values['TIMING_TOLERANCE'] for key, value in values['REFERENCE_TIMES'].items()}


def summarize(path, expected_trials=None):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('CSV contains no benchmark rows')
    limits, seen, total, failures = assignment_limits(), set(), 0, 0
    for row in rows:
        key = tuple(int(row[name]) for name in ('step', 'M', 'N', 'K'))
        if key not in limits or key in seen:
            raise ValueError(f'Unscored or duplicate shape: {key}')
        seen.add(key)
        limit = limits[key]
        if not math.isclose(float(row['limit_ms']), limit, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f'Recorded limit differs from assignment: {key}')
        samples = ast.literal_eval(row['samples_ms'])
        if not isinstance(samples, list) or not samples or len(samples) != int(row['trials']):
            raise ValueError(f'Missing/incomplete sample array: {key}')
        if expected_trials is not None and len(samples) != expected_trials:
            raise ValueError(f'Expected {expected_trials} samples, got {len(samples)}: {key}')
        if any(not isinstance(s, (int, float)) or not math.isfinite(s) or s <= 0 for s in samples):
            raise ValueError(f'Invalid timing sample: {key}')
        passed = sum(s <= limit for s in samples)
        total += len(samples)
        failures += len(samples) - passed
        print(f'Step {key[0]} / {key[1:]}: {passed}/{len(samples)} samples within {limit:.6f} ms')
        print(f'  min={min(samples):.6f} median={statistics.median(samples):.6f} '
              f'max={max(samples):.6f} ms; worst margin={100*(1-max(samples)/limit):+.3f}%')
        print('  samples: ' + ', '.join(f'{s:.6f} {"PASS" if s <= limit else "SLOW"}' for s in samples))
    print(f'Observed samples: {total-failures}/{total} PASS. '
          'This is a diagnostic, not a replacement for full pytest acceptance.')
    return int(failures > 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--expected-trials', type=int)
    args = parser.parse_args(argv)
    if args.expected_trials is not None and args.expected_trials < 1:
        parser.error('--expected-trials must be positive')
    try:
        return summarize(args.csv, args.expected_trials)
    except (OSError, ValueError, SyntaxError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
