#!/usr/bin/env python3
"""Linux host memory guard for an explicitly named benchmark model container.

Run on each model host before loading the model. This only samples /proc and
stops the named container if either memory floor is crossed. It does not change
host settings or stop unrelated containers. Coordinate stopping the peer rank
from the benchmark controller if this process exits nonzero.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time


def snapshot():
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        values[key] = int(value.split()[0])
    full = next(line for line in Path('/proc/pressure/memory').read_text().splitlines()
                if line.startswith('full '))
    psi = float(dict(item.split('=') for item in full.split()[1:])['avg10'])
    return {'time': time.time(), 'available_kb': values['MemAvailable'],
            'free_kb': values['MemFree'], 'full_psi10': psi}


def below_floor(sample):
    return sample['available_kb'] < 768 * 1024 or sample['free_kb'] < 512 * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container', required=True, help='only this model container may be stopped')
    args = parser.parse_args()
    if args.container.startswith('-'):
        parser.error('container name must not start with a dash')
    while True:
        sample = snapshot()
        print(json.dumps(sample), flush=True)
        if below_floor(sample):
            print('SAFETY_ABORT memory floor reached; stopping ' + args.container, flush=True)
            subprocess.run(['docker', 'stop', '-t', '2', args.container], timeout=20, check=True)
            return 1
        time.sleep(1)


if __name__ == '__main__':
    raise SystemExit(main())
