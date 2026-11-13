#!/usr/bin/env python3
"""Read-only Ubuntu/CUDA readiness report. No installs, settings changes or model downloads.

Run in the same Python environment as harmonize.py. CUDA inspection may create a
small driver context until this process exits. It does not load any model.
Exit 1 means a definite prerequisite is missing; warnings do not block execution.
This is not a guarantee that inference fits memory or runs on RTX 2080.
"""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

GIB = 1024**3
PACKAGES = ('torch','diffusers','transformers','accelerate','bitsandbytes','numpy','Pillow','scipy','safetensors','sentencepiece')


def memory_info(path='/proc/meminfo'):
    try:
        values = {}
        for line in Path(path).read_text().splitlines():
            key, value = line.split(':',1)
            fields = value.split()
            if fields:
                values[key] = int(fields[0])*1024
        return {'total_bytes':values.get('MemTotal'), 'available_bytes':values.get('MemAvailable'),
                'swap_total_bytes':values.get('SwapTotal'), 'swap_free_bytes':values.get('SwapFree')}
    except (OSError, ValueError):
        return None


def assess(report, budget_gib):
    errors, warnings = [], []
    if report['platform'] != 'Linux':
        warnings.append('This preflight is intended for Ubuntu/Linux; other platforms have not been validated.')
    missing = [name for name, version in report['packages'].items() if version is None]
    if missing: errors.append('Missing Python packages: ' + ', '.join(missing))
    cuda = report.get('cuda', {})
    if not cuda.get('available'):
        errors.append('PyTorch cannot access CUDA. Check the CUDA-enabled PyTorch build, NVIDIA driver and GPU visibility.')
    else:
        if cuda.get('error'):
            errors.append('CUDA inspection failed: ' + cuda['error'])
        capability = cuda.get('compute_capability')
        if capability and tuple(capability) < (7,5):
            warnings.append('GPU is older than the RTX 2080/Turing target; this configuration has not been validated.')
        if capability:
            target = 'sm_' + ''.join(str(x) for x in capability)
            arches = cuda.get('torch_arch_list', [])
            if target not in arches:
                warnings.append(f'PyTorch does not list native {target} support ({arches}). Compatible cubin/PTX may work, but verify this build before inference.')
        free, total = cuda.get('free_bytes'), cuda.get('total_bytes')
        if total is not None and total < budget_gib*GIB:
            errors.append(f'GPU total memory is smaller than the requested {budget_gib:g} GiB allocator budget.')
        if free is not None and free < (budget_gib+.5)*GIB:
            warnings.append(f'Free VRAM is below {budget_gib+.5:g} GiB (allocator budget plus a minimal 0.5 GiB driver margin). Close GPU-heavy applications or use a smaller budget/resolution; no automatic changes were made.')
    ram = report.get('ram')
    if not ram or ram.get('available_bytes') is None:
        warnings.append('Available system RAM could not be read.')
    elif ram['available_bytes'] < 16*GIB:
        warnings.append('Less than 16 GiB system RAM is currently available. First-load conversion can use substantial RAM; 16 GiB is a heuristic, not a tested minimum or guarantee.')
    return errors, warnings


def collect(torch_loader=lambda: importlib.import_module('torch')):
    packages = {}
    for name in PACKAGES:
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    report = {'platform':platform.system(), 'platform_details':platform.platform(),
              'python':sys.version, 'python_executable':sys.executable,
              'packages':packages, 'ram':memory_info(),
              'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
              'cuda':{}, 'nvidia_smi':None}
    try:
        report['nvidia_smi'] = subprocess.check_output(
            ['nvidia-smi','--query-gpu=index,name,driver_version,memory.total,memory.free','--format=csv'],
            text=True,timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        report['nvidia_smi_error'] = str(exc)
    try:
        torch = torch_loader()
        report['cuda'] = {'available':bool(torch.cuda.is_available()), 'torch_cuda_version':torch.version.cuda}
        if report['cuda']['available']:
            # Logical device 0 matches the model backend, respecting CUDA_VISIBLE_DEVICES.
            gpu = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info(0)
            report['cuda'].update({'logical_device':0,'name':gpu.name,
                                  'compute_capability':list(torch.cuda.get_device_capability(0)),
                                  'torch_arch_list':torch.cuda.get_arch_list(),
                                  'total_bytes':int(total),'free_bytes':int(free)})
    except Exception as exc:
        report['cuda']['error'] = f'{type(exc).__name__}: {exc}'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory-gib', type=float, default=6.5)
    parser.add_argument('--json', action='store_true', help='Print machine-readable report')
    args = parser.parse_args()
    if not 0 < args.memory_gib <= 8: parser.error('--memory-gib must be in (0,8]')
    report = collect()
    errors, warnings = assess(report,args.memory_gib)
    report.update({'errors':errors,'warnings':warnings,'status':'blocked' if errors else ('review-warnings' if warnings else 'prerequisites-found'),
                   'requested_allocator_gib':args.memory_gib,
                   'limitation':'No model inference tested; physical RTX 2080 model inference remains unverified. Available memory can change after this check.'})
    if args.json:
        print(json.dumps(report,indent=2))
    else:
        print('Status: '+report['status'])
        print('Python: '+report['python_executable'])
        print('Packages: '+', '.join(f'{k}={v or "MISSING"}' for k,v in report['packages'].items()))
        cuda = report['cuda']
        print(f'GPU (logical 0): {cuda.get("name","unavailable")}; capability={cuda.get("compute_capability")}; torch CUDA={cuda.get("torch_cuda_version")}')
        print('PyTorch architectures: '+str(cuda.get('torch_arch_list',[])))
        for key in ('free_bytes','total_bytes'):
            if key in cuda: print(f'GPU {key}: {cuda[key]/GIB:.2f} GiB')
        if report['ram']:
            for key,value in report['ram'].items():
                if value is not None: print(f'RAM {key}: {value/GIB:.2f} GiB')
        if report['nvidia_smi']: print('\nnvidia-smi:\n'+report['nvidia_smi'])
        for message in errors: print('ERROR: '+message)
        for message in warnings: print('WARNING: '+message)
        print(report['limitation'])
    return 1 if errors else 0

if __name__ == '__main__':
    sys.exit(main())
