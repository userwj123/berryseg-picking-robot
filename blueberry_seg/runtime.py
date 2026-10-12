"""Paths, portable dataset binding, and explicit device selection."""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def init_runtime(threads=4):
    runtime = ROOT/'.runtime'
    runtime.mkdir(exist_ok=True)
    config_directory = runtime/'ultralytics'
    config_directory.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(config_directory)
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    os.environ.setdefault('OMP_NUM_THREADS', str(threads))
    os.environ.setdefault('MKL_NUM_THREADS', str(threads))
    import torch
    torch.set_num_threads(threads)
    from ultralytics import settings
    settings.update({'sync': False})


def device(value):
    import torch
    value = str(value)
    if value == 'auto':
        return '0' if torch.cuda.is_available() else 'cpu'
    if value != 'cpu':
        if not torch.cuda.is_available():
            raise RuntimeError('GPU requested but CUDA is unavailable. Install a CUDA PyTorch build or use --device cpu.')
        ids = value.removeprefix('cuda:').split(',')
        if not all(i.isdigit() and 0 <= int(i) < torch.cuda.device_count() for i in ids):
            raise ValueError(f'Invalid CUDA device selection: {value}')
        value = ','.join(ids)
    return value


def bind_dataset(path):
    """Generate machine-specific binding without changing portable input YAML."""
    source = project_path(path)
    if not source.is_file():
        raise FileNotFoundError(f'Dataset YAML missing: {source}. Run blueberry_seg.prepare first.')
    data = yaml.safe_load(source.read_text(encoding='utf-8'))
    root = source.parent if 'path' not in data else (source.parent/Path(data['path'])).resolve()
    data['path'] = str(root)
    if not data.get('names') or not data.get('train') or not data.get('val'):
        raise ValueError('Dataset YAML needs train, val and names')
    for key in ('train', 'val', 'test'):
        if data.get(key):
            values = data[key] if isinstance(data[key], list) else [data[key]]
            for value in values:
                if not (root/Path(value)).exists():
                    raise FileNotFoundError(f'Dataset {key} path missing: {root/Path(value)}')
    bound = ROOT/'.runtime'/'datasets'/f'{source.parent.name}_{source.stem}.yaml'
    bound.parent.mkdir(parents=True, exist_ok=True)
    bound.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return bound

