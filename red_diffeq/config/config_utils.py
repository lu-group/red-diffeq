from pathlib import Path
from typing import Union
import yaml
import ml_collections

def load_config(config_path: Union[str, Path]) -> ml_collections.ConfigDict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    if config_dict is None:
        config_dict = {}
    config = ml_collections.ConfigDict(config_dict)
    return config

def save_config(config: ml_collections.ConfigDict, output_path: Union[str, Path]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config_dict = config.to_dict()
    config_dict = _convert_tuples_to_lists(config_dict)
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

def _convert_tuples_to_lists(obj):
    if isinstance(obj, dict):
        return {k: _convert_tuples_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return [_convert_tuples_to_lists(item) for item in obj]
    elif isinstance(obj, list):
        return [_convert_tuples_to_lists(item) for item in obj]
    return obj

def update_config(config: ml_collections.ConfigDict, **kwargs) -> ml_collections.ConfigDict:
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: '{key}' not in config, adding it")
            setattr(config, key, value)
    return config

def print_config(config: ml_collections.ConfigDict, prefix: str='') -> None:
    print('=' * 60)
    print('Configuration:')
    print('=' * 60)
    for key, value in sorted(config.items()):
        if isinstance(value, ml_collections.ConfigDict):
            print(f'{prefix}{key}:')
            print_config(value, prefix=prefix + '  ')
        else:
            print(f'{prefix}{key}: {value}')
    if not prefix:
        print('=' * 60)