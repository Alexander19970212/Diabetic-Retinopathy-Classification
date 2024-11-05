import os
import yaml


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file not found: {config_path}')

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return config
