import argparse
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
import torch

from src.logger.wandb import WandBWriter


def get_optimizer(cfg: DictConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    return instantiate(cfg, params=model.parameters())

def get_scheduler(cfg: DictConfig, optimizer: torch.optim.Optimizer):
    return instantiate(cfg, optimizer=optimizer)

def get_criterion(cfg: DictConfig) -> torch.nn.Module:
    return instantiate(cfg)

def get_logger(cfg: DictConfig):
    return WandBWriter(**cfg)

def load_config(config_name: str) -> DictConfig:
    config_path = f"src/config/{config_name}.yaml"
    
    try:
        config = OmegaConf.load(config_path)
        return config
    except FileNotFoundError:
        raise Exception(f"Ошибка: файл '{config_path}' не найден")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', help='Config yaml name', required=True, type=str)
    args = parser.parse_args()

    return load_config(args.config)
