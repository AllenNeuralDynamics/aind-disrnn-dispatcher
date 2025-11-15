from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

def dictconfig_to_json(cfg: DictConfig) -> dict:
    """Convert Hydra DictConfig to a JSON-serializable dict."""
    return OmegaConf.to_container(cfg, resolve=True)

def json_to_dictconfig(json_data: dict) -> DictConfig:
    """Load JSON dict to Hydra DictConfig."""
    return OmegaConf.create(json_data)