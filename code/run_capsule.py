"""Top level run script"""

import json
import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from config_helpers import dictconfig_to_json

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def generate_jobs(cfg: DictConfig) -> None:
    """Main run function that loads Hydra config and saves to results/jobs"""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(levelname)s:%(asctime)s:%(filename)s:%(lineno)d: %(message)s",
        datefmt="%Y-%m-%d %H-%M-%S",
    )
    
    logger.info("Loaded Hydra configuration")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Convert config to JSON-serializable dict
    config_dict = dictconfig_to_json(cfg)
    
    # Create results/jobs directory if it doesn't exist
    jobs_dir = Path("/root/capsule/results/jobs")
    jobs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config as JSON
    config_path = jobs_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info(f"Saved configuration to {config_path}")


def generate_jobs_with_args(override_list: list[str]):
    """Programmatic entry point."""
    sys.argv = ["run_capsule.py"] + override_list
    generate_jobs()

if __name__ == "__main__":
    #generate_jobs()
    
    generate_jobs_with_args(["data=mice", "model=disrnn"])