"""Top level run script"""

import json
import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from util import dictconfig_to_json

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
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
    
    # Save config as JSON inside the hydra run directory
    run_dir = Path.cwd()
    config_path = run_dir / ".hydra/config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info(f"Saved configuration to {config_path}")


def generate_jobs_with_args(override_list: list[str]):
    """Programmatic entry point."""
    sys.argv = ["run_capsule.py"] + override_list
    generate_jobs()

if __name__ == "__main__":
    # Controled by input args
    generate_jobs()
    
    # Overwrite by python code
    #generate_jobs_with_args(["data=mice", "model=disrnn"])