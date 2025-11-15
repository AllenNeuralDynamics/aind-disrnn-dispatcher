"""Top level run script"""

import json
import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

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


def generate_jobs_with_args(override_list: list[str]):
    """Programmatic entry point."""
    sys.argv = ["run_capsule.py"] + override_list
    generate_jobs()

if __name__ == "__main__":
    generate_jobs()
    
    #generate_jobs_with_args(["data=mice", "model=disrnn"])