import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hydra
from omegaconf import DictConfig

from atif.tools import inference, optimization


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main function.

    Args:
        cfg (DictConfig): configs.
    """
    if cfg.mode == "inference":
        inference(cfg.inference)
    else:
        optimization(cfg.optimization)


if __name__ == '__main__':
    main()
