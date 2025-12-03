"""
Helper Utilities
Author: Novan
"""

import os
import sys
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


def get_project_root():
    """
    Get project root directory
    
    Returns:
        Path: Project root path
    """
    current = Path.cwd()
    
    # Check if we're already in project root
    if (current / 'config' / 'config.yaml').exists():
        return current
    
    # Check if we're in notebooks directory
    if current.name == 'notebooks' and (current.parent / 'config' / 'config.yaml').exists():
        return current.parent
    
    # Search upwards for project root
    while current != current.parent:
        if (current / 'config' / 'config.yaml').exists():
            return current
        current = current.parent
    
    raise FileNotFoundError("Could not find project root directory")


def setup_notebook_environment():
    """
    Setup environment for Jupyter notebooks
    
    Returns:
        Path: Project root path
    """
    # Get project root
    project_root = get_project_root()
    
    # Add to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Change to project root
    os.chdir(project_root)
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Current directory: {Path.cwd()}")
    
    return project_root


def load_config_safe(config_path='config/config.yaml'):
    """
    Load configuration file with automatic path resolution
    
    Args:
        config_path (str): Relative path to config file
        
    Returns:
        dict: Configuration dictionary
    """
    project_root = get_project_root()
    full_path = project_root / config_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {full_path}")
    
    with open(full_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded from: {full_path}")
    return config


def create_dirs_from_config(config):
    """
    Create all directories specified in config
    
    Args:
        config (dict): Configuration dictionary
    """
    project_root = get_project_root()
    
    paths_config = config.get('paths', {})
    
    for key, path in paths_config.items():
        full_path = project_root / path
        full_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ensured: {full_path}")
