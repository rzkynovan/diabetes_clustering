"""
Data Loading Module for Diabetes Clustering Project
Author: Novan
"""

import os
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import requests
from tqdm import tqdm
import logging
import zipfile
import io

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiabetesDataLoader:
    """
    Data loader for Diabetes 130-US Hospitals dataset
    """
    
    def __init__(self, config_path='config/config.yaml'):
        """
        Initialize data loader with configuration
        
        Args:
            config_path (str): Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_raw_path = self.config['paths']['data_raw']
        self.data_processed_path = self.config['paths']['data_processed']
        self.dataset_name = self.config['data']['dataset_name']
        
        # Create directories if not exist
        os.makedirs(self.data_raw_path, exist_ok=True)
        os.makedirs(self.data_processed_path, exist_ok=True)
    
    def download_dataset(self, force_download=False):
        """
        Download diabetes dataset from UCI repository
        
        Args:
            force_download (bool): Force re-download even if file exists
        """
        file_path = os.path.join(self.data_raw_path, self.dataset_name)
        
        if os.path.exists(file_path) and not force_download:
            logger.info(f"Dataset already exists at {file_path}")
            return file_path
        
        logger.info("Downloading Diabetes 130-US Hospitals dataset...")
        
        # Updated UCI dataset URLs (try multiple sources)
        urls = [
            # Primary source: UCI new archive
            "https://archive.ics.uci.edu/static/public/296/diabetes+130+us+hospitals+for+years+1999+2008.zip",
            # Alternative source: Direct CSV link (if available)
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"
        ]
        
        success = False
        
        for url_idx, url in enumerate(urls, 1):
            try:
                logger.info(f"Attempting download from source {url_idx}/{len(urls)}...")
                logger.info(f"URL: {url}")
                
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                # Check if it's a zip file
                content_type = response.headers.get('content-type', '')
                
                if 'zip' in content_type or url.endswith('.zip'):
                    logger.info("Detected ZIP archive, extracting...")
                    
                    # Download to memory
                    zip_content = io.BytesIO()
                    
                    with tqdm(
                        desc="Downloading",
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            zip_content.write(chunk)
                            pbar.update(len(chunk))
                    
                    # Extract CSV from ZIP
                    with zipfile.ZipFile(zip_content) as zip_ref:
                        # List files in ZIP
                        file_list = zip_ref.namelist()
                        logger.info(f"Files in archive: {file_list}")
                        
                        # Find CSV file
                        csv_files = [f for f in file_list if f.endswith('.csv')]
                        
                        if csv_files:
                            # Extract the main dataset CSV
                            target_csv = csv_files[0]  # Usually 'diabetic_data.csv'
                            logger.info(f"Extracting: {target_csv}")
                            
                            with zip_ref.open(target_csv) as csv_file:
                                with open(file_path, 'wb') as f:
                                    f.write(csv_file.read())
                            
                            logger.info(f"Dataset downloaded and extracted to {file_path}")
                            success = True
                            break
                        else:
                            logger.warning("No CSV file found in archive")
                            continue
                
                else:
                    # Direct CSV download
                    with open(file_path, 'wb') as f, tqdm(
                        desc="Downloading",
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                    
                    logger.info(f"Dataset downloaded successfully to {file_path}")
                    success = True
                    break
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to download from source {url_idx}: {e}")
                continue
            except zipfile.BadZipFile as e:
                logger.warning(f"Invalid ZIP file from source {url_idx}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Unexpected error with source {url_idx}: {e}")
                continue
        
        if not success:
            logger.error("Failed to download from all sources")
            logger.info("\n" + "="*70)
            logger.info("MANUAL DOWNLOAD REQUIRED")
            logger.info("="*70)
            logger.info("\nPlease download the dataset manually:")
            logger.info("1. Visit: https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008")
            logger.info("2. Click 'Download' button")
            logger.info("3. Extract the ZIP file")
            logger.info("4. Copy 'diabetic_data.csv' to: " + os.path.abspath(self.data_raw_path))
            logger.info("5. Rename it to: " + self.dataset_name)
            logger.info("\nAlternative (using Kaggle):")
            logger.info("1. Visit: https://www.kaggle.com/datasets/brandao/diabetes")
            logger.info("2. Download 'diabetic_data.csv'")
            logger.info("3. Place it in: " + os.path.abspath(self.data_raw_path))
            logger.info("="*70 + "\n")
            raise RuntimeError("Dataset download failed. Please download manually (see instructions above).")
        
        return file_path
    
    def load_raw_data(self):
        """
        Load raw dataset from CSV
        
        Returns:
            pd.DataFrame: Raw diabetes dataset
        """
        file_path = os.path.join(self.data_raw_path, self.dataset_name)
        
        if not os.path.exists(file_path):
            logger.warning("Dataset not found. Attempting to download...")
            self.download_dataset()
        
        logger.info(f"Loading dataset from {file_path}")
        
        try:
            # Try with different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"Dataset loaded successfully with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                df = pd.read_csv(file_path)  # Let pandas auto-detect
            
            logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def get_data_info(self, df):
        """
        Get comprehensive information about the dataset
        
        Args:
            df (pd.DataFrame): Dataset
            
        Returns:
            dict: Dataset information
        """
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'duplicates': df.duplicated().sum(),
        }
        
        # Categorical vs Numerical columns
        info['categorical_columns'] = df.select_dtypes(include=['object']).columns.tolist()
        info['numerical_columns'] = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Unique values for categorical columns
        info['unique_values'] = {}
        for col in info['categorical_columns']:
            info['unique_values'][col] = df[col].nunique()
        
        return info
    
    def save_processed_data(self, df, filename):
        """
        Save processed data to CSV
        
        Args:
            df (pd.DataFrame): Processed dataset
            filename (str): Output filename
        """
        output_path = os.path.join(self.data_processed_path, filename)
        df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")


def load_config(config_path='config/config.yaml'):
    """
    Utility function to load configuration
    
    Args:
        config_path (str): Path to config file
        
    Returns:
        dict: Configuration dictionary
    """
    import os
    from pathlib import Path
    
    # Handle relative path from different locations
    current_dir = Path.cwd()
    
    # Try multiple possible locations
    possible_paths = [
        Path(config_path),  # Direct path
        current_dir / config_path,  # From current directory
        current_dir.parent / config_path,  # From parent directory (if in notebooks/)
        current_dir / '..' / config_path,  # Relative to parent
    ]
    
    # Find project root (directory containing config folder)
    project_root = current_dir
    while project_root != project_root.parent:
        if (project_root / 'config' / 'config.yaml').exists():
            possible_paths.append(project_root / 'config' / 'config.yaml')
            break
        project_root = project_root.parent
    
    # Try each path
    for path in possible_paths:
        if path.exists():
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from: {path}")
            return config
    
    # If not found, raise error with helpful message
    raise FileNotFoundError(
        f"Configuration file not found. Tried paths:\n" + 
        "\n".join([f"  - {p}" for p in possible_paths]) +
        f"\n\nCurrent directory: {current_dir}" +
        f"\n\nPlease ensure config/config.yaml exists in project root."
    )


def set_random_seed(seed=42):
    """
    Set random seed for reproducibility
    
    Args:
        seed (int): Random seed
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def download_from_kaggle():
    """
    Alternative: Download dataset from Kaggle using kaggle API
    Requires: pip install kaggle
    """
    try:
        import kaggle
        
        logger.info("Downloading from Kaggle...")
        kaggle.api.dataset_download_files(
            'brandao/diabetes',
            path='data/raw/',
            unzip=True
        )
        logger.info("Download from Kaggle successful!")
        return True
        
    except ImportError:
        logger.warning("Kaggle API not installed. Install with: pip install kaggle")
        return False
    except Exception as e:
        logger.warning(f"Kaggle download failed: {e}")
        return False


if __name__ == "__main__":
    # Test the data loader
    print("\n" + "="*70)
    print("DIABETES DATASET LOADER - TEST RUN")
    print("="*70 + "\n")
    
    loader = DiabetesDataLoader()
    
    # Try to download dataset
    try:
        loader.download_dataset()
    except RuntimeError as e:
        print("\n⚠️  Automatic download failed")
        print("\nTrying Kaggle alternative...")
        
        if not download_from_kaggle():
            print("\n" + "="*70)
            print("MANUAL DOWNLOAD INSTRUCTIONS")
            print("="*70)
            print("\nOption 1 - UCI Repository:")
            print("1. Visit: https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008")
            print("2. Click 'Download' button")
            print("3. Extract ZIP file")
            print("4. Copy 'diabetic_data.csv' to: data/raw/")
            print("")
            print("Option 2 - Kaggle:")
            print("1. Visit: https://www.kaggle.com/datasets/brandao/diabetes")
            print("2. Download dataset")
            print("3. Extract and copy 'diabetic_data.csv' to: data/raw/")
            print("")
            print("Option 3 - Direct Link (may work):")
            print("1. Try: https://www.kaggle.com/api/v1/datasets/download/brandao/diabetes")
            print("2. Extract and copy to: data/raw/")
            print("="*70 + "\n")
            
            import sys
            sys.exit(1)
    
    # Load data
    try:
        df = loader.load_raw_data()
        
        # Get info
        info = loader.get_data_info(df)
        
        print("\n" + "="*70)
        print("DATASET INFORMATION")
        print("="*70)
        print(f"Shape: {info['shape']}")
        print(f"Memory Usage: {info['memory_usage']:.2f} MB")
        print(f"Duplicates: {info['duplicates']}")
        print(f"\nCategorical Columns: {len(info['categorical_columns'])}")
        print(f"Numerical Columns: {len(info['numerical_columns'])}")
        print(f"\nColumns with Missing Values:")
        for col, pct in info['missing_percentage'].items():
            if pct > 0:
                print(f"  {col}: {pct:.2f}%")
        
        print("\n" + "="*70)
        print("✅ DATA LOADER TEST SUCCESSFUL!")
        print("="*70 + "\n")
        
    except FileNotFoundError:
        print("\n❌ Dataset file not found. Please download manually.")
        print("See instructions above.\n")
