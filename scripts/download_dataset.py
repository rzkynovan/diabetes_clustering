"""
Interactive Dataset Download Script
Author: Novan
"""

import os
import sys
import requests
from pathlib import Path

def download_with_progress(url, output_path):
    """Download file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                
                # Simple progress
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rDownloading: {percent:.1f}%", end='', flush=True)
        
        print("\n✅ Download complete!")
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False

def main():
    print("="*70)
    print("Diabetes Dataset Download Assistant")
    print("="*70)
    print()
    
    # Create directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = data_dir / "diabetic_data.csv"
    
    if output_file.exists():
        print(f"✅ Dataset already exists at: {output_file}")
        overwrite = input("Do you want to re-download? (y/n): ").lower()
        if overwrite != 'y':
            print("Keeping existing file.")
            return
    
    print("\nAttempting automatic download...")
    print()
    
    # Try Kaggle direct link (may not work without auth)
    urls_to_try = [
        ("Kaggle Mirror", "https://www.kaggle.com/api/v1/datasets/download/brandao/diabetes"),
    ]
    
    for name, url in urls_to_try:
        print(f"Trying {name}...")
        if download_with_progress(url, output_file):
            print(f"✅ Dataset downloaded successfully to: {output_file}")
            return
        print(f"Failed. Trying next source...\n")
    
    # All automatic methods failed
    print("\n" + "="*70)
    print("⚠️  AUTOMATIC DOWNLOAD FAILED - MANUAL DOWNLOAD REQUIRED")
    print("="*70)
    print()
    print("Please download the dataset manually using one of these methods:")
    print()
    print("Method 1 - Kaggle (Easiest):")
    print("  1. Visit: https://www.kaggle.com/datasets/brandao/diabetes")
    print("  2. Click 'Download' (requires free Kaggle account)")
    print("  3. Extract diabetic_data.csv")
    print(f"  4. Place it here: {output_file.absolute()}")
    print()
    print("Method 2 - UCI Repository:")
    print("  1. Visit: https://archive.ics.uci.edu/dataset/296")
    print("  2. Click 'Download' button")
    print("  3. Extract diabetic_data.csv from ZIP")
    print(f"  4. Place it here: {output_file.absolute()}")
    print()
    print("After downloading, run this script again to verify.")
    print("="*70)
    print()

if __name__ == "__main__":
    main()
