#!/bin/bash

# Download Diabetes Dataset Helper Script
# Author: Novan

echo "======================================================================"
echo "Diabetes Dataset Download Helper"
echo "======================================================================"
echo ""

# Create data directory
mkdir -p data/raw

echo "Attempting to download dataset..."
echo ""

# Try direct download from Kaggle mirror (may require authentication)
echo "Option 1: Trying direct download..."
curl -L -o data/raw/diabetes.zip "https://www.kaggle.com/api/v1/datasets/download/brandao/diabetes" 2>/dev/null

if [ -f "data/raw/diabetes.zip" ]; then
    echo "✅ Download successful! Extracting..."
    unzip -o data/raw/diabetes.zip -d data/raw/
    mv data/raw/diabetic_data.csv data/raw/diabetic_data.csv 2>/dev/null
    rm data/raw/diabetes.zip
    echo "✅ Dataset ready at: data/raw/diabetic_data.csv"
    exit 0
fi

echo ""
echo "⚠️  Automatic download failed. Manual download required."
echo ""
echo "======================================================================"
echo "MANUAL DOWNLOAD INSTRUCTIONS"
echo "======================================================================"
echo ""
echo "Please choose one of these options:"
echo ""
echo "Option A - Kaggle (Recommended):"
echo "  1. Visit: https://www.kaggle.com/datasets/brandao/diabetes"
echo "  2. Click 'Download' button (requires Kaggle account)"
echo "  3. Extract the ZIP file"
echo "  4. Copy 'diabetic_data.csv' to: $(pwd)/data/raw/"
echo ""
echo "Option B - UCI Repository:"
echo "  1. Visit: https://archive.ics.uci.edu/dataset/296"
echo "  2. Click 'Download' button"
echo "  3. Extract the ZIP file"
echo "  4. Copy 'diabetic_data.csv' to: $(pwd)/data/raw/"
echo ""
echo "Option C - Alternative Sources:"
echo "  1. Search: 'diabetes 130-US hospitals dataset' on Google"
echo "  2. Download from any trusted source"
echo "  3. Ensure filename is: diabetic_data.csv"
echo "  4. Place in: $(pwd)/data/raw/"
echo ""
echo "======================================================================"
echo ""

exit 1
