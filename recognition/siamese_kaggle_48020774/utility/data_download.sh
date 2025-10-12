#!/usr/bin/bash

echo --- creating data directory ---
mkdir -p data && cd data

echo --- downloading images ---
wget https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip
if [ $? -ne 0 ]; then
    echo "Failed to download training data!"
    exit 1
fi

echo --- downloading metadata ---
wget https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth.csv
if [ $? -ne 0 ]; then
    echo "Failed to download metadata!"
    exit 1
fi

echo --- unzipping data ---
zip --version &> /dev/null
if [ $? -ne 0 ]; then
    echo "zip/unzip not found, installing..."
    sudo apt update
    sudo apt install -y zip
    if [ $? -ne 0 ]; then
        echo "Failed to install zip/unzip!"
        exit 1
    fi
fi

unzip ISIC_2020_Training_JPEG.zip
if [ $? -ne 0 ]; then
    echo "Failed to unzip training data!"
    exit 1
fi

mkdir images
mv train/* images
rmdir train

rm ISIC_2020_Training_JPEG.zip
cd ..
echo --- data downloaded and extracted successfully ---