#!/usr/bin/bash

echo --- creating data directory ---
mkdir -p data && cd data

echo --- downloading training data ---
wget https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip
if [ $? -ne 0 ]; then
    echo "Failed to download training data!"
    exit 1
fi

echo --- downloading test data ---
wget https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Test_JPEG.zip
if [ $? -ne 0 ]; then
    echo "Failed to download test data!"
    exit 1
fi

echo --- unzipping data ---
sudo apt install -y zip

unzip ISIC_2020_Training_JPEG.zip
if [ $? -ne 0 ]; then
    echo "Failed to unzip training data!"
    exit 1
fi

unzip ISIC_2020_Test_JPEG.zip
if [ $? -ne 0 ]; then
    echo "Failed to unzip test data!"
    exit 1
fi

rm ISIC_2020_Training_JPEG.zip ISIC_2020_Test_JPEG.zip

echo --- data downloaded and extracted successfully ---