#!/usr/bin/bash

echo --- creating conda environment ---
conda env create -f utility/env.yaml
if [ $? -ne 0 ]; then
    echo "Failed to create conda environment!"
    exit 1
fi
echo --- conda environment created successfully ---