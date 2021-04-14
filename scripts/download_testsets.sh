#!/bin/bash

cd data
pip install gdown

# Download grayscale testset
gdown https://drive.google.com/uc\?id\=1UptBXV4f56wMDpS365ydhZkej6ABTFq1

# Download color testset
gdown https://drive.google.com/uc\?id\=1rXmauXa_AW8ZrNiD2QPrbmxcIOfsiONE
unzip color_testset.zip
rm color_testset.zip

cd ..