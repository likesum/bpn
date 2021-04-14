#!/bin/bash

pip install gdown

# Download pre-trained models
gdown https://drive.google.com/uc\?id\=1_-AFCj3G5ISJovdprVHvyeGt59hH2iKb
unzip trained_models.zip
rm trained_models.zip