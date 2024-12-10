#! /bin/bash

# ======================= SSiT Baseline ==========================
./train.py --log_dir logs/logs_IDRiD --config_path configs/experiments/ssit_baseline --config_name IDRiD.yaml
./train.py --log_dir logs/logs_Messidor --config_path configs/experiments/ssit_baseline --config_name Messidor.yaml
./train.py --log_dir logs/logs_FGADR --config_path configs/experiments/ssit_baseline --config_name FGADR.yaml
./train.py --log_dir logs/logs_APTOS --config_path configs/experiments/ssit_baseline --config_name APTOS.yaml
./train.py --log_dir logs/logs_DDR --config_path configs/experiments/ssit_baseline --config_name DDR.yaml

# ======================= Basic Attention + ResNext ==========================
./train.py --log_dir logs/logs_IDRiD --config_path configs/experiments/architectures/ResNext --config_name IDRiD.yaml
./train.py --log_dir logs/logs_Messidor --config_path configs/experiments/architectures/ResNext --config_name Messidor.yaml
./train.py --log_dir logs/logs_FGADR --config_path configs/experiments/architectures/ResNext --config_name FGADR.yaml
./train.py --log_dir logs/logs_APTOS --config_path configs/experiments/architectures/ResNext --config_name APTOS.yaml
./train.py --log_dir logs/logs_DDR --config_path configs/experiments/architectures/ResNext --config_name DDR.yaml

# ======================= Basic Attention + EfficientNet ==========================
./train.py --log_dir logs/logs_IDRiD --config_path configs/experiments/architectures/EfficientNet --config_name IDRiD.yaml
./train.py --log_dir logs/logs_Messidor --config_path configs/experiments/architectures/EfficientNet --config_name Messidor.yaml
./train.py --log_dir logs/logs_FGADR --config_path configs/experiments/architectures/EfficientNet --config_name FGADR.yaml
./train.py --log_dir logs/logs_APTOS --config_path configs/experiments/architectures/EfficientNet --config_name APTOS.yaml
./train.py --log_dir logs/logs_DDR --config_path configs/experiments/architectures/EfficientNet --config_name DDR.yaml

# ======================= Basic Attention + ResNext ==========================
./train.py --log_dir logs/logs_IDRiD --config_path configs/experiments/architectures/Inception --config_name IDRiD.yaml
./train.py --log_dir logs/logs_Messidor --config_path configs/experiments/architectures/Inception --config_name Messidor.yaml
./train.py --log_dir logs/logs_FGADR --config_path configs/experiments/architectures/Inception --config_name FGADR.yaml
./train.py --log_dir logs/logs_APTOS --config_path configs/experiments/architectures/Inception --config_name APTOS.yaml
./train.py --log_dir logs/logs_DDR --config_path configs/experiments/architectures/Inception --config_name DDR.yaml

# ======================= Updated Attention ==========================
