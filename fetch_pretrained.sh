#! /bin/bash

OUTPUT_DIR="checkpoints"
SSIT_LINK="https://github.com/YijinHuang/SSiT/releases/download/v1/pretrained_vits_from_scratch.pt"
SSIT_IMAGENET_LINK="https://github.com/YijinHuang/SSiT/releases/download/v1/pretrained_vits_imagenet_initialized.pt"

mkdir -p "${OUTPUT_DIR}"


wget "$SSIT_LINK" -P "$OUTPUT_DIR"
wget "$SSIT_IMAGENET_LINK" -P "$OUTPUT_DIR"
