#!/bin/bash
echo "Checking for GPU availability"
pip list | grep "pytorch-cuda"
nvidia-smi
echo "Torch CUDA: $(python3 -c "import torch ; print(torch.cuda.is_available())")"