#! /usr/bin/env bash

delay1=400
delay2=400
delay3=600

echo "Memcheck > Memory checker started at: $(date)"


sleep $delay1
echo "Memcheck > First1 check: $(date)"
# mkdir "$2" -p
echo "1: Seconds ${delay1}" > "$2"
echo "1: Time $(date)" > "$2"
nvidia-smi >> "$2"

sleep $delay2
echo "Memcheck > Second2 check: $(date)"
echo "2: Seconds ${delay2}" > "$2"
echo "2: Time $(date)" > "$2"
nvidia-smi >> "$2"

sleep $delay3
echo "Memcheck > Third3 check: $(date)"
echo "2: Seconds ${delay2}" > "$2"
echo "2: Time $(date)" > "$2"
nvidia-smi >> "$2"
