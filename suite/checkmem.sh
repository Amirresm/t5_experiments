#! /usr/bin/env bash

delay=$1

sleep $delay

nvidia-smi > "$2"
