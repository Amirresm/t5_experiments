#! /usr/bin/env bash

rsync --progress -r ./ narval:~/files/research/summarization/ -i -v --exclude="adapters*" --exclude="summarization_job_slurm*" --exclude=".env*"
