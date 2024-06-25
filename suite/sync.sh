#! /usr/bin/env bash

rsync -r ./ narval:~/files/research/summarization/ -i -v --exclude="adapters*" --exclude="summarization_job_slurm*" --exclude=".env*"
