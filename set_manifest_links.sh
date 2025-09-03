#!/bin/bash
#
# WIP
#This script automatically sets links dataset/audio path folders that are expected from the "src_audio" key in .jsonl
#for HuggignFace datasets
#
set -e

. .env #Get $H2T_DATADIR

dataset="byan/cs-fleurs"
dataset_id="cs_fleurs"

if huggingface-cli scan-cache | grep -q $dataset; then
    echo "Dataset is cached"
    hf_hub_dataset_path=$(huggingface-cli scan-cache | grep $dataset | awk -F " {2,}" '{print $7}')
    #We asume that we have used the dataset uses the main branch. This should be changed, of course, per dataset if this is not the case
    dataset_folder=$hf_hub_dataset_path/snapshots/$(cat $hf_hub_dataset_path/refs/main)
    ln -f -s $dataset_folder $H2T_DATADIR/$dataset_id/data
    echo "Link has been created for $dataset: $dataset_folder -> $H2T_DATADIR/$dataset_id/data"
else
    echo "Dataset not cached"
fi
