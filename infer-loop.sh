#!/bin/bash

# this script prints commands that run inference on selected datasets

# Example usage:
# ./infer-loop.sh | bash -v

if [ -z "$H2T_DATADIR" ]; then
	# define your env var if not set from outside
	# (this is Dominik's default one)
	H2T_DATADIR=manifests
fi

# Dominik wants to use this p3 environment:
if [ $USER = machacek ]; then
	p3=p3/bin/
	HFTOKEN="HF_TOKEN=$(cat hftoken.txt)"
else # the others don't use any
	p3=""
	HFTOKEN=""
fi


# add or remove the ones you want to run:
for dataset in winoST fleurs ; do
	# same with models
	for model in whisper canary-v2 seamlessm4t ; do 
		mkdir -p outputs/$model/$dataset/
		for inf in manifests/$dataset/*.jsonl ; do 
			b=$(basename $inf)
			langpair=${b/.jsonl/}
			[[ $model = whisper ]] && [[ ! $langpair = *-en ]] && continue
			echo "H2T_DATADIR=$H2T_DATADIR $HFTOKEN $p3""python3 infer.py --model $model --in-modality speech --in-file $inf --out-file outputs/$model/$dataset/$(basename $inf)"
		done
	done
done
