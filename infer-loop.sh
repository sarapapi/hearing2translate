#!/bin/bash

# This script prints commands that run inference on selected datasets.
# It's safe for multiprocessing. Every infer.py process is locked, the lock is checked
# by others. Successful infer process is marked ok with `touch $out.ok`, so it can be
# easily inspected.

# Inspect by bare eyes what processes need to be run:
# ./infer-loop.sh

# Run the processes:
# ./infer-loop.sh | bash -v

# Multiprocessing, e.g.:
# ./infer-loop.sh | bash -v & ./infer-loop.sh | bash -v


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

cmd() {
	echo "if mkdir $out.lock ; then H2T_DATADIR=$H2T_DATADIR $HFTOKEN $p3""python3 infer.py --model $model --in-modality speech --in-file $inf --out-file $out $asr 2>&1 | tee $out.err && touch $out.ok; rm -rf $out.lock ; fi"
}



# add or remove the ones you want to run:
for dataset in winoST fleurs commonAccent ; do
#for dataset in acl6060 ; do
	# same with models
	for model in whisper canary-v2 seamlessm4t ; do 
		mkdir -p outputs/$model/$dataset/
		# translation:
		asr=""
		for inf in manifests/$dataset/*.jsonl ; do 
			b=$(basename $inf)
			langpair=${b/.jsonl/}
			# nl is not supported but some data are there
			[[ $langpair = *-nl ]] && continue
			# whisper is not from en, only into en
			[[ $model = whisper ]] && [[ ! $langpair = *-en ]] && continue
			out=outputs/$model/$dataset/$b
			# filters out successful (ok) or running (locked)
			if [ ! -f $out.ok ] && [ ! -d $out.lock ]; then
				cmd
			fi
		done
		outdir=outputs/$model""_asr/$dataset/
		mkdir -p $outdir
		# ASR:
		asr="--asr"
		for inf in manifests/$dataset/*.jsonl ; do 
			b=$(basename $inf)
			langpair=${b/.jsonl/}
			# nl is not supported but some data are there
       			[[ $langpair = *-nl ]] && continue
			# whisper is not from en, only into en
			out=$outdir/$b
			# filters out successful (ok) or running (locked)
			if [ ! -f $out.ok ] && [ ! -d $out.lock ]; then
				cmd
			fi
		done
	done
done
