#!/bin/bash

# For each dataset and lang dir, prints number of lines of inputs of each model.
# It's useful for investigation, which processes went smoothly/are running/were running but failed.  

# Usage:
# run
#  ./check-len.sh
# then read the outputs and check the outliers, like lower number of lines of one model than the others.

# like the 5 lines below is weird:

#   600 canary-v2_asr/commonAccent/es-en.jsonl
#   600 canary-v2/commonAccent/es-en.jsonl
#   600 seamlessm4t_asr/commonAccent/es-en.jsonl
#   600 seamlessm4t/commonAccent/es-en.jsonl
#     5 whisper_asr/commonAccent/es-en.jsonl
#   600 whisper/commonAccent/es-en.jsonl


# all datasets:
datasets=$(find -name \*jsonl | sed -r 's@/@ @g' | cut -f 3 -d' ' | sort -u)
#datasets="acl6060-long acl6060-short commonAccent fleurs winoST"
#datasets="commonAccent fleurs"
#datasets="mandi mcif-long mcif-short covost2"
#datasets="cs_fleurs"
#datasets=" wmt mexpresso cs-dialogue libristutter"
datasets="mexpresso cs-dialogue libristutter europarl_st noisy_fleurs_ambient noisy_fleurs_babble emotiontalk"
echo $datasets

my_models="{canary-v2,canary-v2_asr,seamlessm4t,seamlessm4t_asr,whisper,whisper_asr}"


for d in $datasets ; do
	echo Dataset: $d
	echo ==============
	echo

	ref_dir=../manifests/$d/

	# all src-tgt.jsonl for that datasets:
	jsonl=$(find */$d/ -name \*jsonl | sed -r 's@/@ @g' | cut -f 3 -d' ' | sort -u)
	for j in $jsonl ; do
		ref_len=$(wc -l < $ref_dir/$j)
		echo "	$j :"
		echo

		# prints number of lines in each together
		for i in {canary-v2,canary-v2_asr,seamlessm4t,seamlessm4t_asr,whisper,whisper_asr}/$d/$j ; do
			if [ -f $i ]; then
				len=$(wc -l < $i)
				if [ $ref_len -eq $len ]; then
					echo ok $ref_len $len $i
					git add $i
				else
					echo ERROR $ref_len $len $i
				fi
			else
				echo NOT-existing $i
			fi
		done
	done
	echo
done
