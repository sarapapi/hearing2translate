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

for d in $datasets ; do
	echo Dataset: $d
	echo ==============
	echo

	# all src-tgt.jsonl for that datasets:
	jsonl=$(find */$d/ -name \*jsonl | sed -r 's@/@ @g' | cut -f 3 -d' ' | sort -u)
	for j in $jsonl ; do
		echo "	$j :"
		echo

		# prints number of lines in each together
		wc -l */$d/$j
		echo
	done
	echo
done
