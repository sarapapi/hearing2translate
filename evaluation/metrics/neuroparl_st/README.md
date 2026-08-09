# USAGE

Metric calculations script is extracted from the Neuroparl-ST release.
Example of usage:

```bash
#Spacy models need to have been previously been downloaded.
for x in en_core_web_lg es_core_news_lg fr_core_news_lg it_core_news_lg; do python3 -m spacy download $x; done

SRC_CODE=en
TGT_CODE=es
SRCTGT="$SRC_CODE"_"$TGT_CODE"
MODEL=canary-v2
python3 evaluation/metrics/neuroparl_st/ne_terms_accuracy.py --input ./evaluation/output_evals/europarl_st/$MODEL/$SRCTGT/results.jsonl --tsv-ref ./evaluation/metrics/neuroparl_st/terms_ne_files/$SRCTGT/test.all.$TGT_CODE.iob --lang $TGT_CODE --save_dir ./evaluation/output_evals/neuroparl_st/$MODEL/$SRCTGT/
``