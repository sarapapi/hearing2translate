#!/usr/bin/python3
# Copyright 2021 FBK
# Modified by Jorge Iranzo in 2025 for the Hearing2Translate paper.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

# Script for the evaluation of accuracies on Named Entities and terminology.
# Details on the required annotated data can be found in the below paper.
# If using, please cite:
# M. Gaido et al., 2021. Is "moby dick" a Whale or a Bird? Named Entities and Terminology in Speech Translation,
# Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)

import argparse
import spacy
import pandas as pd
import json
from pathlib import Path

def ne_and_terms(fp):
    tokens = []
    full_entities = []
    while True:
        ln = fp.readline().strip()
        if ln == "":
            break
        items = ln.split("\t")
        if items[2] != "O":
            entity_type = items[2].split("-")[1]
            entity_pos = items[2].split("-")[0]
            tokens.append((items[1], entity_type))
            if entity_pos == "B":
                full_entities.append(([items[1]], entity_type))
            elif entity_pos == "I":
                full_entities[-1][0].append(items[1])
            else:
                raise ValueError("Unrecognized position {} in \"{}\"".format(entity_pos, ln))
    return tokens, full_entities


def full_entity_index(full_entity, hypothesis):
    tokens_to_match = len(full_entity)
    for i in range(len(hypothesis) - tokens_to_match):
        if hypothesis[i:i+tokens_to_match] == full_entity:
            return i
    return -1


def scores_by_type(in_f, tsv_reference, tokenizer):
    entity_items_scores = {}
    full_entities_scores = {}
    #va linea a linea leyendo en el fichero de salida y tokeniza la entrada.
    with open(in_f) as i_f, open(tsv_reference) as r_f:
        for i_line in i_f:
            reference_tokens, reference_entities = ne_and_terms(r_f)
            tokenized = [str(tok) for tok in tokenizer(i_line)]
            lowercase_tokenized = [tok.lower() for tok in tokenized]

            tokenized_clone = tokenized.copy()
            lowercase_tokenized_clone = lowercase_tokenized.copy()

            for token, entity_type in reference_tokens:
                if entity_type not in entity_items_scores:
                    entity_items_scores[entity_type] = {
                        "found": 0, "total": 0, "ci_found": 0}
                entity_items_scores[entity_type]["total"] += 1
                if token in tokenized:
                    tokenized.remove(token)
                    entity_items_scores[entity_type]["found"] += 1
                if token.lower() in lowercase_tokenized:
                    lowercase_tokenized.remove(token.lower())
                    entity_items_scores[entity_type]["ci_found"] += 1

            for entity, entity_type in reference_entities:
                if entity_type not in full_entities_scores:
                    full_entities_scores[entity_type] = {
                        "found": 0, "total": 0, "ci_found": 0}
                full_entities_scores[entity_type]["total"] += 1
                idx = full_entity_index(entity, tokenized_clone)
                if idx >= 0:
                    del tokenized_clone[idx:idx+len(entity)]
                    full_entities_scores[entity_type]["found"] += 1
                idx_lower = full_entity_index(
                    [t.lower() for t in entity], lowercase_tokenized_clone)
                if idx_lower >= 0:
                    del lowercase_tokenized_clone[idx:idx+len(entity)]
                    full_entities_scores[entity_type]["ci_found"] += 1

    return entity_items_scores, full_entities_scores


def print_scores(out_scores, score_type, save_dir, print_latex=False):
    categories = list(out_scores.keys())
    categories.sort()
    printed_scores = {}
    accuracy_ne_scores = 0
    accuracy_ne_len = 0
    accuracy_ne_ci_scores = 0
    for c in categories:
        printed_scores[c] = {}
        printed_scores[c]["Total"] = out_scores[c]["total"]
        printed_scores[c]["Found"] = out_scores[c]["found"]
        printed_scores[c]["Case Insensitive Found"] = out_scores[c]["ci_found"]
        printed_scores[c]["Accuracy"] = float(out_scores[c]["found"]) / out_scores[c]["total"]
        printed_scores[c]["Case Insensitive Accuracy"] =\
            float(out_scores[c]["ci_found"]) / out_scores[c]["total"]
        if c != "TERM":
            accuracy_ne_scores += out_scores[c]["found"]
            accuracy_ne_ci_scores += out_scores[c]["ci_found"]
            accuracy_ne_len += out_scores[c]["total"]
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "results_summary.jsonl", "w+") as out_json, open(path / "full_breakdown_table.tex", "w+") as out_tex:
        scores = {
                  "accuracy_ne": float(accuracy_ne_scores) / accuracy_ne_len if accuracy_ne_len != 0 else 0, 
                  "accuracy_term": float(out_scores["TERM"]["found"]) / out_scores["TERM"]["total"],
                  "accuracy_ci_ne": float(accuracy_ne_ci_scores) / accuracy_ne_len if accuracy_ne_len != 0 else 0, 
                  "accuracy_ci_term": float(out_scores["TERM"]["ci_found"]) / out_scores["TERM"]["total"]
                   }
        json.dump(scores, out_json)
        table = pd.DataFrame.from_dict(printed_scores, orient='index').to_latex()
        out_tex.write(table)
        if print_latex:
            print(table)
        print(f"Wrote results to {str(path)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str, metavar='FILE',
                        help='Input file to be used to compute accuracies (it must be detokenized).')
    parser.add_argument('--tsv-ref', required=True, type=str, metavar='REFERENCE',
                        help='TSV with NE and terms definition file.')
    parser.add_argument('--lang', required=True, type=str, metavar='LANG',
                        help='Target language.')
    parser.add_argument('--debug', required=False, action='store_true', default=False)
    parser.add_argument("--print-latex", required=False, action='store_true', default=False)
    parser.add_argument("--save_dir", required=True, type=str)

    args = parser.parse_args()

    LANG_MAP = {
        "en": "en_core_web_lg",
        "es": "es_core_news_lg",
        "fr": "fr_core_news_lg",
        "it": "it_core_news_lg"}

    #subprocess.run(f"python3 -m spacy download {LANG_MAP[args.lang]}", shell=True)
    nlp = spacy.load(LANG_MAP[args.lang], disable=['parser', 'ner'])

    items_scores, entities_scores = scores_by_type(args.input, args.tsv_ref, nlp)
    #print_scores(items_scores, "Items", print_latex=args.print_latex)
    print_scores(entities_scores, "Full Entities", args.save_dir, print_latex=args.print_latex)
