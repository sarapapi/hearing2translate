""" Usage:
    <file-name> --dsp=DATASET_FILE_PATH --ds=DATASET_NAME --bi=IN_FILE --align=ALIGN_FILE --lang=LANG  --out=OUT_FILE --morph=MORPH_FN --ds_name=DS_NAME --sys_name=SYS_NAME --batch_size=BATCH_SIZE [--debug]
"""
# External imports
import logging
from pprint import pprint
from pprint import pformat
from docopt import docopt
from collections import defaultdict, Counter
from operator import ge, itemgetter
from tqdm import tqdm
from typing import List
import csv
import json
from multiprocessing import Process
import multiprocessing
import regex as re
from evaluate import evaluate_bias, evaluate_correctness
# Local imports
from languages.spacy_support import SpacyPredictor
from languages.gendered_article import GenderedArticlePredictor, \
    get_german_determiners, GERMAN_EXCEPTION
from evaluate import evaluate_bias
from languages.stanza_support import StanzaPredictor
#=-----

LANGAUGE_PREDICTOR= {
    "es": lambda: SpacyPredictor("es"),
    "fr": lambda: SpacyPredictor("fr"),
    "it": lambda: SpacyPredictor("it"),
    "de": lambda: GenderedArticlePredictor("de", get_german_determiners, GERMAN_EXCEPTION),
}

def get_src_indices(instance: List[str]) -> List[int]:
    """
    (English)
    Determine a list of source side indices pertaining to a
    given instance in the dataset.
    """
    _, src_word_ind, sent = instance[: 3]
    src_word_ind = int(src_word_ind)
    sent_tok = sent.split(" ")
    if (src_word_ind > 0) and (sent_tok[src_word_ind - 1].lower() in ["the", "an", "a"]):
        src_indices = [src_word_ind -1]
    else:
        src_indices = []
    src_indices.append(src_word_ind)

    return src_indices

def get_translated_professions(alignment_fn: str, ds: List[List[str]], bitext: List[List[str]]) -> List[str]:
    """
    (Language independent)
    Load alignments from file and return the translated profession according to
    source indices.
    """
    # Load files and data structures
    ds_src_sents = list(map(itemgetter(2), ds))
    bitext_src_sents = [src_sent for ind, (src_sent, tgt_sent) in bitext]

    # Sanity checks
    assert len(ds) == len(bitext)
    mismatched = [ind for (ind, (ds_src_sent, bitext_src_sent)) in enumerate(zip(ds_src_sents, bitext_src_sents))
                  if ds_src_sent != bitext_src_sent]
    if len(mismatched) != 0:
        raise AssertionError

    bitext = [(ind, (src_sent.split(), tgt_sent.split()))
              for ind, (src_sent, tgt_sent) in bitext]

    src_indices = list(map(get_src_indices, ds))

    full_alignments = []
    for line in open(alignment_fn):
        cur_align = defaultdict(list)
        for word in line.split():
            src, tgt = word.split("-")
            cur_align[int(src)].append(int(tgt))
        full_alignments.append(cur_align)

    bitext_inds = [ind for ind, _ in bitext]

    alignments = []
    for ind in bitext_inds:
        alignments.append(full_alignments[ind])


    assert len(bitext) == len(alignments)
    assert len(src_indices) == len(alignments)

    translated_professions = []
    target_indices = []
    c=0
    for (_, (src_sent, tgt_sent)), alignment, cur_indices in tqdm(zip(bitext, alignments, src_indices)):
        
        cur_tgt_inds = ([cur_tgt_ind
                         for src_ind in cur_indices
                         for cur_tgt_ind in alignment[src_ind]])

        try:
            cur_translated_profession = " ".join([tgt_sent[cur_tgt_ind].replace(',', '')
                                                for cur_tgt_ind in cur_tgt_inds])
        except:
            cur_translated_profession = ""
            c+=1
        
        target_indices.append(cur_tgt_inds)
        translated_professions.append(cur_translated_profession)
    logging.info(c)
    return translated_professions, target_indices

"""
def output_predictions(src_prof, source_sentences, orig_genders, trans_prof, target_sentences, gender_predictions, out_fn):
    
    #Write gender predictions to output file, for comparison with human judgments.
    
    assert(len(list(target_sentences)) == len(list(gender_predictions)))
    print(len(target_sentences))
    with open(out_fn, "w", encoding = "utf8") as fout:
        writer = csv.writer(fout, delimiter=",")
        writer.writerow(["Orig Prof", "Source Sentence", "Orig Gender", "Trans Prof", "Target Sentence", "Predicted gender"])
        for orig_prof, src_sent, orig_gen, trans_prof, tgt_sent, gender in zip(src_prof, source_sentences, orig_genders, trans_prof, target_sentences, gender_predictions):
            writer.writerow([orig_prof, src_sent, orig_gen, trans_prof, tgt_sent, str(gender).split(".")[1]])
"""

def output_predictions(src_prof, source_sentences, orig_genders, trans_prof, target_sentences, gender_predictions, out_fn):
    """
    Write gender predictions to output file in JSONL format,
    for comparison with human judgments.
    """
    assert(len(list(target_sentences)) == len(list(gender_predictions)))
    print(len(target_sentences))
    with open(out_fn, "w", encoding="utf8") as fout:
        for orig_prof, src_sent, orig_gen, trans_prof, tgt_sent, gender in zip(
            src_prof, source_sentences, orig_genders, trans_prof, target_sentences, gender_predictions
        ):
            record = {
                "src_ref": src_sent,
                "output": tgt_sent,
                "metrics" : {
                    "orig_prof": orig_prof,
                    "orig_gender": orig_gen,
                    "trans_prof": trans_prof,
                    "pred_gender": str(gender).split(".")[1]
                }
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

"""
def align_bitext_to_ds(bitext, ds):
    #Return a subset of bitext that's aligned to ds.
    c=0
    bitext_dict = dict([((ind,src.strip()), (ind, tgt.strip())) for ind, (src, tgt) in enumerate(bitext)])
    new_bitext = []
    for i, entry in enumerate(ds):
        en_sent = entry[2]
        en_sent_mod = re.sub(r'""', '"', en_sent)
        if en_sent_mod.startswith('"'):
          en_sent_mod = en_sent_mod[1:]
        if en_sent_mod.endswith('"'):
          en_sent_mod = en_sent_mod[:-1]
        try:
            ind, tgt_sent = bitext_dict[(i, en_sent_mod.strip())]
        except Exception as e:
            ind = -1
            tgt_sent = ""
            c+=1
        new_bitext.append((ind, (en_sent, tgt_sent)))
    print(c)
    logging.info(f"Length new bitext: { len(new_bitext) }" )
    return new_bitext
"""

def align_bitext_to_ds(bitext, ds):
    """
    Return a subset of bitext that's aligned to ds.
    """
    bitext_dict = dict([(src, (ind, tgt)) for ind, (src, tgt) in enumerate(bitext)])
    new_bitext = []
    for entry in ds:
        en_sent = entry[2]
        ind, tgt_sent = bitext_dict[en_sent]
        new_bitext.append((ind, (en_sent, tgt_sent)))

    logging.info(f"Length new bitext: { len(new_bitext) }" )
    return new_bitext

def predict_gender(lang, morph_fn, translated_profs, target_sentences, tgt_inds, ds, return_dict, batch):
    gender_predictor = LANGAUGE_PREDICTOR[lang]() if morph_fn!="stanza" else StanzaPredictor(lang)

    gender_predictions = [gender_predictor.get_gender(prof, tgt_ind, translated_sent, entity_index, ds_entry)
                          for prof, translated_sent, tgt_ind, entity_index, ds_entry
                          in tqdm(zip(translated_profs,
                                      target_sentences,
                                      tgt_inds,
                                      map(lambda ls:min(ls, default = -1), tgt_inds),
                                      ds))]
    
    return_dict[batch] = gender_predictions

def chunk(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

if __name__ == "__main__":
    # Parse command line arguments
    args = docopt(__doc__)
    ds_fn = args["--dsp"]
    bi_fn = args["--bi"]
    align_fn = args["--align"]
    lang = args["--lang"]
    dataset = args["--ds"]
    out_fn = args["--out"]
    morph_fn = args["--morph"]
    ds_name = args["--ds_name"]
    system_name = args["--sys_name"]
    n = int(args["--batch_size"])
    debug = args["--debug"]
    if debug:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)
    
    
    trans_fn = f"{system_name}_{ds_name}"

    eval_out_fn = f"{out_fn}/eval.{ds_name}.jsonl"
    out_fn_csv  = f"{out_fn}/pred.{ds_name}.jsonl"

    gender_predictor = LANGAUGE_PREDICTOR[lang]() if morph_fn!="stanza" else StanzaPredictor(lang)
    ds = [line.strip().split("\t") for line in open(ds_fn, encoding = "utf8")]
    full_bitext = [line.strip().split(" ||| ") for line in open(bi_fn, encoding = "utf8")]
    logging.info(f"LENGTH DS: {len(ds)}, LENGTH BITEXT: {len(full_bitext)}")
    bitext = align_bitext_to_ds(full_bitext, ds)

    #logging.info(bitext)

    translated_profs, tgt_inds = get_translated_professions(align_fn, ds, bitext)
    
    #logging.info(translated_profs)

    comparison_result = len(translated_profs) == len(tgt_inds)
    logging.info( f"N Translated professions == N Tgt indxs: {comparison_result}" )
    
    target_sentences, source_sentences = [tgt_sent for (ind, (src_sent, tgt_sent)) in bitext], [src_sent for (ind, (src_sent, tgt_sent)) in bitext]

    gender_predictions = [gender_predictor.get_gender(prof, tgt_ind, translated_sent, entity_index, ds_entry)
                          for prof, translated_sent, tgt_ind, entity_index, ds_entry
                          in tqdm(zip(translated_profs,
                                      target_sentences,
                                      tgt_inds,
                                      map(lambda ls:min(ls, default = -1), tgt_inds),
                                      ds))]

    orig_genders = list(map(itemgetter(0), ds))
    tgt_prof = list(map(itemgetter(3), ds))
    result = evaluate_bias(ds, gender_predictions)

    # Output predictions
    output_predictions(tgt_prof, source_sentences, orig_genders, translated_profs, target_sentences, gender_predictions, out_fn_csv)

    #with open(eval_out_fn, "a", encoding = "utf8") as fout:
    #    writer = csv.writer(fout, delimiter=",")
    #    writer.writerow([lang, trans_fn, result["acc"], result["f1_male"],\
    #        result["f1_female"], result["del_g"], result["unk_male"], result["unk_female"],\
    #            result["unk_neutral"] ])

    with open(eval_out_fn, "a", encoding="utf8") as fout:
        record = {
            "lang": lang,
            "trans_fn": trans_fn,
            "acc": result["acc"],
            "f1_male": result["f1_male"],
            "f1_female": result["f1_female"],
            "del_g": result["del_g"],
            "unk_male": result["unk_male"],
            "unk_female": result["unk_female"],
            "unk_neutral": result["unk_neutral"]
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    logging.info("DONE")