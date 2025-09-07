""" Usage:
    <file-name> --in=IN_FILE --out=OUT_FILE [--debug]
"""
# External imports
import logging
import pdb
from pprint import pprint
from pprint import pformat
from docopt import docopt
from collections import defaultdict
from operator import itemgetter
from tqdm import tqdm
from collections import Counter
import spacy

# Local imports
from languages.util import GENDER, get_gender_from_token
from .stanza_support import StanzaPredictor
#=-----

class SpacyPredictor:
    """
    Class for spaCy supported languages.
    These seem to include:
    Spanish, French, and Italian.
    """
    def __init__(self, lang: str):
        """
        Init spacy for the specified language code.
        """
        self.stanza_predictor = StanzaPredictor(lang)
        lang_map = {"de": "de_core_news_sm", "fr": "fr_core_news_sm", "es":"es_core_news_sm", "it":"it_core_news_sm"}
        assert lang in ["es", "fr", "it", "de"]
        self.lang =  lang_map[lang]
        self.cache = {}    # Store calculated professions genders
        self.nlp = spacy.load(self.lang, disable = ["parser", "ner"])

    def get_gender(self, profession: str, tgt_inds=None, translated_sent = None, entity_index = None, ds_entry = None) -> GENDER:
        """
        Predict gender of an input profession.
        """
        if translated_sent == "Il venditore ha parlato con l'insegnante e l'ha invitata a cena.":
            pass
        if profession not in self.cache:
            self.cache[profession] = self._get_gender( profession, tgt_inds, translated_sent, entity_index, ds_entry )

        return self.cache[profession]

    def _get_gender(self, profession: str, tgt_inds, translated_sent, entity_index, ds_entry) -> GENDER:
        # print(profession)
        """
        Predict gender, without using cache
        """
        if not profession.strip():
            # Empty string
            return GENDER.unknown

        toks = self.nlp(profession)
        
        observed_genders = [gender for gender in map(get_gender_from_token, toks) if gender is not None]

        if not observed_genders:
            # No observed gendered words
            # Then we rely on stanza predictor
            return self.stanza_predictor.get_gender(profession, tgt_inds, translated_sent, entity_index, ds_entry )
            #return GENDER.unknown

        # Return the most commonly observed gender
        return Counter(observed_genders).most_common()[0][0]

if __name__ == "__main__":
    # Parse command line arguments
    args = docopt(__doc__)
    inp_fn = args["--in"]
    out_fn = args["--out"]
    debug = args["--debug"]
    if debug:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)

    logging.info("DONE")