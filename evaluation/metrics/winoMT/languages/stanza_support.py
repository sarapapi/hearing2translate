""" Usage:
    <file-name> --in=IN_FILE --out=OUT_FILE [--debug]
"""
# External imports
import logging
from docopt import docopt
from tqdm import tqdm
from collections import Counter
import stanza
import numpy as np
import operator

# Local imports
from languages.util import GENDER, get_gender_from_word_stanza
#=-----

DE_DETERMINERS = {"der": GENDER.male, "ein": GENDER.male, "dem": GENDER.male, #"den": GENDER.male, 
                  "einen": GENDER.male, "des": GENDER.male, "er": GENDER.male, "seiner": GENDER.male,
                  "ihn": GENDER.male, "seinen": GENDER.male, "ihm": GENDER.male, "ihren": GENDER.male,
                  "die": GENDER.female, "eine": GENDER.female, "einer": GENDER.female, "seinem": GENDER.male,
                  "ihrem": GENDER.male, "sein": GENDER.male,
                  "sie": GENDER.female, "seine": GENDER.female, "ihrer": GENDER.female, 
                  "ihr": GENDER.neutral, "ihre": GENDER.neutral, "das": GENDER.neutral,
                  "jemanden": GENDER.neutral} 

FR_DETERMINERS = {"le": GENDER.male, "la": GENDER.female, 'au':GENDER.male, 'du': GENDER.male, "un": GENDER.male, "une": GENDER.female} 

ES_DETERMINERS = {"el": GENDER.male, "la": GENDER.female, "un": GENDER.male, "una": GENDER.female} 

IT_DETERMINERS = {"il": GENDER.male, "lo": GENDER.male, "i": GENDER.male, "gli": GENDER.male, "uno": GENDER.male,
                  "la": GENDER.female, "le": GENDER.female, "una": GENDER.female}

class StanzaPredictor:

    def __init__(self, lang: str):
        
        self.lang =  lang
        self.cache = {}    # Store calculated professions genders
        self.nlp = stanza.Pipeline(lang)

    def fallback(self, words):
        if self.lang == "de":
            if any([word.endswith("in") for word in words]):
                return GENDER.female
            for word in words:
                if word.lower() in DE_DETERMINERS.keys():
                    return DE_DETERMINERS[word.lower()]

        if self.lang=="fr":
            for word in words:
                if word.lower() in FR_DETERMINERS.keys():
                    return FR_DETERMINERS[word.lower()]

        if self.lang=="es":
            for word in words:
                if word.lower() in ES_DETERMINERS.keys():
                    return ES_DETERMINERS[word.lower()]

        if self.lang=="it":
            for word in words:
                if word.lower() in IT_DETERMINERS.keys():
                    return IT_DETERMINERS[word.lower()]

            


    def get_gender(self, profession: str, tgt_inds, translated_sent, entity_index, ds_entry) -> GENDER:
        """
        Predict gender of an input profession.
        """
        if profession in self.cache:
            return self.cache[profession]
        if not profession.strip():
            # Empty string
            return GENDER.unknown

        words = np.array([word for sentence in self.nlp(translated_sent).sentences for word in sentence.words ])
        words = list(words[tgt_inds])
        profession_words = [word for word in profession.split()]
       
        observed_genders = [gender for gender in map(get_gender_from_word_stanza, words)
                                if gender is not None]
        
        if not observed_genders:
            # No observed gendered words - check fallback
            fallback = self.fallback(profession_words)
            if not fallback:
                return GENDER.unknown
            observed_genders = [fallback]

        # Return the most commonly observed gender
        self.cache[profession] = Counter(observed_genders).most_common()[0][0]
        return self.cache[profession]

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