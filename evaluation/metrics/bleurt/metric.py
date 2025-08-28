# Code adapted from Tower: An Open Multilingual Large Language Model for Translation-Related Tasks 
# (Duarte M. Alves et al., 2024) available at https://github.com/deep-spin/tower-eval/tree/main
import torch
from bleurt_pytorch import (
    BleurtConfig,
    BleurtForSequenceClassification,
    BleurtTokenizer,
)
from tqdm import tqdm
import torch

class BLEURT():
    def __init__(self, ck_name):

        self.config = BleurtConfig.from_pretrained(ck_name)
        self.model = BleurtForSequenceClassification.from_pretrained(ck_name)
        self.tokenizer = BleurtTokenizer.from_pretrained(ck_name)

        self.model.eval()
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            self.model = self.model.to("cuda")

    def evaluate(
        self, hypotheses: list, references: list, batch_size: int
    ):
        """
        Evaluate function receives the hypotheses and the references and returns a COMETResult object.

        :param hypotheses: List of the MT outputs (sentences).
        :param references: List of the reference sentences.
        :param sources: List of source sentences
        """
        segments_scores = []
        for i in tqdm(range(0, len(references), batch_size)):
            with torch.no_grad():
                batch_references = references[i : i + batch_size]
                batch_hypotheses = hypotheses[i : i + batch_size]
                inputs = self.tokenizer(
                    batch_references,
                    batch_hypotheses,
                    padding="longest",
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to("cuda")
                segments_scores.extend(self.model(**inputs).logits.flatten().tolist())
        system_score = sum(segments_scores) / len(segments_scores)

        result = {
                    "system_score": system_score,
                    "segments_scores": segments_scores
                }
        
        return result