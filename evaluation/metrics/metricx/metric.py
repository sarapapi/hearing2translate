# Code adapted from Tower: An Open Multilingual Large Language Model for Translation-Related Tasks 
# (Duarte M. Alves et al., 2024) available at https://github.com/deep-spin/tower-eval/tree/main
import torch
from .models import MT5ForRegression
from datasets import Dataset
from transformers import AutoTokenizer
import logging

class BaseMetricX():
    def __init__(self, tokenizer: str, model: str, **kwargs) -> None:
        if torch.cuda.is_available():
            # This refers to the first visible GPU
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        super().__init__(**kwargs)
        self.max_input_length = 1536 # set to 1536 as we will use metricX24
        self.model = MT5ForRegression.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def make_samples(
        sources: list[str], hypotheses: list[str], references: list[str] = None
    ):
        pass

    @staticmethod
    def _make_input(example):
        pass

    def evaluate(
        self, sources: list, hypotheses: list, references: list
    ):
        """
        Evaluate function receives the hypotheses and the references and returns a COMETResult object.

        :param hypotheses: List of the MT outputs (sentences).
        :param references: List of the reference sentences.
        """

        def _tokenize(example):
            return self.tokenizer(
                example["input"], max_length=self.max_input_length, truncation=True, padding=False
            )

        def _remove_eos(example):
            example["input_ids"] = example["input_ids"][:-1]
            example["attention_mask"] = example["attention_mask"][:-1]
            return example
        
        samples = self.make_samples(
            sources=sources, hypotheses=hypotheses, references=references
        )

        N = len(samples)
        scores = []
        for i in range(0, N):
            if i % 50 == 0:
                logging.info(f"Predicting batch : {i}. Num samples: {N}")

            _prompts = self._make_input(samples[i])['input']

            tokens = self.tokenizer(
                _prompts,
                truncation=True,
                padding=True,
                max_length=1536,
                return_tensors="pt",
            )

            # remove eos token
            tokens["input_ids"] = tokens["input_ids"][:, :-1]
            tokens["attention_mask"] = tokens["attention_mask"][:, :-1]

            # move tokens to cuda device
            tokens["input_ids"] = tokens["input_ids"].to("cuda")
            tokens["attention_mask"] = tokens["attention_mask"].to("cuda")

            with torch.no_grad():
                outputs = self.model(**tokens)

            _scores = outputs.predictions.cpu().tolist()
            scores.extend(_scores)

        metricx_result = {
            "system_score": float(sum(scores) / max(len(scores), 1)),
            "segments_scores": scores,
        }

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        del self.model 
        return metricx_result


class RefMetricX(BaseMetricX):
    def __init__(self, tokenizer: str, model: str, **kwargs) -> None:
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)

    @staticmethod
    def make_samples(
        hypotheses: list[str], references: list[str], sources: list[str] = None
    ):
        return [
            {"hypothesis": h, "reference": r} for h, r in zip(hypotheses, references)
        ]

    @staticmethod
    def _make_input(example):
        example["input"] = (
            "candidate: "
            + example["hypothesis"]
            + " reference: "
            + example["reference"]
        )
        return example

class RefMetricX_24(BaseMetricX):
    def __init__(self, tokenizer: str, model: str, **kwargs) -> None:
        super().__init__(
            model=model, tokenizer=tokenizer, **kwargs
        )

    @staticmethod
    def make_samples(
        hypotheses: list[str], references: list[str], sources: list[str] = None
    ):
        return [
            {"hypothesis": h, "reference": r, "source": s}
            for h, r, s in zip(hypotheses, references, sources)
        ]

    @staticmethod
    def _make_input(example):
        example["input"] = (
            "source: "
            + example["source"]
            + " candidate: "
            + example["hypothesis"]
            + " reference: "
            + example["reference"]
        )
        return example

class QEMetricX(BaseMetricX):
    def __init__(self, tokenizer: str, model: str, **kwargs) -> None:
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)

    @staticmethod
    def make_samples(
        sources: list[str], hypotheses: list[str], references: list[str] = None
    ):
        return [{"hypothesis": h, "source": s} for h, s in zip(hypotheses, sources)]

    @staticmethod
    def _make_input(example):
        example["input"] = (
            "candidate: " + example["hypothesis"] + " source: " + example["source"]
        )
        return example

class QEMetricX_24(BaseMetricX):
    def __init__(self, tokenizer: str, model: str, **kwargs) -> None:
        super().__init__(
            model=model, tokenizer=tokenizer, **kwargs
        )

    @staticmethod
    def make_samples(
        sources: list[str], hypotheses: list[str], references: list[str] = None
    ):
        return [{"hypothesis": h, "source": s} for h, s in zip(hypotheses, sources)]

    @staticmethod
    def _make_input(example):
        example["input"] = (
            "source: " + example["source"] + " candidate: " + example["hypothesis"]
        )
        return example