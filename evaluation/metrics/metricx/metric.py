# Code adapted from Tower: An Open Multilingual Large Language Model for Translation-Related Tasks 
# (Duarte M. Alves et al., 2024) available at https://github.com/deep-spin/tower-eval/tree/main
import torch
from .models import MT5ForRegression
from datasets import Dataset
from transformers import AutoTokenizer

class BaseMetricX():
    def __init__(self, tokenizer: str, model: str, **kwargs) -> None:
        if torch.cuda.is_available():
            # This refers to the first visible GPU
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        super().__init__(**kwargs)
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
                example["input"], max_length=1024, truncation=True, padding=False
            )

        def _remove_eos(example):
            example["input_ids"] = example["input_ids"][:-1]
            example["attention_mask"] = example["attention_mask"][:-1]
            return example

        samples = self.make_samples(
            sources=sources, hypotheses=hypotheses, references=references
        )
        ds = Dataset.from_list(samples)
        ds = ds.map(self._make_input)
        ds = ds.map(_tokenize)
        ds = ds.map(_remove_eos)
        ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask"],
            device=self.device,
            output_all_columns=True,
        )
        with torch.no_grad():
            predictions = [
                self.model(
                    sample["input_ids"], sample["attention_mask"]
                ).predictions.item()
                for sample in ds.iter(batch_size=1)
            ]
        metricx_result =  {
                "system_score": sum(predictions) / len(predictions),
                "segments_scores": predictions,
            }
        
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