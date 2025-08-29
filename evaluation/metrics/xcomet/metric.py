# Code adapted from Tower: An Open Multilingual Large Language Model for Translation-Related Tasks 
# (Duarte M. Alves et al., 2024) available at https://github.com/deep-spin/tower-eval/tree/main
from comet import download_model, load_from_checkpoint

class XCOMET():
    def __init__(self, model: str):

        model_path = download_model(model)
        self.model = load_from_checkpoint(model_path)
        self.model.eval()


    def make_samples(
        self, sources: list[str], hypotheses: list[str], references: list[str]
    ):
        samples = {"src": sources, "mt": hypotheses, "ref": references}
        samples = [dict(zip(samples, t)) for t in zip(*samples.values())]
        return samples

    def evaluate(
        self, hypotheses: list, references: list, sources: list, batch_size
    ):
        """
        Evaluate function receives the hypotheses and the references and returns a COMETResult object.

        :param hypotheses: List of the MT outputs (sentences).
        :param references: List of the reference sentences.
        :param sources: List of source sentences
        """
        samples = self.make_samples(sources, hypotheses, references)

        outputs = self.model.predict(
            samples=samples,
            batch_size=batch_size
            )

        system_score, segments_scores, error_spans = outputs.system_score, outputs.scores, outputs.metadata.error_spans

        comet_result = {
                            "system_score": system_score,
                            "segments_scores": segments_scores,
                            "error_spans": error_spans
                        }
                        
        return comet_result


class XCOMET_QE(XCOMET):

    def __init__(self, model):
        super().__init__(model=model)

    def make_samples(self, sources: list[str], hypotheses: list[str], references: list[str]):

        samples = {"src": sources, "mt": hypotheses}
        samples = [dict(zip(samples, t)) for t in zip(*samples.values())]
        
        return samples