# Code adapted from Tower: An Open Multilingual Large Language Model for Translation-Related Tasks 
# (Duarte M. Alves et al., 2024) available at https://github.com/deep-spin/tower-eval/tree/main
from comet import download_model, load_from_checkpoint


class COMETKiwi():
    def __init__(self, model: str):

        model_path = download_model(model)
        self.model = load_from_checkpoint(model_path)
        self.model.eval()

    def make_samples(
        self, sources: list[str], hypotheses: list[str]
    ):
        samples = {"src": sources, "mt": hypotheses}
        samples = [dict(zip(samples, t)) for t in zip(*samples.values())]
        return samples

    def evaluate(
        self, hypotheses: list, sources: list, batch_size
    ):
        """
        Evaluate function receives the hypotheses and the references and returns a COMETResult object.

        :param hypotheses: List of the MT outputs (sentences).
        :param sources: List of source sentences
        """
        samples = self.make_samples(sources, hypotheses)

        outputs = self.model.predict(
            samples=samples,
            batch_size=batch_size
            )

        system_score, segments_scores = outputs.system_score, outputs.scores

        comet_result = {
                            "system_score": system_score,
                            "segments_scores": segments_scores,
                        }
        
        return comet_result