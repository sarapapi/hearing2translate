from detoxify import Detoxify

class BaseDetoxify():
    def __init__(self, model: str = "multilingual"):
        """
        Initializes the Detoxify model with the specified variant.
        
        :param model: The Detoxify model variant to use. Default is "original".
        """
        self.model = Detoxify(model)
    
    def evaluate(
        self, texts: list[str], batch_size: int = 8
    ) -> dict:
        """
        Evaluates the given texts for toxicity using the Detoxify model.

        :param texts: List of text inputs to evaluate.
        :param batch_size: Number of texts to process in a batch.
        :return: A dictionary containing the aggregated and per-text toxicity scores.
        """
        # Split texts into batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        segment_scores = []

        for batch in batches:
            batch_predictions = self.model.predict(batch)
            # Keep only the toxicity label for each text
            segment_scores.extend(batch_predictions["toxicity"])
        
        return segment_scores