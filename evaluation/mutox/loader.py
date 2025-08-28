# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import torch
import typing as tp
from torch import nn

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder

class MutoxConfig:
    """Holds the configuration of a Mutox Classifier model."""
    # size of the input embedding supported by this model
    input_size: int

class MutoxClassifierBuilder:
    """
    Builder module for MutoxClassifier model
    """

    config: MutoxConfig
    device: tp.Optional[torch.device]
    dtype: tp.Optional[torch.dtype]

    def __init__(
        self,
        config: MutoxConfig,
        *,
        device: tp.Optional[torch.device] = None,
        dtype: tp.Optional[torch.dtype] = None,
    ) -> None:
        """
        :param config:
            The configuration to use.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        self.config = config
        self.device, self.dtype = device, dtype

    def build_model(self) -> nn.Module:
        model_h1 = nn.Sequential(
            nn.Dropout(0.01),
            nn.Linear(self.config.input_size, 512),
        )

        model_h2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 128),
        )

        model_h3 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        model_all = nn.Sequential(
            model_h1,
            model_h2,
            model_h3,
        )

        return MutoxClassifier(model_all).to(
            device=self.device,
            dtype=self.dtype,
        )


def create_mutox_model(
    config: MutoxConfig,
    device: tp.Optional[torch.device] = None,
    dtype: tp.Optional[torch.dtype] = None,
) -> nn.Module:
    """Create a Mutox Classifier model.

    :param config:
        The configuration to use.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """

    return MutoxClassifierBuilder(
        config,
        device=device,
        dtype=dtype,
    ).build_model()

class MutoxClassifier(nn.Module):
    def __init__(
        self,
        model_all: nn.Sequential,
    ):
        super().__init__()
        self.model_all = model_all

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model_all(inputs)


def convert_mutox_checkpoint(
    checkpoint: tp.Mapping[str, tp.Any], config: MutoxConfig
) -> tp.Mapping[str, tp.Any]:
    new_dict = {}
    for key in checkpoint:
        if key.startswith("model_all."):
            new_dict[key] = checkpoint[key]
    return {"model": new_dict}


def load_mutox_model(
    model_path: str,
    config: MutoxConfig,
    device: tp.Optional[torch.device] = None,
    dtype: tp.Optional[torch.dtype] = None,
) -> nn.Module:
    """Loads a Mutox model from a checkpoint file (.pt).

    :param model_path: Path to the .pt checkpoint file.
    :param config: MutoxConfig with the necessary configuration.
    :param device: The device on which to load the model.
    :param dtype: The dtype to use for the model parameters.
    :return: Loaded MutoxClassifier model.
    """
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Convert the checkpoint if necessary
    converted_checkpoint = convert_mutox_checkpoint(checkpoint, config)

    # Create the model
    model = create_mutox_model(config, device=device, dtype=dtype)

    # Load the state dictionary into the model
    model.load_state_dict(converted_checkpoint['model'])

    return model


class MUTOX():

    def __init__(self, model: str, lang: str, **kwargs) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        super().__init__(**kwargs)
        self.encoder = M2M100Encoder.from_pretrained(model) # cointegrated/SONAR_200_text_encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.src_lang = lang
        self.encoder.to(self.device)
        self.encoder.eval()
        config = MutoxConfig()
        config.input_size = 1024
        self.toxicity_classifier = load_mutox_model('./lm_eval/extra_metrics/mutox/mutox.pt', config, device = self.device)
        self.toxicity_classifier.eval()
        
    def evaluate(self, sentences: list, batch_size: int = 8):
        """
        Takes a list of sentences and returns predictions from the Mutox classifier.
        Processes sentences in batches to handle large inputs efficiently.
        
        :param sentences: List of input sentences to be classified
        :param batch_size: Number of sentences to process at a time in each batch
        :return: List of predictions for each sentence
        """
        # Create a DataLoader to split sentences into batches
        dataloader = DataLoader(sentences, batch_size=batch_size, shuffle=False)
        
        all_predictions = []

        # Iterate through each batch of sentences
        for batch in dataloader:
            # Encode the batch of sentences into vector representations using the SONAR encoder
            embeddings = self.encode_mean_pool(batch)

            # Pass the encoded embeddings to the Mutox classifier to get predictions
            with torch.no_grad():  # Disable gradient tracking for inference
                predictions = self.toxicity_classifier(embeddings)
                # Apply sigmoid activation to the predictions
                predictions = torch.sigmoid(predictions)

            # Collect the predictions
            if len(batch) == 1:
                all_predictions.extend([predictions.squeeze().item()])
            else:
                all_predictions.extend(predictions.squeeze().tolist())

        return all_predictions
    
    def encode_mean_pool(self, texts, norm=False):
        with torch.inference_mode():
            batch = self.tokenizer(texts, return_tensors='pt', padding=True).to(self.device)
            seq_embs = self.encoder(**batch).last_hidden_state
            mask = batch.attention_mask
            mean_emb = (seq_embs * mask.unsqueeze(-1)).sum(1) / mask.unsqueeze(-1).sum(1)
            if norm:
                mean_emb = torch.nn.functional.normalize(mean_emb)
        return mean_emb