import math
import os
from typing import List, Union

import torch
from torch import Tensor, nn
from transformers import CLIPVisionConfig, CLIPVisionModel

import math
import os
from typing import List, Union

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer


class MultiEncoder(nn.Module):
    def __init__(
        self,
        modelpath: str,
        finetune: bool = False,
        last_hidden_state: bool = True,
        unit_scale: bool = False,
        **kwargs,
    ) -> None:

        super().__init__()

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)
        self.clip_model = AutoModel.from_pretrained(modelpath)
        self.unit_scale = unit_scale

        # Don't train the model
        if not finetune:
            self.clip_model.training = False
            for p in self.clip_model.parameters():
                p.requires_grad = False

        # Then configure the model
        self.max_length = self.tokenizer.model_max_length
        if "clip" in modelpath:
            self.vision_encoded_dim = self.clip_model.config.vision_config.hidden_size
            if last_hidden_state:
                self.name = "clip_hidden"
            else:
                self.name = "clip"
        else:
            raise ValueError(f"Model {modelpath} not supported")

    def forward(self, imgs, texts):
        if self.name in ["clip", "clip_hidden"]:
            text_inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            if text_input_ids.shape[-1] > 77:
                text_input_ids = text_input_ids[:, : 77]
        elif self.name == "bert":
            text_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)

        if self.name == "clip":
            # (batch_Size, text_encoded_dim)
            text_embeddings = self.text_model.get_text_features(
                text_input_ids.to(self.text_model.device)
            )
            # (batch_Size, 1, text_encoded_dim)
            text_embeddings = text_embeddings.unsqueeze(1)

            # Rescale the features to have unit variance
            if self.unit_scale:
                text_embeddings = text_embeddings * math.sqrt(text_embeddings.shape[-1])
            img_embeddings = self.clip_model.get_image_features(imgs)

        elif self.name == "clip_hidden":
            text_embeddings = self.clip_model.text_model(
                text_input_ids.to(self.clip_model.device)
            ).last_hidden_state
            img_embeddings = self.clip_model.vision_model(imgs, output_hidden_states=True).last_hidden_state

        else:
            raise NotImplementedError(f"Model {self.name} not implemented")

        return img_embeddings, text_embeddings
