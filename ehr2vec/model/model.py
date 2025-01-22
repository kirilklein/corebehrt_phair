import logging

import torch
import torch.nn as nn
from transformers import ModernBertModel

from ehr2vec.embeddings.ehr import EhrEmbeddings
from ehr2vec.model.heads import FineTuneHead, MLMHead
from ehr2vec.model.loss import neg_partial_log_likelihood

logger = logging.getLogger(__name__)


class BertEHREncoder(ModernBertModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = EhrEmbeddings(config)

    def forward(self, batch: dict = None, inputs_embeds: torch.tensor = None, **kwargs):

        if inputs_embeds is not None:
            return super().forward(inputs_embeds=inputs_embeds, **kwargs)
        if batch is not None:
            # Extract necessary components from the batch
            input_ids = batch["concept"]
            token_type_ids = batch.get("segment", None)
            attention_mask = batch.get("attention_mask", None)
            present_keys = [
                k
                for k in ["age", "abspos", "position_ids", "dosage", "unit"]
                if k in batch
            ]
            position_ids = {key: batch.get(key) for key in present_keys}

            # Compute embeddings manually including token_type_ids and position_ids
            inputs_embeds = self.embeddings(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )

            # Call parent's forward with inputs_embeds and attention_mask
            return super().forward(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
            )
        else:
            raise ValueError("Either batch or inputs_embeds must be provided.")

    def freeze(self):
        """Freeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = False


class BertEHRModel(BertEHREncoder):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fct = nn.CrossEntropyLoss()
        self.cls = MLMHead(config)

    def forward(self, batch: dict = None, inputs_embeds: torch.tensor = None, **kwargs):
        outputs = super().forward(batch=batch, inputs_embeds=inputs_embeds, **kwargs)
        sequence_output = outputs[0]  # Last hidden state
        attention_mask = (
            batch["attention_mask"]
            if batch is not None
            else torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device).int()
        )
        logits = self.cls(sequence_output, attention_mask=attention_mask)

        # Calculate loss if target is provided
        loss = None
        if batch is not None and batch.get("target", None) is not None:
            loss = self.get_loss(logits, batch["target"])

        # Return a dictionary instead of modifying outputs
        return {
            "last_hidden_state": outputs.last_hidden_state,
            # "pooler_output": outputs.pooler_output,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
            "logits": logits,
            "loss": loss,
        }

    def get_loss(self, logits, labels):
        """Calculate loss for masked language model."""
        return self.loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))


class BertForFineTuning(BertEHREncoder):
    def __init__(self, config):
        super().__init__(config)
        if config.pos_weight:
            pos_weight = torch.tensor(config.pos_weight)
        else:
            pos_weight = None

        self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.cls = FineTuneHead(config)
        logger.info(f"Using {self.cls.__class__.__name__} as classifier.")

    def forward(self, batch: dict = None, inputs_embeds: torch.tensor = None, **kwargs):
        outputs = super().forward(batch=batch, inputs_embeds=inputs_embeds, **kwargs)
        sequence_output = outputs["last_hidden_state"]
        logits, patient_vector = self.cls(
            sequence_output,
            batch["attention_mask"],
            exposure=batch.get("exposure", None),
        )

        loss = None
        if batch.get("target", None) is not None:
            loss = self.get_loss(logits, batch["target"])
        return {
            "last_hidden_state": outputs["last_hidden_state"],
            # "pooler_output": outputs["pooler_output"],
            "hidden_states": outputs.get("hidden_states", None),
            "attentions": outputs.get("attentions", None),
            "logits": logits,
            "patient_vector": patient_vector,
            "loss": loss,
        }

    def get_loss(self, logits, labels, labels_mask=None):
        return self.loss_fct(logits.view(-1), labels.view(-1))


class BertForTime2Event(BertEHREncoder):
    def __init__(self, config):
        super().__init__(config)
        self.cls = FineTuneHead(config)
        self.loss_fct = neg_partial_log_likelihood
        logger.info(f"Using {self.cls.__class__.__name__} as model head")

    def forward(self, batch: dict, inputs_embeds: torch.tensor = None):
        outputs = super().forward(batch=batch, inputs_embeds=inputs_embeds)
        sequence_output = outputs["last_hidden_state"]
        logits, patient_vector = self.cls(sequence_output, batch["attention_mask"])

        loss = None
        if (batch.get("target", None) is not None) and (
            batch.get("time2event", None) is not None
        ):
            loss = self.get_loss(logits, batch["target"], batch["time2event"])

        return {
            "last_hidden_state": outputs["last_hidden_state"],
            # "pooler_output": outputs["pooler_output"],
            "hidden_states": outputs["hidden_states"],
            "attentions": outputs["attentions"],
            "logits": logits,
            "patient_vector": patient_vector,
            "loss": loss,
        }

    def get_loss(self, logits, labels, time2event):
        return self.loss_fct(logits.view(-1), labels.view(-1), time2event.view(-1))
