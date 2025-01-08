import logging

import torch
import torch.nn.utils.rnn as rnn_utils
from torch import nn


logger = logging.getLogger(__name__)  # Get the logger for this module


class MLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # BertPredictionHeadTransform
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.GELU()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # BertLMPredictionHead
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor, attention_mask=None) -> torch.Tensor:
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.LayerNorm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x


class BaseRNN(nn.Module):
    """
    A base RNN module that can be used as a 'pooling' mechanism.
    If exposure is used, we adjust the classifier to accept
    (hidden_size * {1 or 2}) + exposure_dim as input.
    """

    def __init__(self, config, rnn_type, exposure_dim=0) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.bidirectional = config.to_dict().get("bidirectional", False)
        self.exposure_dim = exposure_dim

        self.rnn = rnn_type(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        # Adjust the input size of the classifier based on the bidirectionality + exposure
        base_rnn_output_size = self.hidden_size * (2 if self.bidirectional else 1)
        classifier_input_size = base_rnn_output_size + self.exposure_dim
        self.classifier = create_classifier(config, classifier_input_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        exposure: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Runs the hidden_states through the RNN and returns a single logit.
        If `exposure` is provided, it is concatenated to the final representation.
        """
        if self.exposure_dim > 0:
            if exposure is None:
                raise ValueError("Exposure is required for this model.")
        lengths = attention_mask.sum(dim=1).cpu()
        packed = rnn_utils.pack_padded_sequence(
            hidden_states, lengths, batch_first=True, enforce_sorted=False
        )

        output, _ = self.rnn(packed)
        output, _ = rnn_utils.pad_packed_sequence(output, batch_first=True)

        # Index of the last valid (non-padded) token per sequence
        last_sequence_idx = lengths - 1

        # Forward pass last hidden output
        forward_output = output[
            torch.arange(output.shape[0]), last_sequence_idx, : self.hidden_size
        ]

        # If bidirectional, also get the backward pass (first hidden output from the backward direction)
        if self.bidirectional:
            backward_output = output[:, 0, self.hidden_size :]
            x = torch.cat((forward_output, backward_output), dim=-1)
        else:
            x = forward_output

        # Optionally concatenate exposure
        if self.exposure_dim > 0:
            x = torch.cat([x, exposure.unsqueeze(-1)], dim=-1)

        logits = self.classifier(x)
        return logits


class FineTuneHead(nn.Module):
    """
    A unified FineTuneHead that:
     - Can perform CLS pooling
     - Or wrap an RNN-based pooling (GRU or LSTM) using BaseRNN
     - Optionally extends the classifier with an extra hidden layer (if config.extend_head is defined)
     - Optionally incorporates an exposure value (if provided) before the final classification
    """

    def __init__(self, config):
        super().__init__()

        # Decide if we will handle exposure dimension
        # (e.g., if your dataset always has a single floating number for exposure)
        # Set `exposure_dim=1` if you plan on always passing exposure to forward.
        # Otherwise, keep it at 0 (and handle None-checks in forward).
        self.exposure_dim = 1 if config.to_dict().get("use_exposure", False) else 0

        # Pool type
        self.pool_type = config.pool_type.lower()

        # Set up the pooling layer
        if self.pool_type in ["gru", "lstm"]:
            rnn_type = nn.GRU if self.pool_type == "gru" else nn.LSTM
            self.pool = BaseRNN(config, rnn_type, exposure_dim=self.exposure_dim)
            self.classifier = None
        elif self.pool_type == "cls":
            self.pool = self.pool_cls
        else:
            logger.warning(
                f"Unrecognized pool_type: {self.pool_type}. "
                "Defaulting to CLS pooling."
            )
            self.pool_type = "cls"
            self.pool = self.pool_cls

            classifier_input_size = config.hidden_size + self.exposure_dim
            self.classifier = create_classifier(config, classifier_input_size)

        logger.info(f"Using {self.pool_type} pooling for classification.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        exposure: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        - Pools the hidden_states depending on self.pool_type.
        - If RNN-based, relies on BaseRNN to handle classification (and exposure).
        - Otherwise, concatenates exposure (if use_exposure is True) before the final linear layer.
        """
        if self.exposure_dim > 0:
            if exposure is None:
                raise ValueError("Exposure is required for this model.")

        if self.pool_type in ["gru", "lstm"]:
            # BaseRNN handles everything
            return self.pool(
                hidden_states, attention_mask=attention_mask, exposure=exposure
            )
        else:
            # CLS or MEAN
            pooled = self.pool(hidden_states, attention_mask=attention_mask)

            # If exposure is used, concatenate it
            if self.exposure_dim > 0:
                pooled = torch.cat(
                    [pooled, exposure], dim=-1
                )  # shape: [batch, hidden_size + exposure_dim]

            logits = self.classifier(pooled)
            return logits

    def pool_cls(self, x, attention_mask=None):
        """
        CLS pooling just takes the [CLS] token (index 0).
        x shape: [batch, seq_len, hidden_size]
        """
        return x[:, 0]


def create_classifier(config, input_size):
    """Create classifier based on config."""
    if config.to_dict().get("classifier", None) is not None:
        if config.classifier == "big":
            return BigHead(input_size)
    return StandardHead(input_size)


class BigHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.classifier_hidden = 128
        self.classifier = nn.Sequential(
            nn.Linear(input_size, self.classifier_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.classifier_hidden, 1),
        )

    def forward(self, x):
        return self.classifier(x)


class StandardHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.classifier = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.classifier(x)
