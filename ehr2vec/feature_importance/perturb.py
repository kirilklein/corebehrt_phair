import torch
import logging
from ehr2vec.model.model import BertEHRModel
from ehr2vec.common.config import Config
from ehr2vec.embeddings.ehr import PerturbedEHREmbeddings

logger = logging.getLogger(__name__)


class PerturbationModel(torch.nn.Module):
    def __init__(self, bert_model: BertEHRModel, cfg: Config, concept_frequency=None):
        """Lambda determines how much the"""
        super().__init__()
        self.config = cfg

        self.bert_model = bert_model
        self.bert_model.freeze()
        self.noise_simulator = GaussianNoise(bert_model, cfg)

        self.lambda_ = self.config.get("lambda", 0.01)
        self.K = bert_model.config.hidden_size
        regularization_term = 1 / (self.K * self.lambda_)
        logger.info(f"Regularization term: {regularization_term}")
        self.register_buffer("regularization_term", torch.tensor(regularization_term))

        inverse_frequency = self.set_inverse_frequency(concept_frequency)

        self.register_buffer("sqrt_inverse_frequency", torch.sqrt(inverse_frequency))

        self.embeddings_perturb = PerturbedEHREmbeddings(self.bert_model.config)
        self.embeddings_perturb.set_parameters(self.bert_model.embeddings)
        self.embeddings_perturb.freeze()

    def set_inverse_frequency(self, concept_frequency):
        """Set the inverse frequency of the concepts to the sigmas. If not set, all sigmas are set to 1.0."""
        if concept_frequency is not None:
            if len(concept_frequency) != len(
                self.noise_simulator.thetas_embedding.weight
            ):
                raise ValueError(
                    "Concept frequency should have the same length as the sigmas."
                )
            return 1 / (concept_frequency + 1e-6)
        return torch.ones_like(self.noise_simulator.thetas_embedding.weight)

    def forward(self, batch: dict):
        original_output = self.bert_model(batch=batch, output_hidden_states=True)
        perturbed_embeddings = self.embeddings_perturb(batch, self.noise_simulator)
        perturbed_output = self.bert_model(
            batch, perturbed_embeddings, output_hidden_states=True
        )
        loss = self.perturbation_loss(original_output, perturbed_output, batch)
        outputs = ModelOutputs(
            logits=original_output.logits,
            perturbed_logits=perturbed_output.logits,
            loss=loss,
            hidden_states=original_output.hidden_states,
            perturbed_hidden_states=perturbed_output.hidden_states,
        )
        return outputs

    def freeze_bert(self):
        for param in self.bert_model.parameters():
            param.requires_grad = False

    def perturbation_loss(
        self, original_output: torch.Tensor, perturbed_output: torch.Tensor, batch: dict
    ) -> torch.Tensor:
        """
        Calculate the perturbation loss as presented in eq. 7 in the paper:
        Towards a deep and unified understanding of deep neural models in NLP.
        https://proceedings.mlr.press/v97/guan19a.html
        Calculate the perturbation loss, focusing on hidden states for a better alignment with mutual information principles.
        Args:
            original_output: Model output without perturbation
            perturbed_output: Model output with perturbation
            batch: Input batch, needed to access the correct sigmas
        """
        logits = original_output.logits
        perturbed_logits = perturbed_output.logits

        squared_diff = (logits - perturbed_logits) ** 2

        thetas = (
            self.noise_simulator.thetas_embedding.weight
        )  # more efficient than getting sigmas and then applying log

        # Regularization term on sigmas without harsh scaling
        first_term = -thetas.mean()

        # Normalize squared differences
        second_term = (
            self.regularization_term * squared_diff / (logits.std() + 1e-6)
        ).mean()
        return first_term + second_term

    def log(self, logger):
        log_string = "Sigmas:\n"
        sigmas = self.get_sigmas_weights()
        log_string += f"Min: {round(sigmas.min().item(), 3)}, "
        log_string += f"Max: {round(sigmas.max().item(), 3)}, "
        log_string += f"Mean: {round(sigmas.mean().item(), 3)}"
        logger.info(log_string)

    def save_sigmas(self, path: str) -> None:
        torch.save(self.get_sigmas_weights(), path)

    def get_sigmas_weights(self) -> torch.Tensor:
        """Return the sigmas weights (flatten) of the embedding layer."""
        return self.noise_simulator.get_sigmas_weights()


class GaussianNoise(torch.nn.Module):
    """Simulate Gaussian noise with trainable sigma to add to the embeddings"""

    def __init__(self, bert_model, cfg):
        super().__init__()
        self.cfg = cfg
        self.bert_model = bert_model  # BERT model
        self.initialize()

    def initialize(self):
        """Initialize the noise module with an embedding layer for thetas. Sigmas are obtained by exp(thetas)."""
        num_concepts = len(self.bert_model.embeddings.concept_embeddings.weight.data)
        self.thetas_embedding = torch.nn.Embedding(num_concepts, 1)
        # Initialize thetas such that exp(theta) starts close to 0
        self.thetas_embedding.weight.data.fill_(-10)  # exp(-10) = 4.5e-5

    def simulate_noise(self, concepts, embeddings: torch.Tensor) -> torch.Tensor:
        """Simulate Gaussian noise using the sigmas derived from thetas"""
        thetas = self.thetas_embedding(concepts).squeeze(
            -1
        )  # select only for present concepts
        sigmas = torch.exp(thetas)

        std_normal_noise = torch.randn_like(embeddings, device=embeddings.device)
        scaled_noise = std_normal_noise * sigmas.unsqueeze(-1)
        return scaled_noise

    def get_sigmas_weights(self) -> torch.Tensor:
        """Return the sigmas as weights of the embedding layer."""
        return torch.exp(self.thetas_embedding.weight).flatten()


class ModelOutputs:
    def __init__(
        self,
        logits=None,
        perturbed_logits=None,
        loss=None,
        hidden_states=None,
        perturbed_hidden_states=None,
    ):
        self.loss = loss
        self.logits = logits
        self.perturbed_logits = perturbed_logits
        self.hidden_states = hidden_states
        self.perturbed_hidden_states = perturbed_hidden_states
