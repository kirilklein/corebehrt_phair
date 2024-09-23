import torch
import logging

logger = logging.getLogger(__name__)  # Get the logger for this module


def neg_partial_log_likelihood(log_hazard, event, time2event):
    """
    Compute the negative partial log-likelihood for the cox proportional hazards model.
    risk_scores: Tensor of predicted risk scores from the model
    event: Tensor of event indicators (1 if the event occurred, 0 if censored)
    time2event: Tensor of observed times
    """
    if any([event.sum() == 0, len(log_hazard.size()) == 0]):
        logger.warn("No events OR single sample. Returning zero loss for the batch")
        return torch.tensor(0.0, requires_grad=True)

    # Sort the instances by observed time in descending order
    sorted_indices = torch.argsort(time2event)
    log_hazard_sorted = log_hazard[sorted_indices]
    event_sorted = event[sorted_indices].bool()

    # we will not deal with ties, because they are very rare in practice
    log_cum_sum = torch.logcumsumexp(log_hazard_sorted.flip(0), dim=0).flip(0)
    pll = (log_hazard_sorted - log_cum_sum)[event_sorted]

    return -torch.nanmean(pll)
