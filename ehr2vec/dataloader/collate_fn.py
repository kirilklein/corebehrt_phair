import torch


def static(data: list) -> dict:
    padded_data = {
        key: torch.stack([torch.tensor(patient[key]) for patient in data])
        for key in data[0].keys()
    }

    return padded_data


def get_bucket_length(length, buckets=None):
    """Return the smallest bucket size that fits the sequence length"""
    if buckets is None:
        buckets = [64, 128, 256, 512, 1024, 2048]
    for bucket in buckets:
        if length <= bucket:
            return bucket
    return buckets[-1]  # Use largest bucket if sequence is longer


def bucketed_dynamic_padding(data: list) -> dict:
    """Pad batch data to the nearest bucket size.

    Args:
        data: List of patient dictionaries containing:
            - concept: Tensor of variable length sequences
            - target: Tensor (1D) or float (0D) for predictions
            - time2event: Tensor (1D) or float (0D) for time information
            - other fields: Tensor of same length as concept

    Returns:
        dict: Padded data with all sequences in batch padded to same bucket size

    Raises:
        ValueError: If data is empty or missing required fields
    """
    if not data:
        raise ValueError("Empty batch data provided")
    if "concept" not in data[0]:
        raise ValueError("Data must contain 'concept' field")

    # Find max length in batch
    max_len = max([len(patient["concept"]) for patient in data])
    # Get appropriate bucket size
    bucket_len = get_bucket_length(max_len)

    for patient in data:
        difference = bucket_len - len(patient["concept"])
        for key, values in patient.items():
            if key in ["target", "time2event"]:
                if isinstance(values, float):  # 0D: For finetuning
                    patient[key] = torch.tensor(values)
                    continue
                elif values.ndim == 1:  # 1D: For normal pretraining
                    filler = torch.ones(difference, dtype=values.dtype) * -100
            else:
                filler = torch.zeros(difference, dtype=values.dtype)
            patient[key] = torch.cat((values, filler), dim=0)

    padded_data = {}
    for key in data[0].keys():
        padded_data[key] = torch.stack([patient[key] for patient in data])

    return padded_data
