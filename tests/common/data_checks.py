from typing import List

def check_codes(feats: dict, control_code: str, exposed_codes: List[str], vocabulary: dict):
    """Check for presence of control and exposure codes in patient sequences.
    
    This function checks each patient sequence for the presence of control codes and exposure codes,
    validating that they are mutually exclusive (a sequence should not have both).

    Args:
        feats (dict): Dictionary containing patient features, with 'concept' key containing sequences
        control_code (str): Code representing control/unexposed status
        exposed_codes (List[str]): List of regex patterns matching codes representing exposure
        vocabulary (dict): Mapping of codes to vocabulary indices

    Returns:
        Tuple[List[bool], List[bool], List[bool]]: Returns 3 lists:
            - results: True if sequence has either control OR exposure codes but not both
            - both: True if sequence has both control AND exposure codes (invalid)
            - none: True if sequence has neither control NOR exposure codes (invalid)
    """
    import re
    control_code = vocabulary[control_code]
    exposed_codes = [idx for code_pattern in exposed_codes for code, idx in vocabulary.items() if re.match(code_pattern, code)]
    results = []
    both = []
    none = []
    for concepts in feats['concept']:
        has_control = control_code in concepts
        has_exposed = any(code in concepts for code in exposed_codes)
        results.append(has_control != has_exposed)
        both.append(has_control and has_exposed)
        none.append(not has_control and not has_exposed)
    return results, both, none