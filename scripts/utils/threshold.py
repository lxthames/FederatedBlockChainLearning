"""
Adaptive Threshold Utilities

Functions for adaptive PoC threshold calculation.
"""


def get_adaptive_poc_threshold(round_num: int, base_threshold: float = 0.1, 
                               min_threshold: float = 0.01) -> float:
    """
    Adaptive PoC threshold: starts low (permissive) and increases over time.
    This allows updates early in training when model is still learning.
    
    Args:
        round_num: Current round number
        base_threshold: Final threshold (default: 0.1)
        min_threshold: Starting threshold (default: 0.01)
    
    Returns:
        Threshold value for this round
    """
    if round_num <= 3:
        return min_threshold  # Very permissive early (0.01)
    elif round_num <= 10:
        # Gradually increase
        progress = (round_num - 3) / 7  # 0 to 1 over rounds 4-10
        return min_threshold + (base_threshold - min_threshold) * progress
    else:
        return base_threshold  # Full threshold (0.1)

