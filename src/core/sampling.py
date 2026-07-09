"""
Sampling utilities (e.g. replicable random indices for mapping batch custom_id to dataset).
"""

import random
from typing import List


def get_sampled_indices(
    population_size: int,
    sample_size: int = 100,
    seed: int = 50,
) -> List[int]:
    """
    Return a replicable random sample of indices in [0, population_size).
    Uses random.seed(seed) and random.sample. Used to map batch custom_id
    back to original dataset keys (paper_keys[sampled_indices[custom_id]]).
    """
    random.seed(seed)
    return random.sample(range(population_size), k=min(sample_size, population_size))
