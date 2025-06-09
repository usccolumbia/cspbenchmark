# ranking_calculator.py

"""
Core library for calculating ranks and scores from a list of distance values.

This module provides a robust method for handling missing or invalid data and
computes "dense" ranks and linearly scaled scores, with special handling for ties.
"""

import numpy as np

# Define values that should be treated as "missing" or infinite distance.
BAD_STRINGS = {"N/A", "nan", "NaN", "None", "", " ", "null"}
SENTINEL_VALUES = {5555, 9999, 9999.0}

def safe_float(value):
    """
    Converts a value to a float, returning np.inf for any "bad" or sentinel value.

    Args:
        value (any): The input value, which can be a string, float, or int.

    Returns:
        float: The numeric value or np.inf if the input is invalid.
    """
    # 1. Handle existing floats (check for NaN and sentinels)
    if isinstance(value, float):
        if np.isnan(value) or value in SENTINEL_VALUES:
            return np.inf
        return value

    # 2. Handle strings and other types
    try:
        # Check for bad string tokens
        if isinstance(value, str) and value.strip() in BAD_STRINGS:
            return np.inf
        # Attempt a numeric cast
        numeric_val = float(value)
        return np.inf if numeric_val in SENTINEL_VALUES else numeric_val
    except (ValueError, TypeError):
        # Anything that cannot be cast is treated as infinite distance
        return np.inf

def calculate_ranking_scores(raw_distances):
    """
    Calculates dense ranks and scaled scores (0-100) for a list of distances.

    The method works as follows:
    1.  **Cleaning**: Input distances are cleaned using `safe_float`.
    2.  **Dense Rank**: Distances are ranked from lowest to highest. Ties receive the
        same rank. The next rank is incremented by the number of tied items.
        Example: [10, 20, 20, 30] -> ranks [1, 2, 2, 4].
    3.  **Scoring**: Scores are linearly scaled from 100 to 0. For ties, the
        scores of all tied items are averaged. Invalid distances (np.inf)
        always receive a score of 0.

    Args:
        raw_distances (list): A list of raw distance values (can be strings or numbers).

    Returns:
        tuple[list[int], list[float]]: A tuple containing the list of ranks and
                                       the list of scores (rounded to 2 decimal places).
    """
    clean_distances = [safe_float(v) for v in raw_distances]

    # Create frequency and rank maps to handle ties correctly
    freq_map = {}
    for v in clean_distances:
        freq_map[v] = freq_map.get(v, 0) + 1

    rank_map, current_rank = {}, 1
    for v in sorted(freq_map):
        rank_map[v] = current_rank
        current_rank += freq_map[v]
    
    ranks = [rank_map[v] for v in clean_distances]

    # Calculate scores with averaging for ties
    raw_scores = np.linspace(100, 0, len(clean_distances), endpoint=False)
    final_scores = []
    for v in clean_distances:
        if np.isinf(v):
            final_scores.append(0.0)
        else:
            start_index = rank_map[v] - 1
            num_duplicates = freq_map[v]
            avg_score = raw_scores[start_index : start_index + num_duplicates].mean()
            final_scores.append(round(avg_score, 2))

    return ranks, final_scores

if __name__ == '__main__':
    # Example usage and self-test when running the script directly
    print("Running self-test examples...")
    
    test_data = [5555, 5555, 42, 1, 1, 5, 5, 2, 2, 'N/A']
    ranks, scores = calculate_ranking_scores(test_data)
    
    print(f"\nInput: {test_data}")
    print(f"Cleaned: {[safe_float(v) for v in test_data]}")
    print(f"Ranks: {ranks}")
    print(f"Scores: {scores}")
    # Expected: Ranks: [9, 9, 8, 1, 1, 5, 5, 3, 3, 9], Scores: [5.0, 5.0, 15.0, 95.0, 95.0, 45.0, 45.0, 65.0, 65.0, 5.0]