def rolling_origin_splits(sample_count: int, fold_count: int = 4, initial_fraction: float = 0.5):
    if sample_count < 2 or fold_count < 1:
        raise ValueError("sample_count must be at least 2 and fold_count at least 1.")
    if not 0 < initial_fraction < 1:
        raise ValueError("initial_fraction must be between 0 and 1.")

    initial_end = int(sample_count * initial_fraction)
    remaining = sample_count - initial_end
    if initial_end < 1 or remaining < fold_count:
        raise ValueError("Not enough samples for the requested rolling-origin folds.")

    boundaries = [initial_end + (remaining * fold) // fold_count for fold in range(fold_count + 1)]
    return [
        (0, boundaries[fold], boundaries[fold + 1])
        for fold in range(fold_count)
    ]
