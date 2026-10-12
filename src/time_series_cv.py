from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import MinMaxScaler

import src.config as config
from src.data_loader import create_sequences, load_and_preprocess_data
from src.walk_forward import rolling_origin_splits


@dataclass
class ExpandingWindowFold:
    X_train: np.ndarray
    y_train: np.ndarray
    X_calibration: np.ndarray
    y_calibration: np.ndarray
    calibration_timestamps: np.ndarray
    X_score: np.ndarray
    y_score: np.ndarray
    score_timestamps: np.ndarray
    scaler: MinMaxScaler


def build_expanding_window_folds(
    data_dir: str = "Dataset/archive",
    fold_count: int = 4,
    initial_fraction: float = 0.5,
    calibration_fraction: float = 0.5,
) -> list[ExpandingWindowFold]:
    """Build fold-local scaled data with chronological calibration and score blocks."""
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between 0 and 1.")

    frame = load_and_preprocess_data(data_dir)
    all_features = (
        frame[config.FEATURE_COLS]
        .astype(np.float64)
        .ffill(limit=config.MAX_FORWARD_FILL_HOURS)
        .to_numpy()
    )
    raw_targets = frame[config.TARGET_COL].to_numpy(dtype=np.float64)
    timestamps = frame[config.TIME_COL].to_numpy()
    development_end = int(len(frame) * (config.TRAIN_RATIO + config.VAL_RATIO))
    target_idx = config.FEATURE_COLS.index(config.TARGET_COL)

    _, _, valid_target_indices = create_sequences(
        all_features,
        config.WINDOW_SIZE,
        config.HORIZON,
        target_idx,
        target_values=raw_targets,
        return_target_indices=True,
    )
    development_target_indices = valid_target_indices[
        valid_target_indices < development_end
    ]
    splits = rolling_origin_splits(
        len(development_target_indices),
        fold_count=fold_count,
        initial_fraction=initial_fraction,
    )

    folds = []
    for _, train_stop, validation_stop in splits:
        validation_size = validation_stop - train_stop
        calibration_size = int(validation_size * calibration_fraction)
        calibration_stop = train_stop + calibration_size
        if calibration_size < 1 or calibration_stop >= validation_stop:
            raise ValueError(
                "Each validation block must have non-empty calibration and score portions."
            )

        train_target_indices = development_target_indices[:train_stop]
        calibration_target_indices = development_target_indices[
            train_stop:calibration_stop
        ]
        score_target_indices = development_target_indices[
            calibration_stop:validation_stop
        ]
        scaler_fit_stop = int(train_target_indices[-1]) + 1
        complete_rows = np.isfinite(all_features[:scaler_fit_stop]).all(axis=1)
        if not complete_rows.any():
            raise ValueError("A fold has no complete rows for fitting its scaler.")

        scaler = MinMaxScaler()
        scaler.fit(all_features[:scaler_fit_stop][complete_rows])
        scaled_features = scaler.transform(all_features)
        scaled_targets = np.full(raw_targets.shape, np.nan, dtype=np.float64)
        finite_targets = np.isfinite(raw_targets)
        scaled_targets[finite_targets] = (
            raw_targets[finite_targets] * scaler.scale_[target_idx]
            + scaler.min_[target_idx]
        )
        sequences, targets, target_indices = create_sequences(
            scaled_features,
            config.WINDOW_SIZE,
            config.HORIZON,
            target_idx,
            target_values=scaled_targets,
            return_target_indices=True,
        )
        target_positions = {
            int(target_index): position
            for position, target_index in enumerate(target_indices)
        }
        try:
            train_positions = np.asarray(
                [target_positions[int(index)] for index in train_target_indices],
                dtype=np.int64,
            )
            calibration_positions = np.asarray(
                [target_positions[int(index)] for index in calibration_target_indices],
                dtype=np.int64,
            )
            score_positions = np.asarray(
                [target_positions[int(index)] for index in score_target_indices],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError(
                "A fold contains targets without valid 48-hour input sequences."
            ) from exc

        folds.append(
            ExpandingWindowFold(
                X_train=sequences[train_positions],
                y_train=targets[train_positions],
                X_calibration=sequences[calibration_positions],
                y_calibration=targets[calibration_positions],
                calibration_timestamps=timestamps[calibration_target_indices],
                X_score=sequences[score_positions],
                y_score=targets[score_positions],
                score_timestamps=timestamps[score_target_indices],
                scaler=scaler,
            )
        )
    return folds
