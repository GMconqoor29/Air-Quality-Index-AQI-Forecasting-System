import numpy as np
import torch
import torch.nn as nn


def calculate_sample_weights(
    targets: np.ndarray,
    threshold: float,
    severity_span: float,
    high_aqi_weight: float = 3.0,
) -> np.ndarray:
    """Increase training weight linearly for targets above the AQI threshold."""
    values = np.asarray(targets, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("targets must contain only finite values.")
    if severity_span <= 0 or high_aqi_weight < 0:
        raise ValueError("severity_span must be positive and high_aqi_weight non-negative.")

    severity = np.clip((values - threshold) / severity_span, 0.0, 1.0)
    return (1.0 + high_aqi_weight * severity).astype(np.float32)


class AsymmetricWeightedMSE(nn.Module):
    """AQI-weighted MSE with an extra penalty for under-forecasting."""

    def __init__(
        self,
        threshold: float,
        severity_span: float,
        high_aqi_weight: float = 3.0,
        underprediction_penalty: float = 2.0,
    ):
        super().__init__()
        if severity_span <= 0:
            raise ValueError("severity_span must be positive.")
        if high_aqi_weight < 0 or underprediction_penalty < 0:
            raise ValueError("Loss weights must be non-negative.")
        self.threshold = float(threshold)
        self.severity_span = float(severity_span)
        self.high_aqi_weight = float(high_aqi_weight)
        self.underprediction_penalty = float(underprediction_penalty)

    def sample_weights(self, targets: torch.Tensor) -> torch.Tensor:
        severity = ((targets - self.threshold) / self.severity_span).clamp(0.0, 1.0)
        return 1.0 + self.high_aqi_weight * severity

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if predictions.shape != targets.shape:
            raise ValueError("predictions and targets must have the same shape.")
        weights = self.sample_weights(targets)
        underprediction = (predictions < targets).to(predictions.dtype)
        weights = weights * (1.0 + self.underprediction_penalty * underprediction)
        squared_errors = (predictions - targets).square()
        return (weights * squared_errors).mean()


class SoftGatingEnsemble(nn.Module):
    """Learn input-dependent convex weights over the base model forecasts."""

    def __init__(
        self,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        model_count: int,
    ):
        super().__init__()
        self.register_buffer("feature_mean", feature_mean.reshape(1, -1))
        self.register_buffer("feature_std", feature_std.clamp_min(1e-6).reshape(1, -1))
        self.gate = nn.Linear(feature_mean.numel(), model_count)

    def forward(self, model_predictions: torch.Tensor, context: torch.Tensor):
        features = torch.cat((model_predictions, context), dim=1)
        normalized = (features - self.feature_mean) / self.feature_std
        weights = torch.softmax(self.gate(normalized), dim=1)
        prediction = (weights * model_predictions).sum(dim=1, keepdim=True)
        return prediction, weights


def make_gating_context(
    sequences: np.ndarray,
    target_idx: int,
    trend_hours: int = 6,
) -> np.ndarray:
    if sequences.ndim != 3 or sequences.shape[1] <= trend_hours:
        raise ValueError("sequences must contain a time window longer than trend_hours.")
    recent = sequences[:, -1, target_idx]
    previous = sequences[:, -(trend_hours + 1), target_idx]
    return np.column_stack((recent, recent - previous)).astype(np.float32)


def fit_soft_gate(
    validation_predictions: np.ndarray,
    validation_context: np.ndarray,
    validation_targets: np.ndarray,
    criterion: AsymmetricWeightedMSE,
    device: str,
    epochs: int = 300,
    learning_rate: float = 0.03,
) -> SoftGatingEnsemble:
    predictions = torch.as_tensor(validation_predictions, dtype=torch.float32, device=device)
    context = torch.as_tensor(validation_context, dtype=torch.float32, device=device)
    targets = torch.as_tensor(validation_targets, dtype=torch.float32, device=device).reshape(-1, 1)
    if predictions.ndim != 2 or predictions.shape[0] != targets.shape[0]:
        raise ValueError("validation predictions and targets must have matching rows.")
    if context.ndim != 2 or context.shape[0] != targets.shape[0]:
        raise ValueError("validation context and targets must have matching rows.")
    if predictions.shape[1] == 0 or not torch.isfinite(predictions).all():
        raise ValueError("validation predictions must be finite and include at least one model.")
    if not torch.isfinite(context).all() or not torch.isfinite(targets).all():
        raise ValueError("validation context and targets must be finite.")
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive.")

    all_features = torch.cat((predictions, context), dim=1)
    gate = SoftGatingEnsemble(
        feature_mean=all_features.mean(dim=0),
        feature_std=all_features.std(dim=0, unbiased=False),
        model_count=predictions.shape[1],
    ).to(device)
    nn.init.zeros_(gate.gate.weight)
    nn.init.zeros_(gate.gate.bias)

    optimizer = torch.optim.Adam(gate.parameters(), lr=learning_rate, weight_decay=1e-2)
    gate.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        forecast, _ = gate(predictions, context)
        loss = criterion(forecast, targets)
        loss.backward()
        optimizer.step()
    gate.eval()
    return gate


def xgboost_asymmetric_objective(
    threshold: float,
    severity_span: float,
    high_aqi_weight: float = 3.0,
    underprediction_penalty: float = 2.0,
):
    def objective(predictions, matrix):
        targets = matrix.get_label()
        sample_weights = matrix.get_weight()
        if sample_weights.size == 0:
            sample_weights = np.ones_like(targets)
        elif sample_weights.shape != targets.shape:
            raise ValueError("XGBoost sample weights must match the labels.")
        sample_weights = sample_weights * calculate_sample_weights(
            targets, threshold, severity_span, high_aqi_weight
        )
        underprediction = predictions < targets
        asymmetric_weights = sample_weights * np.where(
            underprediction, 1.0 + underprediction_penalty, 1.0
        )
        errors = predictions - targets
        gradient = 2.0 * asymmetric_weights * errors
        hessian = 2.0 * asymmetric_weights
        return gradient, hessian

    return objective


def xgboost_asymmetric_metric(
    threshold: float,
    severity_span: float,
    high_aqi_weight: float = 3.0,
    underprediction_penalty: float = 2.0,
):
    def metric(predictions, matrix):
        targets = matrix.get_label()
        weights = calculate_sample_weights(
            targets, threshold, severity_span, high_aqi_weight
        )
        weights *= np.where(
            predictions < targets, 1.0 + underprediction_penalty, 1.0
        )
        errors = np.square(predictions - targets)
        return "amse", float(np.mean(weights * errors))

    return metric
