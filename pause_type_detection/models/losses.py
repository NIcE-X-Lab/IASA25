import torch
import torch.nn as nn
import torch.nn.functional as F

class PenaltyHuberLoss(nn.Module):
    def __init__(self, delta=0.5, penalty_weights=None):
        """
        Args:
            delta (float): Huber threshold
            penalty_weights (dict): e.g., {0.0: 1.0, 1.0: 5.0, 2.0: 5.0, 3.0: 5.0}
        """
        super().__init__()
        self.delta = delta
        if penalty_weights is None:
            self.penalty_weights = {0.0: 1.0, 1.0: 5.0, 2.0: 4.0, 3.0: 2.0}
        else:
            self.penalty_weights = penalty_weights

    def forward(self, y_pred, y_true):
        y_pred = y_pred.float()
        y_true = y_true.float()

        # Flatten
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        class_ids = y_true.round().clamp(0, 3)
        weights = torch.tensor(
            [self.penalty_weights[cls.item()] for cls in class_ids],
            device=y_true.device
        )

        diff = y_pred - y_true
        abs_diff = torch.abs(diff)
        huber_loss = torch.where(
            abs_diff < self.delta,
            0.5 * diff ** 2,
            self.delta * (abs_diff - 0.5 * self.delta)
        )
        weighted_loss = (huber_loss * weights).mean()
        return weighted_loss
    
class PenaltyCrossEntropyLoss(nn.Module):
    def __init__(self, penalty_matrix, weight=None, ignore_index=-100):
        super().__init__()
        self.penalty_matrix = torch.tensor(penalty_matrix, dtype=torch.float32)
        self.class_weight = torch.tensor(weight, dtype=torch.float32) if weight is not None else None
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        device = logits.device
        logits = logits.to(device)
        targets = targets.to(device)
        penalty_matrix = self.penalty_matrix.to(device)
        
        log_probs = F.log_softmax(logits, dim=1)
        nll = F.nll_loss(log_probs, targets, reduction='none', ignore_index=self.ignore_index)
        
        if self.class_weight is not None:
            class_weight = self.class_weight.to(device)
            nll = nll * class_weight[targets]

        # Compute weighted penalty
        probs = F.softmax(logits, dim=1)
        penalties = penalty_matrix[targets]  # shape: (N, C)
        weighted_penalty = (penalties * probs).sum(dim=1)

        total_loss = nll + weighted_penalty

        # Mask ignored targets
        mask = targets != self.ignore_index
        return total_loss[mask].mean()

class DurationAwareFocalLoss(nn.Module):
    """Duration-aware regression loss that weights short breaks more heavily"""
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None, delta=0.5):
        super(DurationAwareFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta
        
        if class_weights is None:
            # Duration-aware weights: short breaks get higher weights
            # 'o': 1.0, 's': 4.0 (very short), 'b': 3.0 (short), 'bs': 2.0 (longer)
            self.class_weights = {0.0: 1.0, 1.0: 4.0, 2.0: 3.0, 3.0: 2.0}
        else:
            self.class_weights = class_weights
    
    def forward(self, y_pred, y_true):
        y_pred = y_pred.float()
        y_true = y_true.float()
        
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)
        
        # Class weights based on duration characteristics
        true_classes = y_true_flat.round().clamp(0, 3)
        weights = torch.tensor([
            self.class_weights[cls.item()] for cls in true_classes
        ], device=y_true.device)
        
        # Distance-based "difficulty" - further predictions are harder
        distances = torch.abs(y_pred_flat - y_true_flat)
        difficulties = torch.clamp(distances / self.delta, min=0.0, max=1.0)
        
        # Focal weighting: focus more on hard examples
        focal_weights = torch.pow(difficulties, self.gamma)
        
        # Base loss (smooth L1)
        base_loss = F.smooth_l1_loss(y_pred_flat, y_true_flat, 
                                   reduction='none', beta=self.delta)
        
        # Apply focal and class weighting
        focal_loss = self.alpha * focal_weights * base_loss * weights
        
        return focal_loss.mean()

class TwoStageRegressionLoss(nn.Module):
    """Combined loss for two-stage regression model"""
    def __init__(self, 
                 break_loss_weight=1.0,
                 regression_loss_weight=1.0,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 class_weights=None,
                 delta=0.5):
        super(TwoStageRegressionLoss, self).__init__()
        
        self.break_loss_weight = break_loss_weight
        self.regression_loss_weight = regression_loss_weight
        
        # Stage 1: Break detection loss (Binary Cross-Entropy)
        self.break_loss = nn.BCELoss()
        
        # Stage 2: Duration-aware regression loss
        self.regression_loss = DurationAwareFocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            class_weights=class_weights,
            delta=delta
        )
    
    def forward(self, break_predictions, regression_predictions, 
                break_targets, regression_targets):
        """
        Args:
            break_predictions: (B, T) - Stage 1 break probabilities
            regression_predictions: (B, T) - Stage 2 regression values
            break_targets: (B, T) - Binary break targets (0=no break, 1=any break)
            regression_targets: (B, T) - Continuous regression targets (0-3)
        """
        # Safety checks for break_targets
        break_targets = torch.clamp(break_targets, 0.0, 1.0)
        break_predictions = torch.clamp(break_predictions, 0.0, 1.0)
        
        # Align break_predictions to break_targets time length if needed
        if break_predictions.dim() == 2 and break_targets.dim() == 2 and break_predictions.size(1) != break_targets.size(1):
            # Interpolate along time dimension
            break_predictions = F.interpolate(
                break_predictions.unsqueeze(1), size=break_targets.size(1), mode='linear', align_corners=False
            ).squeeze(1)

        # Stage 1: Break detection loss
        break_loss = self.break_loss(break_predictions, break_targets)
        
        # Align regression_predictions to regression_targets time length if needed
        if (regression_predictions.dim() == 2 and regression_targets.dim() == 2 and 
            regression_predictions.size(1) != regression_targets.size(1)):
            regression_predictions = F.interpolate(
                regression_predictions.unsqueeze(1), size=regression_targets.size(1), mode='linear', align_corners=False
            ).squeeze(1)

        # Stage 2: Duration-aware regression loss
        regression_loss = self.regression_loss(regression_predictions, regression_targets)
        
        # Combined loss
        total_loss = (self.break_loss_weight * break_loss + 
                     self.regression_loss_weight * regression_loss)
        
        return total_loss, break_loss, regression_loss

class TwoStageClassificationLoss(nn.Module):

    def __init__(
        self,
        break_loss_weight: float = 1.0,
        duration_loss_weight: float = 1.0,
        ignore_index: int = -100,
        class_to_value: torch.Tensor | None = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        class_weights: dict | None = None,
        delta: float = 0.5,
    ):
        super().__init__()
        self.break_loss_weight = break_loss_weight
        self.duration_loss_weight = duration_loss_weight
        self.ignore_index = ignore_index

        # BCE for stage 1
        self.break_loss = nn.BCELoss()

        # Duration-aware regression for stage 2
        self.duration_loss = DurationAwareFocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            class_weights=class_weights,
            delta=delta,
        )

        # Optional mapping from class index -> scalar value for logits-based mode
        # If None and logits are provided, defaults to [1, 2, 3, ...] per class index
        self.register_buffer(
            "class_to_value",
            class_to_value if class_to_value is not None else torch.tensor([], dtype=torch.float),
            persistent=False,
        )

    def _align_time(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 2 and target.dim() == 2 and pred.size(1) != target.size(1):
            return F.interpolate(pred.unsqueeze(1), size=target.size(1), mode='linear', align_corners=False).squeeze(1)
        return pred

    def _logits_to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert class logits (B, T, C) to scalar regression predictions (B, T).

        Uses expected value under softmax with class_to_value mapping.
        If class_to_value is empty, uses [1, 2, ..., C].
        """
        B, T, C = logits.shape
        probs = logits.softmax(dim=-1)
        if self.class_to_value.numel() == 0:
            values = torch.arange(1, C + 1, device=logits.device, dtype=probs.dtype)
        else:
            if self.class_to_value.numel() != C:
                raise ValueError(f"class_to_value length {self.class_to_value.numel()} != num classes {C}")
            values = self.class_to_value.to(device=logits.device, dtype=probs.dtype)
        # weighted sum over class dimension
        scalar = (probs * values.view(1, 1, C)).sum(dim=-1)
        return scalar  # (B, T)

    def forward(
        self,
        break_predictions: torch.Tensor,
        stage2_predictions: torch.Tensor,
        break_targets: torch.Tensor,
        stage2_targets: torch.Tensor,
        use_logits_for_stage2: bool = False,
    ):
        # Align break_predictions to break_targets if needed
        break_predictions = torch.clamp(break_predictions, 0.0, 1.0)
        break_targets = torch.clamp(break_targets, 0.0, 1.0)
        if break_predictions.size(1) != break_targets.size(1):
            break_predictions = self._align_time(break_predictions, break_targets)

        # Stage 1 BCE
        bce = self.break_loss(break_predictions, break_targets)

        # Prepare stage-2 predictions and targets
        if use_logits_for_stage2:
            # Convert class logits (B, T, C) to scalar (B, T)
            regression_predictions = self._logits_to_scalar(stage2_predictions)
            # Map class targets (B, T) -> scalar targets (B, T)
            cls_targets = stage2_targets.long()
            if self.class_to_value.numel() == 0:
                # default mapping 0..C-1 -> 1..C
                C = stage2_predictions.size(-1)
                default_map = torch.arange(1, C + 1, device=stage2_predictions.device)
                regression_targets = default_map[cls_targets.clamp(0, C - 1)]
            else:
                map_vec = self.class_to_value.to(stage2_predictions.device)
                C = map_vec.numel()
                regression_targets = map_vec[cls_targets.clamp(0, C - 1)]
        else:
            # Direct regression predictions/targets (B, T)
            regression_predictions = stage2_predictions
            regression_targets = stage2_targets

        # Optionally align time length
        if regression_predictions.dim() == 2 and regression_targets.dim() == 2 and regression_predictions.size(1) != regression_targets.size(1):
            regression_predictions = self._align_time(regression_predictions, regression_targets)

        # Mask out ignore_index if stage2_targets are class indices or if a mask is provided
        if use_logits_for_stage2 or regression_targets.dtype in (torch.long, torch.int64, torch.int32):
            flat_targets = stage2_targets.view(-1)
            valid_mask = flat_targets != self.ignore_index
            if not valid_mask.any():
                duration = regression_predictions.view(-1).sum() * 0.0
            else:
                duration = self.duration_loss(
                    regression_predictions.view(-1)[valid_mask],
                    regression_targets.view(-1)[valid_mask],
                )
        else:
            # Treat all as valid
            duration = self.duration_loss(regression_predictions, regression_targets)

        total = self.break_loss_weight * bce + self.duration_loss_weight * duration
        return total, bce, duration

