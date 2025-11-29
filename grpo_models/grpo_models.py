"""
GRPO (Group Relative Policy Optimization) Models for MLB Prediction

This module implements GRPO-style models adapted for baseball prediction.
GRPO uses relative comparisons and reward-based learning to optimize predictions.

Key concepts adapted from GRPO:
1. Group-based comparisons: Compare teams/matchups within groups
2. Relative policy optimization: Learn from relative outcomes
3. Reward shaping: Use betting outcomes as rewards
4. Preference learning: Model which team is more likely to win

Model Types:
- GRPOClassifier: Neural network with GRPO-style training
- GRPOEnsemble: Ensemble of GRPO models with different feature subsets
- GRPORankingModel: Pairwise ranking approach for matchups
- GRPOBettingOptimizer: Model optimized for betting ROI

Author: MLB Prediction Project
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Deep learning imports with fallback
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Using sklearn-based GRPO models.")

# Sklearn imports
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV


@dataclass
class GRPOConfig:
    """Configuration for GRPO models"""
    # Neural network architecture
    hidden_layers: List[int] = None
    dropout_rate: float = 0.3
    learning_rate: float = 0.001

    # GRPO-specific parameters
    group_size: int = 8  # Number of games to compare in each group
    reward_scale: float = 1.0  # Scale factor for rewards
    kl_coef: float = 0.1  # KL divergence coefficient
    clip_epsilon: float = 0.2  # PPO-style clipping

    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    early_stopping_patience: int = 10

    # Regularization
    l2_weight: float = 0.01
    use_batch_norm: bool = True

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [128, 64, 32]


class GRPOBaseModel(BaseEstimator, ClassifierMixin):
    """
    Base class for GRPO-style models.

    This implements the core GRPO concepts:
    1. Group-based training: Games are grouped and compared
    2. Relative rewards: Learning from relative outcomes
    3. Policy optimization: Optimizing for decision-making
    """

    def __init__(self, config: Optional[GRPOConfig] = None):
        self.config = config or GRPOConfig()
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None

    def _compute_relative_rewards(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        group_indices: np.ndarray
    ) -> np.ndarray:
        """
        Compute relative rewards within groups.

        GRPO compares outcomes within groups rather than using absolute metrics.
        This helps the model learn relative preferences.
        """
        rewards = np.zeros_like(predictions, dtype=float)

        for group_id in np.unique(group_indices):
            mask = group_indices == group_id
            group_preds = predictions[mask]
            group_outcomes = outcomes[mask]

            # Compute baseline (mean prediction in group)
            baseline = group_preds.mean()

            # Reward is relative to baseline
            # Correct predictions above baseline get higher rewards
            for i, (pred, outcome) in enumerate(zip(group_preds, group_outcomes)):
                correct = (pred > 0.5) == outcome
                relative_confidence = abs(pred - 0.5) - abs(baseline - 0.5)

                if correct:
                    rewards[mask][i] = 1.0 + relative_confidence * self.config.reward_scale
                else:
                    rewards[mask][i] = -1.0 - relative_confidence * self.config.reward_scale

        return rewards

    def _create_groups(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """Create group indices for GRPO training"""
        n_samples = len(X)
        n_groups = n_samples // self.config.group_size

        # Assign samples to groups
        group_indices = np.repeat(np.arange(n_groups), self.config.group_size)

        # Handle remaining samples
        remainder = n_samples - len(group_indices)
        if remainder > 0:
            group_indices = np.concatenate([
                group_indices,
                np.full(remainder, n_groups)
            ])

        # Shuffle group assignments
        np.random.shuffle(group_indices)

        return group_indices


class GRPOClassifier(GRPOBaseModel):
    """
    Neural network classifier with GRPO-style training.

    Uses a policy gradient approach where the policy is the predicted
    probability of home team winning, and rewards are based on correct
    predictions and betting outcomes.
    """

    def __init__(self, config: Optional[GRPOConfig] = None):
        super().__init__(config)
        self.model = None
        self.optimizer = None

    def _build_model(self, input_dim: int):
        """Build the neural network model using sklearn"""
        self.model = MLPClassifier(
            hidden_layer_sizes=tuple(self.config.hidden_layers),
            activation='relu',
            solver='adam',
            alpha=self.config.l2_weight,
            learning_rate='adaptive',
            learning_rate_init=self.config.learning_rate,
            max_iter=self.config.epochs,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=self.config.early_stopping_patience,
            random_state=42
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GRPOClassifier':
        """
        Fit the GRPO classifier.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Binary outcomes (1 = home team won, 0 = away team won)

        Returns:
        --------
        self
        """
        # Store feature names if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values

        y = np.array(y).astype(int)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Build model
        input_dim = X_scaled.shape[1]
        self._build_model(input_dim)

        # Fit with sample weights based on GRPO rewards
        # Initial fit to get baseline predictions
        self.model.fit(X_scaled, y)

        # GRPO refinement iterations
        for iteration in range(3):
            # Get current predictions
            proba = self.model.predict_proba(X_scaled)[:, 1]

            # Create groups and compute relative rewards
            group_indices = self._create_groups(X_scaled, y)
            rewards = self._compute_relative_rewards(proba, y, group_indices)

            # Convert rewards to sample weights (higher reward = higher weight)
            sample_weights = np.clip(rewards + 2, 0.1, 3.0)  # Shift to positive

            # Refit with sample weights
            self.model.fit(X_scaled, y, sample_weight=sample_weights)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        if isinstance(X, pd.DataFrame):
            X = X.values

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if isinstance(X, pd.DataFrame):
            X = X.values

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


if TORCH_AVAILABLE:
    class GRPONeuralNetwork(nn.Module):
        """
        PyTorch neural network for GRPO training.
        """

        def __init__(self, input_dim: int, config: GRPOConfig):
            super().__init__()
            self.config = config

            layers = []
            prev_dim = input_dim

            for hidden_dim in config.hidden_layers:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                if config.use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(config.dropout_rate))
                prev_dim = hidden_dim

            # Output layer
            layers.append(nn.Linear(prev_dim, 1))
            layers.append(nn.Sigmoid())

            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    class GRPOTorchClassifier(GRPOBaseModel):
        """
        PyTorch-based GRPO classifier with full policy gradient training.
        """

        def __init__(self, config: Optional[GRPOConfig] = None):
            super().__init__(config)
            self.model = None
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        def _grpo_loss(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
            old_predictions: torch.Tensor,
            advantages: torch.Tensor
        ) -> torch.Tensor:
            """
            Compute GRPO loss combining PPO-style clipping with group advantages.
            """
            # Ratio of new to old policy
            ratio = predictions / (old_predictions + 1e-8)

            # Clipped ratio
            clipped_ratio = torch.clamp(
                ratio,
                1 - self.config.clip_epsilon,
                1 + self.config.clip_epsilon
            )

            # PPO-style loss
            loss1 = ratio * advantages
            loss2 = clipped_ratio * advantages
            policy_loss = -torch.min(loss1, loss2).mean()

            # Binary cross-entropy for calibration
            bce_loss = nn.BCELoss()(predictions, targets.float())

            # Combined loss
            total_loss = policy_loss + bce_loss

            return total_loss

        def fit(self, X: np.ndarray, y: np.ndarray) -> 'GRPOTorchClassifier':
            """Fit the GRPO model using PyTorch"""
            if isinstance(X, pd.DataFrame):
                self.feature_names = X.columns.tolist()
                X = X.values

            y = np.array(y).astype(float)

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Build model
            input_dim = X_scaled.shape[1]
            self.model = GRPONeuralNetwork(input_dim, self.config).to(self.device)
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.l2_weight
            )

            # Convert to tensors
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)

            # Training loop
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )

            best_loss = float('inf')
            patience_counter = 0

            for epoch in range(self.config.epochs):
                self.model.train()
                epoch_loss = 0

                # Get old predictions for ratio calculation
                with torch.no_grad():
                    old_preds = self.model(X_tensor).squeeze()

                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()

                    # Forward pass
                    predictions = self.model(batch_X).squeeze()

                    # Compute advantages using group-relative rewards
                    with torch.no_grad():
                        batch_indices = torch.randint(
                            0, self.config.group_size,
                            (len(batch_X),)
                        )
                        advantages = self._compute_advantages(
                            predictions.detach().cpu().numpy(),
                            batch_y.cpu().numpy(),
                            batch_indices.numpy()
                        )
                        advantages = torch.FloatTensor(advantages).to(self.device)

                    # Compute old predictions for this batch
                    batch_old_preds = old_preds[
                        torch.randint(0, len(old_preds), (len(batch_X),))
                    ]

                    # GRPO loss
                    loss = self._grpo_loss(
                        predictions, batch_y, batch_old_preds, advantages
                    )

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                # Early stopping
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config.early_stopping_patience:
                    break

            self.is_fitted = True
            return self

        def _compute_advantages(
            self,
            predictions: np.ndarray,
            outcomes: np.ndarray,
            group_indices: np.ndarray
        ) -> np.ndarray:
            """Compute advantages for policy gradient"""
            rewards = self._compute_relative_rewards(predictions, outcomes, group_indices)

            # Normalize advantages
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            return advantages

        def predict(self, X: np.ndarray) -> np.ndarray:
            """Predict class labels"""
            proba = self.predict_proba(X)
            return (proba[:, 1] > 0.5).astype(int)

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            """Predict class probabilities"""
            if isinstance(X, pd.DataFrame):
                X = X.values

            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)

            self.model.eval()
            with torch.no_grad():
                proba = self.model(X_tensor).squeeze().cpu().numpy()

            # Return as (n_samples, 2) array
            return np.column_stack([1 - proba, proba])


class GRPOEnsemble(GRPOBaseModel):
    """
    Ensemble of GRPO models using different feature subsets and algorithms.

    This combines multiple approaches:
    1. Different feature subsets (prior season, rolling, advanced)
    2. Different base algorithms (NN, GradientBoosting, RandomForest)
    3. Weighted averaging based on validation performance
    """

    def __init__(self, config: Optional[GRPOConfig] = None):
        super().__init__(config)
        self.models = []
        self.model_weights = []
        self.feature_subsets = []

    def _create_feature_subsets(
        self,
        X: pd.DataFrame
    ) -> List[List[str]]:
        """Create different feature subsets for ensemble diversity"""
        all_features = X.columns.tolist()
        subsets = []

        # Subset 1: All features
        subsets.append(all_features)

        # Subset 2: Prior season stats only
        prior_season_features = [f for f in all_features if 'prior' in f.lower() or
                                  any(x in f for x in ['BA', 'ERA', 'WHIP', 'avg'])]
        if prior_season_features:
            subsets.append(prior_season_features)

        # Subset 3: Rolling averages only
        rolling_features = [f for f in all_features if any(x in f for x in
                           ['3d_', '7d_', 'rolling', 'recent'])]
        if rolling_features:
            subsets.append(rolling_features)

        # Subset 4: Pitcher features
        pitcher_features = [f for f in all_features if 'pitcher' in f.lower() or
                           any(x in f for x in ['ERA', 'WHIP', 'K9', 'BB9'])]
        if pitcher_features:
            subsets.append(pitcher_features)

        # Subset 5: Team momentum/form features
        momentum_features = [f for f in all_features if any(x in f for x in
                            ['pythag', 'streak', 'momentum', 'h2h', 'run_diff'])]
        if momentum_features:
            subsets.append(momentum_features)

        return [s for s in subsets if len(s) > 0]

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'GRPOEnsemble':
        """
        Fit the GRPO ensemble.

        Creates and trains multiple models with different feature subsets.
        """
        y = np.array(y).astype(int)

        # Create feature subsets
        self.feature_subsets = self._create_feature_subsets(X)

        # Initialize different model types
        model_configs = [
            ('GRPO_NN', GRPOClassifier(self.config)),
            ('GradientBoost', GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            )),
            ('RandomForest', RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42
            )),
            ('AdaBoost', AdaBoostClassifier(
                n_estimators=100, learning_rate=0.1, random_state=42
            ))
        ]

        self.models = []
        self.model_weights = []
        self.feature_mappings = []

        for i, (name, base_model) in enumerate(model_configs):
            for j, feature_subset in enumerate(self.feature_subsets[:3]):  # Limit subsets
                try:
                    X_subset = X[feature_subset].values if isinstance(X, pd.DataFrame) else X

                    # Clone and fit model
                    if hasattr(base_model, 'config'):
                        model = type(base_model)(base_model.config)
                    else:
                        model = type(base_model)(**base_model.get_params())

                    model.fit(X_subset, y)

                    # Evaluate with cross-validation
                    cv_score = cross_val_score(
                        model, X_subset, y, cv=5, scoring='accuracy'
                    ).mean()

                    self.models.append(model)
                    self.model_weights.append(cv_score)
                    self.feature_mappings.append(feature_subset)

                    print(f"Trained {name} with {len(feature_subset)} features, CV score: {cv_score:.4f}")

                except Exception as e:
                    print(f"Error training {name} with subset {j}: {e}")
                    continue

        # Normalize weights
        total_weight = sum(self.model_weights)
        self.model_weights = [w / total_weight for w in self.model_weights]

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels using weighted voting"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities using weighted averaging"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        ensemble_proba = np.zeros((len(X), 2))

        for model, weight, features in zip(self.models, self.model_weights, self.feature_mappings):
            X_subset = X[features].values if isinstance(X, pd.DataFrame) else X
            proba = model.predict_proba(X_subset)
            ensemble_proba += proba * weight

        return ensemble_proba


class GRPORankingModel(GRPOBaseModel):
    """
    Pairwise ranking model for MLB matchup prediction.

    Instead of predicting absolute probabilities, this model learns
    to rank teams in a matchup, which aligns better with the GRPO
    philosophy of relative comparisons.
    """

    def __init__(self, config: Optional[GRPOConfig] = None):
        super().__init__(config)
        self.model = None

    def _create_pairwise_features(
        self,
        X: np.ndarray,
        feature_names: List[str] = None
    ) -> np.ndarray:
        """
        Create pairwise difference features.

        For each feature pair (home_X, visitor_X), create:
        - Difference: home_X - visitor_X
        - Ratio: home_X / visitor_X
        """
        if feature_names is None:
            # Assume features alternate between home and visitor
            n_features = X.shape[1]
            pairwise_features = []

            # Just use differences
            for i in range(0, n_features - 1, 2):
                diff = X[:, i] - X[:, i + 1]
                pairwise_features.append(diff)

            return np.column_stack(pairwise_features) if pairwise_features else X

        # If we have feature names, match home/visitor pairs
        home_features = [f for f in feature_names if 'home' in f.lower()]
        visitor_features = [f for f in feature_names if 'visitor' in f.lower()]

        pairwise_features = []

        for hf in home_features:
            # Find matching visitor feature
            base_name = hf.lower().replace('home', '').replace('_', '')
            matching_vf = None

            for vf in visitor_features:
                vf_base = vf.lower().replace('visitor', '').replace('_', '')
                if base_name == vf_base:
                    matching_vf = vf
                    break

            if matching_vf:
                h_idx = feature_names.index(hf)
                v_idx = feature_names.index(matching_vf)
                diff = X[:, h_idx] - X[:, v_idx]
                pairwise_features.append(diff)

        return np.column_stack(pairwise_features) if pairwise_features else X

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GRPORankingModel':
        """Fit the ranking model"""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values

        y = np.array(y).astype(int)

        # Create pairwise features
        X_pairwise = self._create_pairwise_features(X, self.feature_names)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_pairwise)

        # Train gradient boosting on pairwise features
        self.model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=4,
            min_samples_split=5,
            random_state=42
        )

        self.model.fit(X_scaled, y)
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict which team wins the matchup"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of home team winning"""
        if isinstance(X, pd.DataFrame):
            X = X.values

        X_pairwise = self._create_pairwise_features(X, self.feature_names)
        X_scaled = self.scaler.transform(X_pairwise)

        return self.model.predict_proba(X_scaled)


class GRPOBettingOptimizer(GRPOBaseModel):
    """
    GRPO model optimized for betting ROI rather than just accuracy.

    This model incorporates betting rewards directly into the training
    objective, learning to maximize expected betting returns.
    """

    def __init__(self, config: Optional[GRPOConfig] = None):
        super().__init__(config)
        self.model = None
        self.calibrator = None

    def _compute_betting_rewards(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        odds: np.ndarray = None
    ) -> np.ndarray:
        """
        Compute rewards based on betting outcomes.

        Rewards consider:
        1. Correct predictions (basic reward)
        2. Confidence-weighted rewards (higher confidence = higher reward)
        3. Odds-based rewards (if betting odds available)
        """
        if odds is None:
            # Assume even odds
            odds = np.full_like(predictions, 2.0)

        rewards = np.zeros_like(predictions)

        for i, (pred, outcome, odd) in enumerate(zip(predictions, outcomes, odds)):
            confidence = abs(pred - 0.5)
            bet_on_home = pred > 0.5

            if (bet_on_home and outcome) or (not bet_on_home and not outcome):
                # Won the bet
                rewards[i] = (odd - 1) * confidence * 2
            else:
                # Lost the bet
                rewards[i] = -confidence * 2

        return rewards

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        odds: np.ndarray = None
    ) -> 'GRPOBettingOptimizer':
        """
        Fit the betting-optimized GRPO model.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Binary outcomes
        odds : np.ndarray
            Betting odds for each game (optional)
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values

        y = np.array(y).astype(int)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Build base model
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=10,
            subsample=0.8,
            random_state=42
        )

        # Initial fit
        self.model.fit(X_scaled, y)

        # Iterative refinement with betting rewards
        for iteration in range(5):
            # Get predictions
            proba = self.model.predict_proba(X_scaled)[:, 1]

            # Compute betting rewards
            betting_rewards = self._compute_betting_rewards(proba, y, odds)

            # Convert to sample weights
            # Games where we make correct confident predictions get higher weight
            sample_weights = np.clip(betting_rewards + 1.5, 0.1, 3.0)

            # Refit with sample weights
            self.model.fit(X_scaled, y, sample_weight=sample_weights)

        # Calibrate probabilities for better betting decisions
        self.calibrator = CalibratedClassifierCV(
            self.model, cv=5, method='isotonic'
        )
        self.calibrator.fit(X_scaled, y)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities"""
        if isinstance(X, pd.DataFrame):
            X = X.values

        X_scaled = self.scaler.transform(X)

        if self.calibrator is not None:
            return self.calibrator.predict_proba(X_scaled)
        else:
            return self.model.predict_proba(X_scaled)

    def get_betting_decisions(
        self,
        X: np.ndarray,
        home_odds: np.ndarray,
        away_odds: np.ndarray,
        min_edge: float = 0.03
    ) -> pd.DataFrame:
        """
        Get betting decisions based on model predictions and odds.

        Returns DataFrame with betting recommendations.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        proba = self.predict_proba(X)[:, 1]

        decisions = []
        for i, (prob, h_odds, a_odds) in enumerate(zip(proba, home_odds, away_odds)):
            home_implied = 1 / h_odds
            away_implied = 1 / a_odds

            home_edge = prob - home_implied
            away_edge = (1 - prob) - away_implied

            if home_edge > min_edge and home_edge > away_edge:
                decisions.append({
                    'game_idx': i,
                    'bet_on': 'home',
                    'model_prob': prob,
                    'implied_prob': home_implied,
                    'edge': home_edge,
                    'odds': h_odds,
                    'expected_value': prob * (h_odds - 1) - (1 - prob)
                })
            elif away_edge > min_edge:
                decisions.append({
                    'game_idx': i,
                    'bet_on': 'away',
                    'model_prob': 1 - prob,
                    'implied_prob': away_implied,
                    'edge': away_edge,
                    'odds': a_odds,
                    'expected_value': (1 - prob) * (a_odds - 1) - prob
                })
            else:
                decisions.append({
                    'game_idx': i,
                    'bet_on': 'no_bet',
                    'model_prob': prob,
                    'implied_prob': max(home_implied, away_implied),
                    'edge': max(home_edge, away_edge),
                    'odds': 0,
                    'expected_value': 0
                })

        return pd.DataFrame(decisions)


def get_best_model(model_results: Dict) -> str:
    """
    Determine the best performing model based on various metrics.
    """
    best_model = None
    best_score = 0

    for name, results in model_results.items():
        # Composite score: 0.4 * accuracy + 0.3 * ROI + 0.3 * CV_score
        score = (
            0.4 * results.get('accuracy', 0) +
            0.3 * (results.get('roi', 0) / 100 + 0.5) +  # Normalize ROI
            0.3 * results.get('cv_score', 0)
        )

        if score > best_score:
            best_score = score
            best_model = name

    return best_model
