"""
GRPO Models for MLB Prediction

This package provides Group Relative Policy Optimization (GRPO) models
adapted for MLB game outcome prediction and betting optimization.

Modules:
--------
- feature_engineering: Enhanced feature engineering with advanced MLB stats
- grpo_models: GRPO model implementations (Classifier, Ensemble, Ranking, Betting)
- betting_integration: Betting odds, ROI calculation, and bankroll management
- training_pipeline: Complete training and evaluation pipeline

Example Usage:
--------------
from grpo_models import (
    MLBFeatureEngineer,
    GRPOClassifier,
    GRPOEnsemble,
    BettingAnalyzer,
    run_full_pipeline
)

# Run the complete pipeline
results = run_full_pipeline(
    game_path='cleaned_game_df.csv',
    batting_path='cleaned_batting_df.csv',
    pitching_path='cleaned_pitching_df.csv'
)

# Or train individual models
model = GRPOEnsemble()
model.fit(X_train, y_train)
predictions = model.predict_proba(X_test)
"""

from .feature_engineering import (
    MLBFeatureEngineer,
    FeatureConfig,
    get_feature_columns
)

from .grpo_models import (
    GRPOConfig,
    GRPOClassifier,
    GRPOEnsemble,
    GRPORankingModel,
    GRPOBettingOptimizer,
    get_best_model
)

from .betting_integration import (
    BettingConfig,
    OddsConverter,
    BettingAnalyzer,
    ProbabilityCalibrator,
    ROIAnalyzer,
    add_synthetic_odds,
    generate_betting_report
)

from .training_pipeline import (
    TrainingConfig,
    DataPreparer,
    ModelTrainer,
    BettingEvaluator,
    ResultsReporter,
    run_full_pipeline,
    quick_train_evaluate
)

__version__ = '1.0.0'
__author__ = 'MLB Prediction Project'

__all__ = [
    # Feature Engineering
    'MLBFeatureEngineer',
    'FeatureConfig',
    'get_feature_columns',

    # GRPO Models
    'GRPOConfig',
    'GRPOClassifier',
    'GRPOEnsemble',
    'GRPORankingModel',
    'GRPOBettingOptimizer',
    'get_best_model',

    # Betting Integration
    'BettingConfig',
    'OddsConverter',
    'BettingAnalyzer',
    'ProbabilityCalibrator',
    'ROIAnalyzer',
    'add_synthetic_odds',
    'generate_betting_report',

    # Training Pipeline
    'TrainingConfig',
    'DataPreparer',
    'ModelTrainer',
    'BettingEvaluator',
    'ResultsReporter',
    'run_full_pipeline',
    'quick_train_evaluate'
]
