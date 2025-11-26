"""
Training and Evaluation Pipeline for MLB GRPO Models

This module provides:
- Data preparation and splitting by season
- Model training and evaluation
- Cross-validation and hyperparameter tuning
- Performance metrics and visualization
- Model comparison and selection

Author: MLB Prediction Project
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, log_loss, confusion_matrix,
    classification_report
)
from sklearn.calibration import calibration_curve

# Local imports
from .feature_engineering import MLBFeatureEngineer, FeatureConfig, get_feature_columns
from .grpo_models import (
    GRPOClassifier, GRPOEnsemble, GRPORankingModel,
    GRPOBettingOptimizer, GRPOConfig
)
from .betting_integration import (
    BettingAnalyzer, BettingConfig, ProbabilityCalibrator,
    ROIAnalyzer, add_synthetic_odds, generate_betting_report
)


@dataclass
class TrainingConfig:
    """Configuration for training pipeline"""
    # Data split
    train_years: List[int] = field(default_factory=lambda: list(range(2010, 2017)))
    val_years: List[int] = field(default_factory=lambda: [2017])
    test_years: List[int] = field(default_factory=lambda: [2018, 2019])

    # Training parameters
    n_cv_folds: int = 5
    random_state: int = 42

    # Model selection
    models_to_train: List[str] = field(default_factory=lambda: [
        'grpo_classifier', 'grpo_ensemble', 'grpo_ranking', 'grpo_betting'
    ])

    # Feature engineering
    include_h2h: bool = True
    include_situational: bool = True
    include_pitcher_rolling: bool = True
    include_advanced: bool = True

    # Output
    save_models: bool = True
    output_dir: str = './models'


class DataPreparer:
    """Prepare data for training and evaluation"""

    def __init__(self, config: TrainingConfig):
        self.config = config

    def load_data(
        self,
        game_path: str,
        batting_path: str,
        pitching_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load data from CSV files"""
        game_df = pd.read_csv(game_path)
        batting_df = pd.read_csv(batting_path)
        pitching_df = pd.read_csv(pitching_path)

        # Clean up index columns if present
        for df in [game_df, batting_df, pitching_df]:
            if 'Unnamed: 0' in df.columns:
                df.drop('Unnamed: 0', axis=1, inplace=True)

        return game_df, batting_df, pitching_df

    def prepare_features(
        self,
        game_df: pd.DataFrame,
        batting_df: pd.DataFrame,
        pitching_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Run feature engineering pipeline"""
        feature_engineer = MLBFeatureEngineer()

        df = feature_engineer.engineer_all_features(
            game_df, batting_df, pitching_df,
            include_h2h=self.config.include_h2h,
            include_situational=self.config.include_situational,
            include_pitcher_rolling=self.config.include_pitcher_rolling,
            include_advanced=self.config.include_advanced
        )

        return df

    def train_test_split_by_season(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str = 'Home_team_won?'
    ) -> Dict[str, Tuple[pd.DataFrame, np.ndarray]]:
        """
        Split data by season for train/val/test.

        Returns dictionary with train, val, test sets.
        """
        # Ensure we have a year column
        if 'current_year' not in df.columns and 'New_Date' in df.columns:
            df['current_year'] = pd.to_datetime(df['New_Date']).dt.year

        # Create masks
        train_mask = df['current_year'].isin(self.config.train_years)
        val_mask = df['current_year'].isin(self.config.val_years)
        test_mask = df['current_year'].isin(self.config.test_years)

        # Prepare X and y
        def get_xy(mask):
            X = df.loc[mask, feature_columns].copy()
            y = df.loc[mask, target_column].values.astype(int)
            # Convert to numeric and fill NaN
            X = X.apply(pd.to_numeric, errors='coerce').fillna(-1)
            return X, y

        splits = {
            'train': get_xy(train_mask),
            'val': get_xy(val_mask),
            'test': get_xy(test_mask)
        }

        return splits

    def create_time_series_cv(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_splits: int = 5
    ) -> TimeSeriesSplit:
        """Create time series cross-validation splitter"""
        return TimeSeriesSplit(n_splits=n_splits)


class ModelTrainer:
    """Train and evaluate GRPO models"""

    def __init__(
        self,
        training_config: TrainingConfig,
        grpo_config: Optional[GRPOConfig] = None
    ):
        self.training_config = training_config
        self.grpo_config = grpo_config or GRPOConfig()
        self.models = {}
        self.results = {}

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame = None,
        y_val: np.ndarray = None
    ) -> Dict[str, Any]:
        """Train all specified models"""
        models_to_train = self.training_config.models_to_train
        trained_models = {}

        for model_name in models_to_train:
            print(f"\nTraining {model_name}...")

            try:
                if model_name == 'grpo_classifier':
                    model = GRPOClassifier(self.grpo_config)
                    model.fit(X_train, y_train)

                elif model_name == 'grpo_ensemble':
                    model = GRPOEnsemble(self.grpo_config)
                    model.fit(X_train, y_train)

                elif model_name == 'grpo_ranking':
                    model = GRPORankingModel(self.grpo_config)
                    model.fit(X_train, y_train)

                elif model_name == 'grpo_betting':
                    model = GRPOBettingOptimizer(self.grpo_config)
                    model.fit(X_train, y_train)

                else:
                    print(f"Unknown model type: {model_name}")
                    continue

                trained_models[model_name] = model
                print(f"Successfully trained {model_name}")

            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue

        self.models = trained_models
        return trained_models

    def evaluate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: np.ndarray,
        model_name: str = "model"
    ) -> Dict[str, float]:
        """Evaluate a single model"""
        # Get predictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba),
            'brier_score': brier_score_loss(y, y_proba),
            'log_loss': log_loss(y, y_proba)
        }

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        # Classification report
        report = classification_report(y, y_pred, output_dict=True)
        metrics['classification_report'] = report

        return metrics

    def evaluate_all_models(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate all trained models"""
        results = {}

        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            results[model_name] = metrics

            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  Brier Score: {metrics['brier_score']:.4f}")

        self.results = results
        return results

    def cross_validate(
        self,
        model_class: type,
        X: pd.DataFrame,
        y: np.ndarray,
        n_folds: int = 5
    ) -> Dict[str, float]:
        """Perform cross-validation"""
        cv = TimeSeriesSplit(n_splits=n_folds)

        accuracy_scores = []
        roc_scores = []

        for train_idx, val_idx in cv.split(X):
            X_train_cv = X.iloc[train_idx]
            y_train_cv = y[train_idx]
            X_val_cv = X.iloc[val_idx]
            y_val_cv = y[val_idx]

            # Train model
            model = model_class(self.grpo_config)
            model.fit(X_train_cv, y_train_cv)

            # Evaluate
            y_pred = model.predict(X_val_cv)
            y_proba = model.predict_proba(X_val_cv)[:, 1]

            accuracy_scores.append(accuracy_score(y_val_cv, y_pred))
            roc_scores.append(roc_auc_score(y_val_cv, y_proba))

        return {
            'cv_accuracy_mean': np.mean(accuracy_scores),
            'cv_accuracy_std': np.std(accuracy_scores),
            'cv_roc_auc_mean': np.mean(roc_scores),
            'cv_roc_auc_std': np.std(roc_scores)
        }


class BettingEvaluator:
    """Evaluate models for betting performance"""

    def __init__(self, betting_config: Optional[BettingConfig] = None):
        self.config = betting_config or BettingConfig()
        self.analyzer = BettingAnalyzer(self.config)
        self.roi_analyzer = ROIAnalyzer()

    def evaluate_betting_performance(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        df_test: pd.DataFrame = None
    ) -> Dict:
        """
        Evaluate model performance for betting.

        Parameters:
        -----------
        model : Any
            Trained model with predict_proba method
        X_test : pd.DataFrame
            Test features
        y_test : np.ndarray
            Test labels
        df_test : pd.DataFrame
            Full test dataframe (for odds if available)

        Returns:
        --------
        Dict
            Betting simulation results
        """
        # Get predictions
        y_proba = model.predict_proba(X_test)[:, 1]

        # Create predictions DataFrame
        predictions = pd.DataFrame({
            'home_win_prob': y_proba,
            'Home_team_won?': y_test
        })

        # Add synthetic odds if real odds not available
        if df_test is not None and 'home_odds_decimal' in df_test.columns:
            predictions['home_odds_decimal'] = df_test['home_odds_decimal'].values
            predictions['away_odds_decimal'] = df_test['away_odds_decimal'].values
        else:
            predictions = add_synthetic_odds(predictions, 'home_win_prob')

        # Run betting simulation
        results = self.analyzer.simulate_betting_season(predictions)

        return results

    def analyze_by_confidence(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """Analyze performance by confidence level"""
        y_proba = model.predict_proba(X_test)[:, 1]

        predictions = pd.DataFrame({
            'home_win_prob': y_proba,
            'Home_team_won?': y_test
        })

        return self.roi_analyzer.analyze_roi_by_confidence(predictions)


class ResultsReporter:
    """Generate comprehensive reports on model performance"""

    def __init__(self, output_dir: str = './results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_comparison_report(
        self,
        model_results: Dict[str, Dict],
        betting_results: Dict[str, Dict]
    ) -> str:
        """Generate a comparison report of all models"""
        report = []
        report.append("=" * 80)
        report.append("MLB GRPO MODELS COMPARISON REPORT")
        report.append("=" * 80)
        report.append("")

        # Accuracy comparison
        report.append("PREDICTION ACCURACY")
        report.append("-" * 40)
        for model_name, metrics in model_results.items():
            report.append(f"  {model_name}:")
            report.append(f"    Accuracy: {metrics['accuracy']:.4f}")
            report.append(f"    ROC-AUC:  {metrics['roc_auc']:.4f}")
            report.append(f"    F1 Score: {metrics['f1']:.4f}")
            report.append("")

        # Betting performance comparison
        if betting_results:
            report.append("BETTING PERFORMANCE")
            report.append("-" * 40)
            for model_name, results in betting_results.items():
                report.append(f"  {model_name}:")
                report.append(f"    ROI: {results['roi_pct']:.2f}%")
                report.append(f"    Win Rate: {results['win_rate_pct']:.1f}%")
                report.append(f"    Total Profit: ${results['total_profit']:,.2f}")
                report.append(f"    Max Drawdown: {results['max_drawdown']:.1f}%")
                report.append("")

        # Best model selection
        report.append("BEST MODEL SELECTION")
        report.append("-" * 40)

        best_accuracy = max(model_results.items(), key=lambda x: x[1]['accuracy'])
        report.append(f"  Best Accuracy: {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.4f})")

        if betting_results:
            best_roi = max(betting_results.items(), key=lambda x: x[1]['roi_pct'])
            report.append(f"  Best ROI: {best_roi[0]} ({best_roi[1]['roi_pct']:.2f}%)")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def save_results(
        self,
        model_results: Dict,
        betting_results: Dict,
        models: Dict,
        filename_prefix: str = "grpo_results"
    ):
        """Save all results to files"""
        # Save metrics as JSON
        metrics_path = self.output_dir / f"{filename_prefix}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                'model_results': {k: {kk: vv for kk, vv in v.items()
                                      if not isinstance(vv, np.ndarray)}
                                  for k, v in model_results.items()},
                'betting_results': {k: {kk: vv for kk, vv in v.items()
                                        if not isinstance(vv, (list, np.ndarray)) or len(vv) < 100}
                                    for k, v in betting_results.items()}
            }, f, indent=2, default=str)

        # Save models as pickle
        for model_name, model in models.items():
            model_path = self.output_dir / f"{filename_prefix}_{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

        print(f"Results saved to {self.output_dir}")


def run_full_pipeline(
    game_path: str,
    batting_path: str,
    pitching_path: str,
    training_config: Optional[TrainingConfig] = None,
    grpo_config: Optional[GRPOConfig] = None
) -> Dict:
    """
    Run the complete training and evaluation pipeline.

    Parameters:
    -----------
    game_path : str
        Path to game data CSV
    batting_path : str
        Path to batting data CSV
    pitching_path : str
        Path to pitching data CSV
    training_config : TrainingConfig
        Training configuration
    grpo_config : GRPOConfig
        GRPO model configuration

    Returns:
    --------
    Dict
        Complete results including models, metrics, and betting results
    """
    training_config = training_config or TrainingConfig()
    grpo_config = grpo_config or GRPOConfig()

    print("=" * 60)
    print("MLB GRPO MODELS TRAINING PIPELINE")
    print("=" * 60)

    # Step 1: Load and prepare data
    print("\n[1/5] Loading and preparing data...")
    data_preparer = DataPreparer(training_config)
    game_df, batting_df, pitching_df = data_preparer.load_data(
        game_path, batting_path, pitching_path
    )

    # Step 2: Feature engineering
    print("\n[2/5] Running feature engineering...")
    df = data_preparer.prepare_features(game_df, batting_df, pitching_df)

    # Get feature columns
    feature_columns = get_feature_columns(df)
    print(f"  Created {len(feature_columns)} features")

    # Step 3: Split data
    print("\n[3/5] Splitting data by season...")
    splits = data_preparer.train_test_split_by_season(df, feature_columns)

    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']

    print(f"  Train: {len(X_train)} games")
    print(f"  Validation: {len(X_val)} games")
    print(f"  Test: {len(X_test)} games")

    # Step 4: Train models
    print("\n[4/5] Training models...")
    trainer = ModelTrainer(training_config, grpo_config)
    models = trainer.train_all_models(X_train, y_train, X_val, y_val)

    # Step 5: Evaluate models
    print("\n[5/5] Evaluating models...")
    model_results = trainer.evaluate_all_models(X_test, y_test)

    # Betting evaluation
    betting_evaluator = BettingEvaluator()
    betting_results = {}

    for model_name, model in models.items():
        betting_results[model_name] = betting_evaluator.evaluate_betting_performance(
            model, X_test, y_test
        )

    # Generate report
    reporter = ResultsReporter()
    report = reporter.generate_comparison_report(model_results, betting_results)
    print("\n" + report)

    # Save results
    if training_config.save_models:
        reporter.save_results(model_results, betting_results, models)

    return {
        'models': models,
        'model_results': model_results,
        'betting_results': betting_results,
        'feature_columns': feature_columns,
        'data_splits': {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    }


def quick_train_evaluate(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    model_type: str = 'grpo_ensemble'
) -> Tuple[Any, Dict]:
    """
    Quick function to train and evaluate a single model.

    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    model_type : str
        One of: 'grpo_classifier', 'grpo_ensemble', 'grpo_ranking', 'grpo_betting'

    Returns:
    --------
    Tuple[model, metrics]
    """
    config = GRPOConfig()

    if model_type == 'grpo_classifier':
        model = GRPOClassifier(config)
    elif model_type == 'grpo_ensemble':
        model = GRPOEnsemble(config)
    elif model_type == 'grpo_ranking':
        model = GRPORankingModel(config)
    elif model_type == 'grpo_betting':
        model = GRPOBettingOptimizer(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    trainer = ModelTrainer(TrainingConfig(), config)
    metrics = trainer.evaluate_model(model, X_test, y_test)

    return model, metrics
