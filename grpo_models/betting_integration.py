"""
Betting Integration Module for MLB GRPO Models

This module provides comprehensive betting analysis including:
- Betting odds conversion and normalization
- ROI calculations
- Kelly Criterion betting sizing
- Probability calibration
- Expected value analysis
- Bankroll management simulation

Author: MLB Prediction Project
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class OddsFormat(Enum):
    """Supported odds formats"""
    AMERICAN = "american"
    DECIMAL = "decimal"
    FRACTIONAL = "fractional"
    IMPLIED_PROB = "implied_probability"


@dataclass
class BettingConfig:
    """Configuration for betting simulation"""
    initial_bankroll: float = 10000.0
    bet_sizing: str = "kelly"  # "kelly", "half_kelly", "flat", "proportional"
    flat_bet_pct: float = 0.02  # 2% of bankroll for flat betting
    max_bet_pct: float = 0.10  # Maximum 10% of bankroll per bet
    min_edge_threshold: float = 0.02  # Minimum 2% edge to place bet
    kelly_fraction: float = 0.5  # Half-Kelly by default for safety


class OddsConverter:
    """Convert between different odds formats"""

    @staticmethod
    def american_to_decimal(american_odds: float) -> float:
        """Convert American odds to decimal odds"""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1

    @staticmethod
    def decimal_to_american(decimal_odds: float) -> float:
        """Convert decimal odds to American odds"""
        if decimal_odds >= 2.0:
            return (decimal_odds - 1) * 100
        else:
            return -100 / (decimal_odds - 1)

    @staticmethod
    def decimal_to_implied_prob(decimal_odds: float) -> float:
        """Convert decimal odds to implied probability"""
        return 1 / decimal_odds if decimal_odds > 0 else 0

    @staticmethod
    def american_to_implied_prob(american_odds: float) -> float:
        """Convert American odds to implied probability"""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)

    @staticmethod
    def implied_prob_to_decimal(prob: float) -> float:
        """Convert implied probability to decimal odds"""
        return 1 / prob if prob > 0 else float('inf')

    @staticmethod
    def implied_prob_to_american(prob: float) -> float:
        """Convert implied probability to American odds"""
        if prob <= 0 or prob >= 1:
            return 0
        if prob >= 0.5:
            return -100 * prob / (1 - prob)
        else:
            return 100 * (1 - prob) / prob

    @staticmethod
    def remove_vig(home_prob: float, away_prob: float) -> Tuple[float, float]:
        """
        Remove the vigorish (vig) from implied probabilities.

        The sum of implied probabilities typically exceeds 100% due to the vig.
        This normalizes them to sum to 100%.
        """
        total = home_prob + away_prob
        return home_prob / total, away_prob / total


class BettingAnalyzer:
    """
    Analyze betting outcomes and calculate ROI.

    This class provides methods for:
    - Calculating expected value
    - Computing ROI across a betting history
    - Kelly Criterion bet sizing
    - Profit/loss tracking
    """

    def __init__(self, config: Optional[BettingConfig] = None):
        self.config = config or BettingConfig()
        self.converter = OddsConverter()
        self.betting_history = []

    def calculate_expected_value(
        self,
        model_prob: float,
        decimal_odds: float
    ) -> float:
        """
        Calculate expected value of a bet.

        EV = (probability of winning * potential profit) - (probability of losing * stake)

        Parameters:
        -----------
        model_prob : float
            Model's predicted probability of winning (0-1)
        decimal_odds : float
            Decimal odds offered by the bookmaker

        Returns:
        --------
        float
            Expected value as a percentage of the stake
        """
        potential_profit = decimal_odds - 1
        ev = (model_prob * potential_profit) - ((1 - model_prob) * 1)
        return ev

    def calculate_edge(
        self,
        model_prob: float,
        implied_prob: float
    ) -> float:
        """
        Calculate the edge (advantage) over the bookmaker.

        Edge = Model Probability - Implied Probability

        Parameters:
        -----------
        model_prob : float
            Model's predicted probability
        implied_prob : float
            Bookmaker's implied probability

        Returns:
        --------
        float
            Edge as a decimal (e.g., 0.05 = 5% edge)
        """
        return model_prob - implied_prob

    def kelly_criterion(
        self,
        model_prob: float,
        decimal_odds: float,
        fraction: float = None
    ) -> float:
        """
        Calculate Kelly Criterion bet size.

        Kelly % = (bp - q) / b
        where:
        - b = decimal odds - 1 (potential profit per unit staked)
        - p = probability of winning
        - q = probability of losing (1 - p)

        Parameters:
        -----------
        model_prob : float
            Model's predicted probability of winning
        decimal_odds : float
            Decimal odds offered
        fraction : float
            Fraction of Kelly to use (e.g., 0.5 for half-Kelly)

        Returns:
        --------
        float
            Recommended bet size as fraction of bankroll
        """
        if fraction is None:
            fraction = self.config.kelly_fraction

        b = decimal_odds - 1  # Potential profit per unit
        p = model_prob
        q = 1 - model_prob

        kelly = (b * p - q) / b

        # Apply fractional Kelly
        kelly = kelly * fraction

        # Ensure non-negative and capped at max bet
        kelly = max(0, min(kelly, self.config.max_bet_pct))

        return kelly

    def calculate_bet_size(
        self,
        model_prob: float,
        decimal_odds: float,
        current_bankroll: float
    ) -> float:
        """
        Calculate recommended bet size based on configuration.

        Parameters:
        -----------
        model_prob : float
            Model's predicted probability
        decimal_odds : float
            Decimal odds offered
        current_bankroll : float
            Current bankroll amount

        Returns:
        --------
        float
            Recommended bet amount
        """
        implied_prob = self.converter.decimal_to_implied_prob(decimal_odds)
        edge = self.calculate_edge(model_prob, implied_prob)

        # Don't bet if edge is below threshold
        if edge < self.config.min_edge_threshold:
            return 0.0

        if self.config.bet_sizing == "kelly":
            pct = self.kelly_criterion(model_prob, decimal_odds)
        elif self.config.bet_sizing == "half_kelly":
            pct = self.kelly_criterion(model_prob, decimal_odds, fraction=0.5)
        elif self.config.bet_sizing == "flat":
            pct = self.config.flat_bet_pct
        elif self.config.bet_sizing == "proportional":
            # Bet proportional to edge
            pct = min(edge * 2, self.config.max_bet_pct)
        else:
            pct = self.config.flat_bet_pct

        return current_bankroll * pct

    def simulate_betting_season(
        self,
        predictions: pd.DataFrame,
        prob_column: str = 'home_win_prob',
        home_odds_column: str = 'home_odds_decimal',
        away_odds_column: str = 'away_odds_decimal',
        result_column: str = 'Home_team_won?'
    ) -> Dict:
        """
        Simulate betting over a season based on model predictions.

        Parameters:
        -----------
        predictions : pd.DataFrame
            DataFrame with predictions and odds
        prob_column : str
            Column name for home team win probability
        home_odds_column : str
            Column name for home team decimal odds
        away_odds_column : str
            Column name for away team decimal odds
        result_column : str
            Column name for actual result

        Returns:
        --------
        Dict
            Dictionary containing simulation results
        """
        bankroll = self.config.initial_bankroll
        bankroll_history = [bankroll]
        bet_history = []

        total_bets = 0
        winning_bets = 0
        total_wagered = 0
        total_profit = 0

        for idx, row in predictions.iterrows():
            home_prob = row[prob_column]
            home_odds = row.get(home_odds_column, 2.0)  # Default to even odds
            away_odds = row.get(away_odds_column, 2.0)
            actual_result = row[result_column]

            # Calculate implied probabilities
            home_implied = self.converter.decimal_to_implied_prob(home_odds)
            away_implied = self.converter.decimal_to_implied_prob(away_odds)

            # Remove vig for fair comparison
            home_fair, away_fair = self.converter.remove_vig(home_implied, away_implied)

            # Decide whether to bet on home or away
            home_edge = home_prob - home_fair
            away_edge = (1 - home_prob) - away_fair

            bet_on_home = home_edge > away_edge and home_edge > self.config.min_edge_threshold
            bet_on_away = away_edge > home_edge and away_edge > self.config.min_edge_threshold

            if bet_on_home:
                bet_amount = self.calculate_bet_size(home_prob, home_odds, bankroll)
                if bet_amount > 0:
                    total_bets += 1
                    total_wagered += bet_amount

                    if actual_result:  # Home team won
                        profit = bet_amount * (home_odds - 1)
                        winning_bets += 1
                    else:
                        profit = -bet_amount

                    bankroll += profit
                    total_profit += profit

                    bet_history.append({
                        'game_idx': idx,
                        'bet_type': 'home',
                        'bet_amount': bet_amount,
                        'odds': home_odds,
                        'edge': home_edge,
                        'won': actual_result,
                        'profit': profit,
                        'bankroll': bankroll
                    })

            elif bet_on_away:
                away_prob = 1 - home_prob
                bet_amount = self.calculate_bet_size(away_prob, away_odds, bankroll)
                if bet_amount > 0:
                    total_bets += 1
                    total_wagered += bet_amount

                    if not actual_result:  # Away team won
                        profit = bet_amount * (away_odds - 1)
                        winning_bets += 1
                    else:
                        profit = -bet_amount

                    bankroll += profit
                    total_profit += profit

                    bet_history.append({
                        'game_idx': idx,
                        'bet_type': 'away',
                        'bet_amount': bet_amount,
                        'odds': away_odds,
                        'edge': away_edge,
                        'won': not actual_result,
                        'profit': profit,
                        'bankroll': bankroll
                    })

            bankroll_history.append(bankroll)

        # Calculate summary statistics
        roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0
        win_rate = (winning_bets / total_bets * 100) if total_bets > 0 else 0

        return {
            'final_bankroll': bankroll,
            'total_profit': total_profit,
            'roi_pct': roi,
            'total_bets': total_bets,
            'winning_bets': winning_bets,
            'win_rate_pct': win_rate,
            'total_wagered': total_wagered,
            'bankroll_history': bankroll_history,
            'bet_history': bet_history,
            'max_drawdown': self._calculate_max_drawdown(bankroll_history)
        }

    def _calculate_max_drawdown(self, bankroll_history: List[float]) -> float:
        """Calculate maximum drawdown from bankroll history"""
        peak = bankroll_history[0]
        max_dd = 0

        for value in bankroll_history:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        return max_dd * 100  # Return as percentage


class ProbabilityCalibrator:
    """
    Calibrate model probabilities for better betting decisions.

    Well-calibrated probabilities are essential for profitable betting.
    This class provides methods to assess and improve calibration.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.calibration_mapping = None

    def assess_calibration(
        self,
        predicted_probs: np.ndarray,
        actual_outcomes: np.ndarray
    ) -> Dict:
        """
        Assess probability calibration using reliability diagram data.

        Parameters:
        -----------
        predicted_probs : np.ndarray
            Predicted probabilities
        actual_outcomes : np.ndarray
            Actual binary outcomes (0 or 1)

        Returns:
        --------
        Dict
            Calibration assessment including ECE and MCE
        """
        bins = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(predicted_probs, bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        bin_accuracy = []
        bin_confidence = []
        bin_counts = []

        for i in range(self.n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_accuracy.append(actual_outcomes[mask].mean())
                bin_confidence.append(predicted_probs[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_accuracy.append(0)
                bin_confidence.append((bins[i] + bins[i + 1]) / 2)
                bin_counts.append(0)

        # Calculate Expected Calibration Error (ECE)
        total_samples = len(predicted_probs)
        ece = sum(
            (count / total_samples) * abs(acc - conf)
            for acc, conf, count in zip(bin_accuracy, bin_confidence, bin_counts)
        )

        # Calculate Maximum Calibration Error (MCE)
        mce = max(
            abs(acc - conf)
            for acc, conf in zip(bin_accuracy, bin_confidence)
        )

        return {
            'ece': ece,
            'mce': mce,
            'bin_accuracy': bin_accuracy,
            'bin_confidence': bin_confidence,
            'bin_counts': bin_counts
        }

    def fit_platt_scaling(
        self,
        predicted_probs: np.ndarray,
        actual_outcomes: np.ndarray
    ) -> 'ProbabilityCalibrator':
        """
        Fit Platt scaling for probability calibration.

        Platt scaling fits a logistic regression to calibrate probabilities:
        p_calibrated = 1 / (1 + exp(A * log(p/(1-p)) + B))
        """
        from scipy.optimize import minimize

        def neg_log_likelihood(params, probs, outcomes):
            A, B = params
            # Avoid log(0)
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            log_odds = np.log(probs / (1 - probs))
            calibrated = 1 / (1 + np.exp(A * log_odds + B))
            calibrated = np.clip(calibrated, 1e-10, 1 - 1e-10)

            ll = outcomes * np.log(calibrated) + (1 - outcomes) * np.log(1 - calibrated)
            return -ll.sum()

        # Initial parameters
        x0 = [1.0, 0.0]
        result = minimize(neg_log_likelihood, x0, args=(predicted_probs, actual_outcomes))

        self.platt_A, self.platt_B = result.x
        return self

    def calibrate_probabilities(
        self,
        predicted_probs: np.ndarray
    ) -> np.ndarray:
        """
        Calibrate probabilities using fitted Platt scaling.
        """
        if not hasattr(self, 'platt_A'):
            raise ValueError("Must fit Platt scaling first using fit_platt_scaling()")

        probs = np.clip(predicted_probs, 1e-10, 1 - 1e-10)
        log_odds = np.log(probs / (1 - probs))
        calibrated = 1 / (1 + np.exp(self.platt_A * log_odds + self.platt_B))

        return calibrated


class ROIAnalyzer:
    """
    Comprehensive ROI analysis for betting strategies.
    """

    def __init__(self):
        self.results = {}

    def analyze_roi_by_confidence(
        self,
        predictions: pd.DataFrame,
        prob_column: str = 'home_win_prob',
        result_column: str = 'Home_team_won?',
        home_odds_column: str = 'home_odds_decimal',
        confidence_bins: List[float] = None
    ) -> pd.DataFrame:
        """
        Analyze ROI by confidence level buckets.
        """
        if confidence_bins is None:
            confidence_bins = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 1.0]

        df = predictions.copy()
        df['confidence'] = df[prob_column].apply(lambda x: max(x, 1 - x))
        df['conf_bin'] = pd.cut(df['confidence'], bins=confidence_bins)

        results = []
        for conf_bin in df['conf_bin'].unique():
            if pd.isna(conf_bin):
                continue

            bin_data = df[df['conf_bin'] == conf_bin]
            n_games = len(bin_data)

            # Calculate accuracy
            correct_predictions = (
                (bin_data[prob_column] > 0.5) == bin_data[result_column]
            ).sum()
            accuracy = correct_predictions / n_games if n_games > 0 else 0

            # Calculate theoretical ROI (assuming even odds)
            theoretical_roi = (accuracy - 0.5) * 2 * 100 - 10  # Assuming 10% vig

            results.append({
                'confidence_range': str(conf_bin),
                'n_games': n_games,
                'accuracy': accuracy,
                'theoretical_roi_pct': theoretical_roi
            })

        return pd.DataFrame(results)

    def analyze_roi_by_edge(
        self,
        simulation_results: Dict
    ) -> pd.DataFrame:
        """
        Analyze ROI by edge size.
        """
        bet_df = pd.DataFrame(simulation_results['bet_history'])

        if len(bet_df) == 0:
            return pd.DataFrame()

        # Bin by edge
        edge_bins = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 1.0]
        bet_df['edge_bin'] = pd.cut(bet_df['edge'], bins=edge_bins)

        results = []
        for edge_bin in bet_df['edge_bin'].unique():
            if pd.isna(edge_bin):
                continue

            bin_data = bet_df[bet_df['edge_bin'] == edge_bin]
            n_bets = len(bin_data)
            wins = bin_data['won'].sum()
            total_wagered = bin_data['bet_amount'].sum()
            total_profit = bin_data['profit'].sum()
            roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0

            results.append({
                'edge_range': str(edge_bin),
                'n_bets': n_bets,
                'win_rate': wins / n_bets if n_bets > 0 else 0,
                'total_wagered': total_wagered,
                'total_profit': total_profit,
                'roi_pct': roi
            })

        return pd.DataFrame(results)


def add_synthetic_odds(
    game_df: pd.DataFrame,
    home_win_prob_column: str = None,
    base_vig: float = 0.05
) -> pd.DataFrame:
    """
    Add synthetic betting odds to game data for historical analysis.

    Since historical betting odds may not be available, this function
    creates reasonable synthetic odds based on team strength indicators.

    Parameters:
    -----------
    game_df : pd.DataFrame
        Game data with team performance metrics
    home_win_prob_column : str
        Column with estimated home win probability (if available)
    base_vig : float
        Vigorish (vig) to add to odds (default 5%)

    Returns:
    --------
    pd.DataFrame
        DataFrame with added odds columns
    """
    df = game_df.copy()

    # If no probability column provided, estimate from available features
    if home_win_prob_column and home_win_prob_column in df.columns:
        base_prob = df[home_win_prob_column]
    else:
        # Use home field advantage as baseline (typically ~54% for MLB)
        base_prob = 0.54

        # Adjust based on available features
        if 'home_pythag' in df.columns and 'visitor_pythag' in df.columns:
            # Use Pythagorean expectation difference
            prob_diff = df['home_pythag'] - df['visitor_pythag']
            base_prob = 0.54 + prob_diff * 0.5  # Scale the difference
            base_prob = base_prob.clip(0.25, 0.75)  # Reasonable bounds

    # Add vig to create realistic odds
    home_implied = base_prob + base_vig / 2
    away_implied = (1 - base_prob) + base_vig / 2

    # Convert to decimal odds
    df['home_odds_decimal'] = 1 / home_implied
    df['away_odds_decimal'] = 1 / away_implied

    # Also add American odds for reference
    converter = OddsConverter()
    df['home_odds_american'] = df['home_odds_decimal'].apply(
        converter.decimal_to_american
    )
    df['away_odds_american'] = df['away_odds_decimal'].apply(
        converter.decimal_to_american
    )

    return df


def generate_betting_report(
    simulation_results: Dict,
    output_format: str = 'text'
) -> str:
    """
    Generate a comprehensive betting report.

    Parameters:
    -----------
    simulation_results : Dict
        Results from BettingAnalyzer.simulate_betting_season()
    output_format : str
        'text' or 'markdown'

    Returns:
    --------
    str
        Formatted report string
    """
    if output_format == 'markdown':
        report = f"""
# MLB Betting Simulation Report

## Summary Statistics
| Metric | Value |
|--------|-------|
| Initial Bankroll | $10,000.00 |
| Final Bankroll | ${simulation_results['final_bankroll']:,.2f} |
| Total Profit/Loss | ${simulation_results['total_profit']:,.2f} |
| ROI | {simulation_results['roi_pct']:.2f}% |
| Total Bets Placed | {simulation_results['total_bets']} |
| Winning Bets | {simulation_results['winning_bets']} |
| Win Rate | {simulation_results['win_rate_pct']:.1f}% |
| Total Amount Wagered | ${simulation_results['total_wagered']:,.2f} |
| Maximum Drawdown | {simulation_results['max_drawdown']:.1f}% |

## Performance Analysis
- Average bet size: ${simulation_results['total_wagered']/max(simulation_results['total_bets'],1):,.2f}
- Profit per bet: ${simulation_results['total_profit']/max(simulation_results['total_bets'],1):,.2f}
"""
    else:
        report = f"""
========================================
MLB BETTING SIMULATION REPORT
========================================

SUMMARY STATISTICS
------------------
Initial Bankroll:     $10,000.00
Final Bankroll:       ${simulation_results['final_bankroll']:,.2f}
Total Profit/Loss:    ${simulation_results['total_profit']:,.2f}
ROI:                  {simulation_results['roi_pct']:.2f}%

BETTING ACTIVITY
----------------
Total Bets Placed:    {simulation_results['total_bets']}
Winning Bets:         {simulation_results['winning_bets']}
Win Rate:             {simulation_results['win_rate_pct']:.1f}%
Total Amount Wagered: ${simulation_results['total_wagered']:,.2f}

RISK METRICS
------------
Maximum Drawdown:     {simulation_results['max_drawdown']:.1f}%
Average Bet Size:     ${simulation_results['total_wagered']/max(simulation_results['total_bets'],1):,.2f}
Profit per Bet:       ${simulation_results['total_profit']/max(simulation_results['total_bets'],1):,.2f}
========================================
"""

    return report
