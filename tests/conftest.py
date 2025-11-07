"""
Pytest configuration and shared fixtures for backtestbuddy tests.

This file contains shared fixtures and configurations that are available
to all test modules.
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier


# Mark tests automatically based on their location
def pytest_collection_modifyitems(config, items):
    """
    Automatically mark tests based on their file path.
    
    - tests/unit/** get marked with @pytest.mark.unit
    - tests/integration/** get marked with @pytest.mark.integration
    """
    for item in items:
        # Get the path relative to the tests directory
        rel_path = item.fspath.relto(item.session.fspath)
        
        if "unit" in rel_path:
            item.add_marker(pytest.mark.unit)
        if "integration" in rel_path:
            item.add_marker(pytest.mark.integration)
        if "metrics" in rel_path:
            item.add_marker(pytest.mark.metrics)
        if "strategies" in rel_path:
            item.add_marker(pytest.mark.strategies)
        if "backtest" in rel_path:
            item.add_marker(pytest.mark.backtest)
        if "plots" in rel_path:
            item.add_marker(pytest.mark.plots)


# Shared fixtures available to all tests

@pytest.fixture
def sample_deterministic_data():
    """
    Create deterministic sample data for consistent testing across modules.
    
    Returns:
        pd.DataFrame: Sample betting data with known values.
    """
    return pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=10),
        'feature_1': [5, 3, 7, 2, 8, 4, 9, 1, 6, 5],
        'feature_2': [2, 6, 4, 9, 1, 8, 3, 7, 5, 2],
        'odds_1': [2.0, 1.8, 2.5, 1.5, 2.2, 1.9, 2.3, 1.7, 2.1, 2.0],
        'odds_2': [1.8, 2.2, 1.6, 2.8, 1.9, 2.1, 1.7, 2.4, 2.0, 1.8],
        'outcome': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })


@pytest.fixture
def dummy_classifier():
    """
    Create a dummy classifier for testing with fixed random seed.
    
    Returns:
        DummyClassifier: Sklearn dummy classifier for consistent testing.
    """
    return DummyClassifier(strategy="stratified", random_state=42)


@pytest.fixture
def sample_backtest_results():
    """
    Create sample backtest results DataFrame for metrics testing.
    
    Returns:
        pd.DataFrame: Sample backtest results with standard columns.
    """
    return pd.DataFrame({
        'bt_date_column': pd.date_range(start='2023-01-01', periods=10),
        'bt_starting_bankroll': [1000, 1100, 1050, 1200, 1150, 1300, 1250, 1400, 1350, 1500],
        'bt_ending_bankroll': [1100, 1050, 1200, 1150, 1300, 1250, 1400, 1350, 1500, 1500],
        'bt_profit': [100, -50, 150, -50, 150, -50, 150, -50, 150, 0],
        'bt_win': [True, False, True, False, True, False, True, False, True, None],
        'bt_odds': [2.0, 1.8, 2.2, 1.9, 2.1, 1.7, 2.3, 1.6, 2.4, None],
        'bt_stake': [100] * 9 + [0],
        'bt_bet_on': [0, 1, 0, 1, 0, 1, 0, 1, 0, -1]
    })


# Configuration hooks

def pytest_configure(config):
    """
    Configure pytest with custom settings.
    """
    # You can add custom configuration here if needed
    pass


def pytest_report_header(config):
    """
    Add custom header to pytest report.
    """
    return [
        "backtestbuddy Test Suite",
        "=" * 60,
        "Test Structure:",
        "  - tests/unit/        : Unit tests",
        "  - tests/integration/ : Integration tests",
        "=" * 60
    ]

