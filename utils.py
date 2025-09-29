# Utility functions and constants for financial analytics.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
from scipy.stats import norm, shapiro
from scipy.optimize import minimize


PERIODS_PER_YEAR = {"1d": 252, "1wk": 52, "1mo": 12}

def _periods_per_year(interval: str) -> int:
    return PERIODS_PER_YEAR.get(interval, 252)

def _ensure_alpha(alpha_or_pct) -> float:
    a = float(alpha_or_pct)
    return a / 100.0 if a > 1.0 else a

__all__ = [
    "np", "pd", "plt", "yf", "dt", "norm", "shapiro",
    "minimize", "PERIODS_PER_YEAR", "_ensure_alpha",
    "_periods_per_year"
    ]