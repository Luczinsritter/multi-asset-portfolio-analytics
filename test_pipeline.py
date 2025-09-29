# Demo script

from portfolio import *
from risk_analysis import *
from simulation import *


TICKERS = ['AAPL', 'MSFT', 'TLT']
CAPITAL = 100_000
ALLOW_SHORT = False
RISK_FREE_RATE = 0.03
confidence_level = 0.95
HORIZON_DAYS = 10
N_SIMS = 50_000
MC_FORWARD_DAYS = 100
MC_FORWARD_SIMS = 1_000
END_DATE = pd.Timestamp.today() 
LOOKBACK_DAYS = 365 * 2
INTERVAL = '1d' 


print("\n=== Portfolio Optimisation ===")
portfolio = Portfolio(
    tickers_list = TICKERS,
    end_date = END_DATE,
    duration = LOOKBACK_DAYS,
    chosen_interval = INTERVAL,
    capital = CAPITAL,
    allow_short = ALLOW_SHORT
)

portfolio.efficient_frontier_with_alpha(alpha=0.65)
portfolio_df = portfolio.optimized_pf()

print("\n=== Risk Analysis ===")
risk = RiskAnalysis(portfolio_df)

risk.var_hist(confidence_level=confidence_level)

risk.var_param(confidence_level=confidence_level, days=HORIZON_DAYS)

risk.var_montecarlo(confidence_level=confidence_level, horizon=HORIZON_DAYS, n_sims=N_SIMS)


print("\n=== Forward Simulation ===")
simulator = PriceSimulator(portfolio_df['Total_Portfolio_Value'])

simulator.simulate_pnl_distribution(
    n_days=MC_FORWARD_DAYS,
    n_sims=MC_FORWARD_SIMS,
    # export_excel="pnl_results.xlsx"
)

simulator.simulate_drawdowns(
    n_days=MC_FORWARD_DAYS,
    n_sims=MC_FORWARD_SIMS,
    # export_excel="drawdown_results.xlsx"
)