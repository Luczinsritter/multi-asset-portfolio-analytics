from utils import *

class PriceSimulator:
    '''
    Class to perform forward price simulation via Monte Carlo
    (simple returns, Normal assumption, i.i.d. daily shocks)
    '''

    def __init__(self, price_series: pd.Series):
        '''
        price_series : pd.Series
            Index = dates, Values = prices (float)
        '''
        if not isinstance(price_series, pd.Series):
            raise ValueError('PriceSimulator expects a Pandas Series of prices.')
        self.price_series = price_series.sort_index()
        self.last_price = float(self.price_series.iloc[-1])

    def monte_carlo_paths(self, n_days: int, n_sims: int, seed: int = 22):
        returns = self.price_series.pct_change(1).dropna()
        mean_ret = float(returns.mean())   
        std_dev = float(returns.std())    

        np.random.seed(seed)
        price_paths = np.zeros((n_days, n_sims))

        for sim in range(n_sims):
            cum_return = 1.0
            for day in range(n_days):
                daily_ret = np.random.normal(mean_ret, std_dev)
                cum_return *= (1 + daily_ret)
                price_paths[day, sim] = self.last_price * cum_return

        future_dates = pd.date_range(
            start=self.price_series.index[-1] + pd.Timedelta(days=1),
            periods=n_days,
            freq='D'
        )

        return pd.DataFrame(price_paths, index=future_dates)

    def plot_paths(self, n_days: int, n_sims: int, n_show: int = 50, seed: int = 22):
        '''
        Plot Monte Carlo simulated price paths and mean path
        n_show : number of trajectories to plot for readability
        '''
        price_df = self.monte_carlo_paths(n_days, n_sims, seed)
        mean_path = price_df.mean(axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(price_df.iloc[:, :n_show], color='green', alpha=0.3)
        plt.plot(mean_path, color='darkgreen', linewidth=2, label='Mean Path')
        plt.title(f'Monte Carlo Simulation ({n_days} days, {n_sims} runs)')
        plt.xlabel('Date')
        plt.ylabel('Forecast Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def simulate_pnl_distribution(self, n_days: int, n_sims: int, seed: int = 22, export_excel: str = None):
        '''
        Simulate future price paths and compute final PnL distribution
        export_excel : path to file .xlsx to save results (optional)
        '''
        price_df = self.monte_carlo_paths(n_days, n_sims, seed)
        initial_price = price_df.iloc[0, 0] 
        pnl_df = price_df.iloc[-1] - initial_price  

        stats = {
            'Mean PnL': pnl_df.mean(),
            'Median PnL': pnl_df.median(),
            'Std Dev': pnl_df.std(),
            '5% Quantile': pnl_df.quantile(0.05),
            '95% Quantile': pnl_df.quantile(0.95),
            'Probability of Loss': (pnl_df < 0).mean()
        }
        stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])

        plt.figure(figsize=(8, 5))
        plt.hist(pnl_df, bins=50, color='green', edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', label='Break-even')
        plt.title(f'PnL Distribution after {n_days} days ({n_sims} simulations)')
        plt.xlabel('PnL')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        if export_excel:
            with pd.ExcelWriter(export_excel) as writer:
                pnl_df.to_frame(name='Final_PnL').to_excel(writer, sheet_name='PnL_Distribution')
                stats_df.to_excel(writer, sheet_name='PnL_Stats')

        return pnl_df, stats_df

    def simulate_drawdowns(self, n_days: int, n_sims: int, seed: int = 22, export_excel: str = None):
        '''
        Simulate future price paths and compute drawdowns
        export_excel : same as before
        '''
        price_df = self.monte_carlo_paths(n_days, n_sims, seed)

        dd_df = price_df / price_df.cummax() - 1
        max_dd_per_sim = dd_df.min(axis=0) 
        mean_dd_curve = dd_df.mean(axis=1)  

        plt.figure(figsize=(8, 5))
        plt.plot(mean_dd_curve, color='darkorange', lw=2, label='Average Drawdown')
        plt.title(f'Average Drawdown over {n_days} days ({n_sims} simulations)')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

        if export_excel:
            with pd.ExcelWriter(export_excel) as writer:
                dd_df.to_excel(writer, sheet_name='Drawdown_Trajectories')
                max_dd_per_sim.to_frame(name='Max_Drawdown').to_excel(writer, sheet_name='Max_Drawdowns')
                mean_dd_curve.to_frame(name='Average_Drawdown').to_excel(writer, sheet_name='Average_Drawdown')

        return dd_df, max_dd_per_sim, mean_dd_curve
