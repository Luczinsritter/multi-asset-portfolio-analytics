from utils import *

class FinancialData:
    '''
    Class to enable (1) data downloading via Yahoo Finance AND (2) clean the data
    '''

    def __init__(self, ticker: str | list[str], end_date, duration: int, chosen_interval: str):
        self.ticker = ticker
        self.end_date = pd.Timestamp(end_date)
        self.duration = int(duration)
        self.chosen_interval = chosen_interval
        self._df = None  # cache

    def download(self):
        if self._df is not None:
            return self._df.copy()
        delta = dt.timedelta(self.duration)
        start_date = self.end_date - delta
        df = yf.download(
            self.ticker,
            start=start_date,
            end=self.end_date,
            interval=self.chosen_interval,
            auto_adjust=True,
            progress=False
        )
        self._df = self.clean_data(df)
        return self._df.copy()

    @staticmethod
    def clean_data(df):
        if isinstance(df.columns, pd.MultiIndex):
            if "Close" in df.columns.get_level_values(0):
                df = df['Close']
            df.columns = [f"{c}_Close" for c in df.columns]
            return df.sort_index().ffill().dropna(how='any')
        cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
        return df[cols].sort_index().ffill().dropna(how='any')


class ChosenTickers:
    '''
    Class to enable user to choose several Tickers and type of returns to work with
    '''

    def __init__(self, tickers_list: list[str], end_date, duration: int, chosen_interval: str):
        self.tickers_list = tickers_list
        self.end_date = pd.Timestamp(end_date)
        self.duration = int(duration)
        self.chosen_interval = chosen_interval
        self.ppy = _periods_per_year(chosen_interval)
        self._prices = None
        self._simp_returns = None
        self._log_returns = None

    def download_all_close(self):
        if self._prices is not None:
            return self._prices.copy()
        fd = FinancialData(self.tickers_list, self.end_date, self.duration, self.chosen_interval)
        df = fd.download()
        if {'Open', 'High', 'Low', 'Close', 'Volume'}.issubset(set(df.columns)):
            name = f'{self.tickers_list[0]}_Close'
            df = df[['Close']].rename(columns={'Close': name})
        self._prices = df.sort_index().dropna(how='any')
        return self._prices.copy()

    def download_all_simp_returns(self):
        if self._simp_returns is not None:
            return self._simp_returns.copy()
        prices = self.download_all_close()
        simp = prices.pct_change().dropna()
        simp.columns = [c.replace('_Close', '_simp_returns') for c in simp.columns]
        self._simp_returns = simp
        return self._simp_returns.copy()

    def download_all_log_returns(self):
        if self._log_returns is not None:
            return self._log_returns.copy()
        prices = self.download_all_close()
        log = np.log(prices / prices.shift(1)).dropna()
        log.columns = [c.replace('_Close', '_log_returns') for c in log.columns]
        self._log_returns = log
        return self._log_returns.copy()


class Portfolio(ChosenTickers):
    """
    Portfolio class for creating a portfolio object,running optimisation, computing portfolio value series,
    and plotting the efficient frontier
    """

    def __init__(self, tickers_list: list[str], end_date, duration: int, chosen_interval: str,
                 capital: float, allow_short: bool = False):
        super().__init__(tickers_list, end_date, duration, chosen_interval)
        self.capital = float(capital)
        self.allow_short = allow_short
        self.weights = None
        self.value_series = None

    def optimized_pf(self):
        """
        Compute the portfolio value series using the efficient frontier
        """
        if self.weights is None:
            self.efficient_frontier_with_alpha(alpha=0.5)
        prices_df = self.download_all_close()
        ordered_cols = [f"{t}_Close" for t in self.tickers_list]
        prices_df = prices_df[ordered_cols]
        w = np.array([self.weights[f"{t}_log_returns"] if f"{t}_log_returns" in self.weights else self.weights[t]
                      for t in self.tickers_list])
        norm_px = prices_df / prices_df.iloc[0]
        total_value = self.capital * (norm_px.to_numpy() @ w)
        weighted_df = prices_df.copy()
        weighted_df['Total_Portfolio_Value'] = total_value
        self.value_series = weighted_df
        return weighted_df

    def efficient_frontier_with_alpha(self, num_points: int = 50, use_log_returns=True,
                                   risk_free_rate_annual=0.0, alpha: float = 1.0):
        """
        Compute the efficient frontier and select a portfolio based on alpha:
        alpha = 1.0 -> Max Sharpe
        alpha = 0.0 -> Min Volatility
        0 < alpha < 1 -> Compromise between Sharpe and low volatility
        """
        R = self.download_all_log_returns() if use_log_returns else self.download_all_simp_returns()
        mu_per = R.mean().to_numpy()
        cov_per = R.cov().to_numpy()
        ppy = self.ppy
        mu_ann_assets = mu_per * ppy
        lo, hi = np.quantile(mu_ann_assets, [0.1, 0.9])
        targets_ann = np.linspace(lo, hi, num_points)
        n = len(mu_per)
        bounds = None if self.allow_short else tuple((0.0, 1.0) for _ in range(n))
        risk_ann, ret_ann, sharpe_ratios, weights_list = [], [], [], []
        for m_target_ann in targets_ann:
            m_target_per = m_target_ann / ppy
            cons = (
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'eq', 'fun': lambda w, mu=mu_per: w @ mu - m_target_per}
            )

            def port_var(w):
                return w @ cov_per @ w
            res = minimize(port_var, x0=np.full(n, 1/n), method='SLSQP', bounds=bounds, constraints=cons)
            if res.success:
                w = res.x
                sigma_ann = np.sqrt(max(w @ cov_per @ w, 1e-18)) * np.sqrt(ppy)
                ret_a = w @ mu_per * ppy
                sharpe = (ret_a - risk_free_rate_annual) / sigma_ann if sigma_ann > 0 else -np.inf
                risk_ann.append(sigma_ann)
                ret_ann.append(ret_a)
                sharpe_ratios.append(sharpe)
                weights_list.append(w)

        max_sharpe_val = max(sharpe_ratios)
        max_vol = max(risk_ann)
        scores = [
            alpha * (sh / max_sharpe_val) - (1 - alpha) * (vol / max_vol)
            for sh, vol in zip(sharpe_ratios, risk_ann)
        ]
        best_idx = int(np.argmax(scores))
        self.weights = dict(zip(R.columns, weights_list[best_idx]))

        plt.figure(figsize=(7, 5))
        plt.plot(risk_ann, ret_ann, color='darkgreen', lw=2, label='Efficient Frontier')
        plt.scatter(risk_ann[best_idx], ret_ann[best_idx], color='red', s=80,
                    label=f'Alpha={alpha:.2f} Portfolio')
        plt.xlabel('Annualized Volatility')
        plt.ylabel('Annualized Return')
        plt.title('Efficient Frontier with Alpha-Selected Portfolio')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

        return self.weights