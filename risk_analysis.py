from utils import *

class RiskAnalysis ():

    '''
    Compute VaR and CVaR and display a visualization
    Convention: losses are shown as negative values on plots (so left side = worse losses)
    Internally, VaR/CVaR are computed as loss magnitudes (positive), but we plot their negatives
    '''

    def __init__(self, df: pd.DataFrame):
        if 'Total_Portfolio_Value' not in df.columns:
            raise ValueError("DataFrame must contain 'Total_Portfolio_Value'")
        self.df = df.copy().sort_index()

    def _simple_returns (self) -> pd.Series :
        ret = self.df['Total_Portfolio_Value'].pct_change().dropna()
        return ret

    def _log_returns (self) -> pd.Series : 
        ret = np.log(self.df['Total_Portfolio_Value'] / self.df['Total_Portfolio_Value'].shift(1)).dropna()
        return ret

    def var_hist (self, confidence_level) :
        alpha = _ensure_alpha(confidence_level)
        ret = self._simple_returns()
        var_val = float(np.quantile(ret, 1 - alpha))
        cvar_val = float(ret[ret <= var_val].mean())  
        returns = np.sort(ret.to_numpy())
        n = returns.size
        cdf = (np.arange(1, n+1)/n)

        plt.figure(figsize=(9,4))
        plt.plot(returns, cdf, marker='.', linestyle='-', color='green', linewidth=2, label ='CDF of returns')
        plt.axvline(x=var_val, color='red', linestyle=':', linewidth=1, label=f'VaR {int(alpha*100)}% = {var_val:.4f}')
        plt.axvline(x=cvar_val, color='orange', linestyle=':', linewidth=1, label=f'CVaR {int(alpha*100)}% = {cvar_val:.4f}')
        plt.xlabel('Simple Returns (losses on the left)')
        plt.ylabel('CDF')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title('Historical VaR/CVaR (returns view)')
        plt.show()

        var_mag = abs(var_val)
        cvar_mag = abs(cvar_val)
        return var_mag, cvar_mag, var_val, cvar_val

    
    def var_param(self, confidence_level, days: int):
        alpha = _ensure_alpha(confidence_level)
        ret = self._log_returns()
    
        stat, p_value = shapiro(ret)
        if p_value > 0.05:
            print(f"[Shapiro-Wilk] p-value = {p_value:.4f} → Normality check of returns: positive")
        else:
            print(f"[Shapiro-Wilk] p-value = {p_value:.4f} → Normality check of returns: negative")
        
        mu = ret.mean()
        sigma = ret.std(ddof=1)
        z = norm.ppf(alpha)
        hs = np.arange(1, days+1)
        var_path = (-mu*hs + sigma*np.sqrt(hs)*z)
        cvar_path = (-mu*hs + sigma*np.sqrt(hs)*(norm.pdf(z)/(1-alpha)))
        var_signed = -var_path
        cvar_signed = -cvar_path

        plt.figure(figsize=(9,4))
        plt.plot(hs, -var_path, color='purple', lw=2, label=f'VaR {int(alpha*100)}%')
        plt.plot(hs, -cvar_path, color='orange', lw=2, ls=':', label=f'CVaR {int(alpha*100)}%')
        plt.xlabel('Horizon (periods)')
        plt.ylabel('Loss (display in negative form)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title('Parametric VaR/CVaR (Normal) — time scaling')
        plt.show()

        return float(var_path[-1]), float(cvar_path[-1]), float(var_signed[-1]), float(cvar_signed[-1])


    def var_montecarlo(self, confidence_level, n_sims: int, horizon: int, seed: int = 22):
        confidence_level = _ensure_alpha(confidence_level)
        tail_prob = 1 - confidence_level
        ret = self._log_returns()
        mu = ret.mean()
        sigma = ret.std(ddof=1)
        rng = np.random.default_rng(seed)
        sims_log = rng.normal(mu, sigma, size=(horizon, n_sims)).sum(axis=0)
        sims_simple = np.expm1(sims_log)
        var_simple = float(np.quantile(sims_simple, tail_prob))
        cvar_simple = float(sims_simple[sims_simple <= var_simple].mean())

        plt.figure(figsize=(8, 4))
        plt.hist(sims_simple, bins=60, color='green', edgecolor='black', alpha=0.85)
        plt.axvline(var_simple, color='red', linestyle='--',
                    label=f'VaR {int(confidence_level*100)}% = {abs(var_simple):.2%}')
        plt.axvline(cvar_simple, color='orange', linestyle='--',
                    label=f'CVaR {int(confidence_level*100)}% = {abs(cvar_simple):.2%}')
        plt.xlabel('Aggregated simple returns over horizon (losses on the left)')
        plt.title(f'Monte Carlo — returns distribution (H={horizon}, n={n_sims})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        return abs(var_simple), abs(cvar_simple), var_simple, cvar_simple