import duckdb
import pandas as pd
import numpy as np

def main(conn):
    # Query 5-minute candle aggregates directly from GBPUSD view
    df_5m = conn.execute("""
        SELECT
            timestamp_utc - (timestamp_utc % 300000) AS bar_ms,
            ARG_MIN((bid + ask) / 2.0, timestamp_utc) AS open_mid,
            MAX((bid + ask) / 2.0) AS high_mid,
            MIN((bid + ask) / 2.0) AS low_mid,
            ARG_MAX((bid + ask) / 2.0, timestamp_utc) AS close_mid,
            AVG(ask - bid) AS avg_spread,
            COUNT(*) AS tick_count
        FROM GBPUSD
        GROUP BY 1
        ORDER BY 1
    """).df()

    df_5m['datetime'] = pd.to_datetime(df_5m['bar_ms'], unit='ms', utc=True)
    df_5m['hour'] = df_5m['datetime'].dt.hour
    df_5m['ret'] = df_5m['close_mid'].pct_change()

    # Strategy 1: 5-minute Mean Reversion (Z-Score on close_mid)
    window = 36  # 3 hours
    roll_mean = df_5m['close_mid'].rolling(window).mean()
    roll_std = df_5m['close_mid'].rolling(window).std().clip(lower=1e-8)
    df_5m['z_score'] = (df_5m['close_mid'] - roll_mean) / roll_std

    # Signal logic: buy low z-score, sell high z-score
    # Filter: London & NY hours (07:00 to 19:00 UTC) for better liquidity & tighter spreads
    session_filter = (df_5m['hour'] >= 7) & (df_5m['hour'] <= 19)
    
    raw_sig = np.zeros(len(df_5m))
    raw_sig[df_5m['z_score'] < -1.5] = 1.0
    raw_sig[df_5m['z_score'] > 1.5] = -1.0
    # Apply session filter
    raw_sig[~session_filter] = 0.0

    df_5m['signal'] = raw_sig
    # Compute position changes to estimate spread cost
    df_5m['pos_change'] = df_5m['signal'].diff().abs().fillna(0.0)
    
    # Simple backtest approximation
    df_5m['gross_pnl'] = df_5m['signal'].shift(1) * df_5m['ret']
    # Spread cost as fraction of price
    df_5m['spread_cost'] = df_5m['pos_change'] * (df_5m['avg_spread'] / df_5m['close_mid'])
    df_5m['net_pnl'] = df_5m['gross_pnl'] - df_5m['spread_cost']

    # Annualized metric calculations (assuming 5-min bars, 105120 bars/year)
    bars_per_yr = 105120
    net_mean = df_5m['net_pnl'].mean()
    net_std = df_5m['net_pnl'].std()
    net_sharpe = (net_mean / net_std * np.sqrt(bars_per_yr)) if net_std > 0 else 0.0
    total_trades = int(df_5m['pos_change'].sum() / 2)
    total_net_ret = df_5m['net_pnl'].sum()

    print(f"EDA Result: 5m Z-Score Mean Reversion (win={window}) Net Sharpe={net_sharpe:.2f}, Total Trades={total_trades}, Net Return={total_net_ret:.4f}")

if __name__ == "__main__": routine = main(conn) if 'conn' in globals() else None
