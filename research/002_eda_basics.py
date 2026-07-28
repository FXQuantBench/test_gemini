import pandas as pd
import numpy as np

def main(conn):
    # Tick stats
    stats = conn.execute("""
        SELECT 
            COUNT(*) as total_ticks,
            MIN(timestamp_utc) as min_ts,
            MAX(timestamp_utc) as max_ts,
            AVG(ask - bid) as avg_spread,
            MEDIAN(ask - bid) as median_spread,
            QUANTILE_CONT(ask - bid, 0.95) as p95_spread
        FROM GBPUSD
    """).df()

    avg_sp = stats['avg_spread'].iloc[0]
    med_sp = stats['median_spread'].iloc[0]
    p95_sp = stats['p95_spread'].iloc[0]
    total = stats['total_ticks'].iloc[0]

    # 5-minute bar resampling
    bars_df = conn.execute("""
        SELECT 
            timestamp_utc - (timestamp_utc % 300000) as bar_ms,
            ARG_MIN((bid + ask) / 2.0, timestamp_utc) as open,
            MAX((bid + ask) / 2.0) as high,
            MIN((bid + ask) / 2.0) as low,
            ARG_MAX((bid + ask) / 2.0, timestamp_utc) as close,
            AVG(ask - bid) as avg_spread,
            COUNT(*) as tick_count
        FROM GBPUSD
        GROUP BY 1
        ORDER BY 1
    """).df()

    bars_df['ret'] = bars_df['close'].pct_change()
    bars_df['ret_lag1'] = bars_df['ret'].shift(1)
    autocorr_5m = bars_df['ret'].corr(bars_df['ret_lag1'])

    # Hourly volatility pattern
    bars_df['dt'] = pd.to_datetime(bars_df['bar_ms'], unit='ms', utc=True)
    bars_df['hour'] = bars_df['dt'].dt.hour
    hourly_vol = bars_df.groupby('hour')['ret'].std()

    print(f"EDA Result: Total ticks={total}, Avg Spread={avg_sp:.6f}, Med Spread={med_sp:.6f}, 5m Ret Autocorr={autocorr_5m:.4f}")
    print("\nHourly Volatility (5m return std dev):")
    print(hourly_vol.to_string())

if __name__ == '__main__':
    main(conn)
