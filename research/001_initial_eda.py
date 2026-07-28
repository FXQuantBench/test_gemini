import pandas as pd
import numpy as np

def main(conn):
    # 1. Overall summary & spread statistics
    spread_stats = conn.execute("""
        SELECT
            COUNT(*) as total_ticks,
            AVG(ask - bid) as avg_spread,
            MEDIAN(ask - bid) as median_spread,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ask - bid) as p95_spread,
            AVG((bid + ask) / 2.0) as avg_mid
        FROM GBPUSD
    """).df()
    
    avg_spread = spread_stats['avg_spread'].iloc[0]
    avg_mid = spread_stats['avg_mid'].iloc[0]
    spread_pips = avg_spread * 10000
    total_ticks = spread_stats['total_ticks'].iloc[0]
    
    print(f"KEY EDA RESULT: Total ticks={total_ticks:,}, Avg Spread={spread_pips:.2f} pips ({avg_spread:.6f}), Avg Mid={avg_mid:.5f}")
    print("\n--- Overall Spread Statistics ---")
    print(spread_stats.to_string(index=False))

    # 2. Hourly seasonality: Spread & Volatility by UTC Hour
    hourly_stats = conn.execute("""
        WITH minute_bars AS (
            SELECT
                timestamp_utc - (timestamp_utc % 60000) AS minute_ms,
                epoch_ms(timestamp_utc - (timestamp_utc % 60000)) AS minute_dt,
                AVG((bid + ask) / 2.0) AS mid,
                AVG(ask - bid) AS avg_spread,
                COUNT(*) AS ticks_per_min
            FROM GBPUSD
            GROUP BY 1, 2
        ),
        minute_returns AS (
            SELECT
                minute_ms,
                EXTRACT(HOUR FROM minute_dt) AS hour_utc,
                mid,
                (mid - LAG(mid) OVER (ORDER BY minute_ms)) / LAG(mid) OVER (ORDER BY minute_ms) AS ret,
                avg_spread,
                ticks_per_min
            FROM minute_bars
        )
        SELECT
            hour_utc,
            COUNT(*) as minute_count,
            AVG(ticks_per_min) as avg_ticks_per_min,
            AVG(avg_spread) * 10000 as avg_spread_pips,
            STDDEV(ret) * 10000 as min_volatility_pips
        FROM minute_returns
        WHERE ret IS NOT NULL
        GROUP BY hour_utc
        ORDER BY hour_utc
    """).df()

    print("\n--- Hourly Seasonality (UTC) ---")
    print(hourly_stats.to_string(index=False))

    # 3. Autocorrelation of returns at different time aggregations (sampled into memory)
    sampled_minutes = conn.execute("""
        SELECT
            timestamp_utc - (timestamp_utc % 60000) AS minute_ms,
            AVG((bid + ask) / 2.0) AS mid
        FROM GBPUSD
        GROUP BY 1
        ORDER BY 1
    """).df()

    sampled_minutes['dt'] = pd.to_datetime(sampled_minutes['minute_ms'], unit='ms', utc=True)
    sampled_minutes = sampled_minutes.set_index('dt').sort_index()
    
    for freq in ['1min', '5min', '15min', '1h']:
        resampled = sampled_minutes['mid'].resample(freq).last().dropna()
        rets = resampled.pct_change().dropna()
        autocorr_lag1 = rets.autocorr(lag=1)
        autocorr_lag2 = rets.autocorr(lag=2)
        autocorr_lag3 = rets.autocorr(lag=3)
        print(f"\nAutocorrelation for {freq} bars (N={len(rets):,}): Lag1={autocorr_lag1:.4f}, Lag2={autocorr_lag2:.4f}, Lag3={autocorr_lag3:.4f}")

if __name__ == '__main__': 