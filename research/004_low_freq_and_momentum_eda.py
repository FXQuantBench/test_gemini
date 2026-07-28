import pandas as pd
import numpy as np

def main(conn):
    # Aggregate ticks into 1-hour bars using DuckDB
    df_1h = conn.execute("""
        SELECT
            timestamp_utc - (timestamp_utc % 3600000) AS timestamp_utc,
            ARG_MIN((bid + ask) / 2.0, timestamp_utc) AS open_mid,
            MAX((bid + ask) / 2.0) AS high_mid,
            MIN((bid + ask) / 2.0) AS low_mid,
            ARG_MAX((bid + ask) / 2.0, timestamp_utc) AS close_mid,
            AVG(ask - bid) AS avg_spread
        FROM GBPUSD
        GROUP BY 1
        ORDER BY 1
    """).df()

    df_1h['dt'] = pd.to_datetime(df_1h['timestamp_utc'], unit='ms', utc=True)
    df_1h['hour'] = df_1h['dt'].dt.hour
    df_1h['date'] = df_1h['dt'].dt.date
    df_1h['ret'] = df_1h['close_mid'].pct_change()

    autocorr_1h = df_1h['ret'].autocorr(1)

    # Strategy 1: Donchian 24h Breakout with SMA200 trend filter on 1h bars
    df_1h['upper'] = df_1h['high_mid'].shift(1).rolling(24).max()
    df_1h['lower'] = df_1h['low_mid'].shift(1).rolling(24).min()
    df_1h['sma200'] = df_1h['close_mid'].shift(1).rolling(200).mean()

    sig1 = np.zeros(len(df_1h))
    pos = 0.0
    for i in range(200, len(df_1h)):
        c = df_1h['close_mid'].iloc[i]
        u = df_1h['upper'].iloc[i]
        l = df_1h['lower'].iloc[i]
        sma = df_1h['sma200'].iloc[i]
        
        if pd.isna(u) or pd.isna(l) or pd.isna(sma):
            continue
            
        if c > u and c > sma:
            pos = 1.0
        elif c < l and c < sma:
            pos = -1.0
        elif (pos == 1.0 and c < sma) or (pos == -1.0 and c > sma):
            pos = 0.0
        sig1[i] = pos

    df_1h['sig1'] = sig1
    df_1h['sig1_shift'] = df_1h['sig1'].shift(1).fillna(0)
    sig_diff1 = df_1h['sig1_shift'].diff().abs().fillna(0)
    cost1 = sig_diff1 * (0.000088 / df_1h['close_mid'])
    strat_ret1 = df_1h['sig1_shift'] * df_1h['ret'] - cost1
    sharpe1 = strat_ret1.mean() / (strat_ret1.std() + 1e-8) * np.sqrt(24 * 252)
    trades1 = int((sig_diff1 > 0).sum())
    tot_ret1 = float(strat_ret1.sum())

    # Strategy 2: London Breakout Strategy (Asian range breakout)
    asian_high = df_1h[df_1h['hour'] < 7].groupby('date')['high_mid'].max()
    asian_low = df_1h[df_1h['hour'] < 7].groupby('date')['low_mid'].min()

    df_1h['asian_high'] = df_1h['date'].map(asian_high)
    df_1h['asian_low'] = df_1h['date'].map(asian_low)

    sig2 = np.zeros(len(df_1h))
    for i in range(len(df_1h)):
        h = df_1h['hour'].iloc[i]
        c = df_1h['close_mid'].iloc[i]
        ah = df_1h['asian_high'].iloc[i]
        al = df_1h['asian_low'].iloc[i]
        
        if pd.isna(ah) or pd.isna(al):
            continue
            
        if 7 <= h <= 16:
            if c > ah:
                sig2[i] = 1.0
            elif c < al:
                sig2[i] = -1.0
            else:
                sig2[i] = sig2[i-1] if i > 0 and 7 <= df_1h['hour'].iloc[i-1] <= 16 else 0.0
        else:
            sig2[i] = 0.0

    df_1h['sig2'] = sig2
    df_1h['sig2_shift'] = df_1h['sig2'].shift(1).fillna(0)
    sig_diff2 = df_1h['sig2_shift'].diff().abs().fillna(0)
    cost2 = sig_diff2 * (0.000088 / df_1h['close_mid'])
    strat_ret2 = df_1h['sig2_shift'] * df_1h['ret'] - cost2
    sharpe2 = strat_ret2.mean() / (strat_ret2.std() + 1e-8) * np.sqrt(24 * 252)
    trades2 = int((sig_diff2 > 0).sum())
    tot_ret2 = float(strat_ret2.sum())

    print(f"EDA Result: Donchian 1h Sharpe={sharpe1:.2f} (Trades={trades1}, Ret={tot_ret1:.4f}) | London Breakout Sharpe={sharpe2:.2f} (Trades={trades2}, Ret={tot_ret2:.4f}) | 1h Autocorr={autocorr_1h:.4f}")

if __name__ == "__main__":
    main(conn)
