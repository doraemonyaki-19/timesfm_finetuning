"""
Global 2-week live tracker — March 16–27, 2026.

Tracks actual daily close prices for the Option A portfolio:
  2382.HK, UNH, CVX, VWAGY, 0388.HK

Compares against TimesFM regional model forecasts.
Run each trading day after market close.

Usage:
  cd C:\\Users\\ylchen\\workspace\\timesfm
  source .venv/Scripts/activate
  python global_sim/track_2week.py
"""
import json
import datetime
import sys
from pathlib import Path

import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# ── Config ─────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
TICKERS_A   = ["2382.HK", "UNH", "CVX", "VWAGY", "0388.HK"]
CAPITAL     = 10_000.0
ALLOC       = CAPITAL / len(TICKERS_A)            # $2,000 each
START_DATE  = datetime.date(2026, 3, 16)
END_DATE    = datetime.date(2026, 3, 27)
FORECAST_FILE = ROOT / "forecast_mar14.json"
OUT_PNG     = ROOT / "track_2week.png"
OUT_JSON    = ROOT / "track_2week.json"

# HKD tickers — returns computed in local currency (% change) same as USD tickers
# HKD is USD-pegged (~7.78), so using local-currency % return is a valid approximation.
HKD_TICKERS = {"2382.HK", "0388.HK"}

COLORS = {
    "2382.HK": "#E53935",   # red
    "UNH":     "#1E88E5",   # blue
    "CVX":     "#43A047",   # green
    "VWAGY":   "#FB8C00",   # orange
    "0388.HK": "#8E24AA",   # purple
}

LABELS = {
    "2382.HK": "2382.HK (Sunny Opt.)",
    "UNH":     "UNH (UnitedHealth)",
    "CVX":     "CVX (Chevron)",
    "VWAGY":   "VWAGY (Volkswagen ADR)",
    "0388.HK": "0388.HK (HKEX)",
}

CURRENCY = {
    "2382.HK": "HKD",
    "UNH":     "USD",
    "CVX":     "USD",
    "VWAGY":   "USD",
    "0388.HK": "HKD",
}

STOP_LOSS_PCT = {       # ~4% below March 13 close (entry estimate)
    "2382.HK": 0.04,
    "UNH":     0.04,
    "CVX":     0.04,
    "VWAGY":   0.04,
    "0388.HK": 0.04,
}

# ── Load forecasts ──────────────────────────────────────────────────────────
with open(FORECAST_FILE) as f:
    fdata = json.load(f)

forecast_dates = [datetime.date.fromisoformat(d) for d in fdata["biz_days"]]
forecasts      = {t: fdata["forecasts"][t] for t in TICKERS_A}
prev_close     = {t: fdata["last_prices"][t] for t in TICKERS_A}   # March 13 close

stop_loss = {t: prev_close[t] * (1 - STOP_LOSS_PCT[t]) for t in TICKERS_A}

# ── Download actual prices ──────────────────────────────────────────────────
print("Downloading actual prices from yfinance...")
raw = yf.download(
    TICKERS_A,
    start=START_DATE.strftime("%Y-%m-%d"),
    end=(END_DATE + datetime.timedelta(days=4)).strftime("%Y-%m-%d"),
    interval="1d",
    auto_adjust=True,
    progress=False,
)

open_prices  = {}   # {ticker: {date: price}}
close_prices = {}   # {ticker: {date: price}}

for ticker in TICKERS_A:
    open_prices[ticker]  = {}
    close_prices[ticker] = {}
    try:
        if len(TICKERS_A) > 1:
            op = raw["Open"][ticker]
            cl = raw["Close"][ticker]
        else:
            op = raw["Open"]
            cl = raw["Close"]
    except KeyError:
        op = raw["Open"]
        cl = raw["Close"]

    for ts, price in op.items():
        d = ts.date() if hasattr(ts, "date") else datetime.date.fromisoformat(str(ts)[:10])
        if not (isinstance(price, float) and np.isnan(price)):
            open_prices[ticker][d] = float(price)
    for ts, price in cl.items():
        d = ts.date() if hasattr(ts, "date") else datetime.date.fromisoformat(str(ts)[:10])
        if not (isinstance(price, float) and np.isnan(price)):
            close_prices[ticker][d] = float(price)

# ── Entry prices = open on March 16 (or fallback to March 13 close) ─────────
entry = {}
for ticker in TICKERS_A:
    if START_DATE in open_prices[ticker]:
        entry[ticker] = open_prices[ticker][START_DATE]
    elif START_DATE in close_prices[ticker]:
        entry[ticker] = close_prices[ticker][START_DATE]
    else:
        entry[ticker] = prev_close[ticker]
        ccy = CURRENCY[ticker]
        print(f"  {ticker}: March 16 open not yet available, "
              f"using March 13 close {ccy} {prev_close[ticker]:.2f}")

# ── Shares / units — all expressed in local currency / allocation USD equivalent
# For HKD tickers: allocation is $2,000 USD equivalent; shares = (2000 * 7.78) / entry_HKD
# For USD tickers: shares = 2000 / entry_USD
# Since we track % returns and apply them to $2,000 alloc, we only need entry price
# for computing per-ticker return. No explicit share count needed.
print("\nEntry prices (March 16 open estimate):")
for t in TICKERS_A:
    ccy = CURRENCY[t]
    print(f"  {t:<10}  {ccy} {entry[t]:.2f}  (stop-loss: {ccy} {stop_loss[t]:.2f})")

# ── Build daily actual portfolio % return ──────────────────────────────────
# We work in % return space to avoid cross-currency issues.
# Daily portfolio return = equal-weighted avg of individual ticker returns from entry.

all_actual_dates = sorted(set(
    d for t in TICKERS_A for d in close_prices[t]
    if START_DATE <= d <= END_DATE
))

actual_portfolio = []   # [(date, portfolio_return_pct, {ticker: return_pct})]
for d in all_actual_dates:
    ticker_rets = {}
    for t in TICKERS_A:
        price = close_prices[t].get(d, None)
        if price is None:
            past = [v for dd, v in sorted(close_prices[t].items()) if dd <= d]
            price = past[-1] if past else entry[t]
        ticker_rets[t] = (price - entry[t]) / entry[t] * 100.0
    port_ret = sum(ticker_rets.values()) / len(TICKERS_A)
    actual_portfolio.append((d, port_ret, ticker_rets))

# ── Build forecast portfolio % return ─────────────────────────────────────
forecast_portfolio = []   # [(date, portfolio_return_pct, {ticker: return_pct})]
for i, d in enumerate(forecast_dates):
    ticker_rets = {}
    for t in TICKERS_A:
        fc_price = forecasts[t][i]
        ticker_rets[t] = (fc_price - prev_close[t]) / prev_close[t] * 100.0
    port_ret = sum(ticker_rets.values()) / len(TICKERS_A)
    forecast_portfolio.append((d, port_ret, ticker_rets))

# ── Print summary table ─────────────────────────────────────────────────────
fc_by_date = {d: (r, bk) for d, r, bk in forecast_portfolio}

print(f"\n{'='*72}")
print(f"{'Date':<13} {'Actual Ret':>10} {'Forecast':>10}  {'Delta':>8}")
print("-" * 72)

for d, actual_ret, _ in actual_portfolio:
    fc_r, _ = fc_by_date.get(d, (None, None))
    fc_str  = f"{fc_r:>+8.2f}%" if fc_r is not None else "     N/A"
    delta_s = f"{actual_ret - fc_r:>+7.2f}pp" if fc_r is not None else ""
    print(f"{d.strftime('%b %d %a'):<13} {actual_ret:>+9.2f}%  {fc_str}  {delta_s}")

# ── Per-ticker breakdown (latest date) ─────────────────────────────────────
if actual_portfolio:
    last_date, last_ret, last_bk = actual_portfolio[-1]
    print(f"\n{'='*72}")
    print(f"Latest: {last_date.strftime('%b %d, %Y')}  "
          f"Portfolio return: {last_ret:+.2f}%  "
          f"(${CAPITAL * (1 + last_ret/100):,.0f})")
    print(f"\n{'Ticker':<12} {'Entry':>10} {'Now':>10} {'Return':>8} "
          f"{'FC Return':>10} {'Stop-Loss':>10} {'Status':>10}")
    print("-" * 72)
    for t in TICKERS_A:
        ccy = CURRENCY[t]
        ret = last_bk[t]
        now_price = entry[t] * (1 + ret / 100)
        fc_i = next((i for i, d in enumerate(forecast_dates) if d == last_date), None)
        fc_ret = (forecasts[t][fc_i] - prev_close[t]) / prev_close[t] * 100 if fc_i is not None else None
        sl = stop_loss[t]
        below_stop = now_price < sl
        status = "STOP!!" if below_stop else "OK"
        fc_str = f"{fc_ret:>+9.2f}%" if fc_ret is not None else "       N/A"
        print(f"  {t:<10} {ccy}{entry[t]:>8.2f} {ccy}{now_price:>8.2f} "
              f"{ret:>+7.2f}%{fc_str}  {ccy}{sl:>8.2f}  {status}")

    # ── Final vs target comparison ──────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"10-Day Predicted vs Actual Returns  (as of {last_date.strftime('%b %d')})")
    print(f"{'Ticker':<12} {'Entry':>8} {'Pred 10d':>10} {'Actual':>8} "
          f"{'Pred Ret':>10} {'Act Ret':>10} {'Delta':>8}")
    for t in TICKERS_A:
        ccy    = CURRENCY[t]
        pred_f = forecasts[t][-1]
        act_r  = last_bk[t]
        pred_r = (pred_f - prev_close[t]) / prev_close[t] * 100
        delta  = act_r - pred_r
        act_p  = entry[t] * (1 + act_r / 100)
        print(f"  {t:<10} {ccy}{entry[t]:>6.2f} {ccy}{pred_f:>8.2f}  {ccy}{act_p:>6.2f} "
              f"{pred_r:>+9.2f}%{act_r:>+10.2f}%  {delta:>+7.2f}pp")

# ── Save results JSON ───────────────────────────────────────────────────────
results = {
    "generated": datetime.date.today().isoformat(),
    "entry_prices": entry,
    "stop_losses": stop_loss,
    "capital": CAPITAL,
    "actual_portfolio": [
        {"date": d.isoformat(), "return_pct": r,
         "per_ticker": {t: bk[t] for t in TICKERS_A}}
        for d, r, bk in actual_portfolio
    ],
    "forecast_portfolio": [
        {"date": d.isoformat(), "return_pct": r,
         "per_ticker": {t: bk[t] for t in TICKERS_A}}
        for d, r, bk in forecast_portfolio
    ],
}
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved: {OUT_JSON}")

# ── Plot ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16))
gs_layout = GridSpec(4, 5, figure=fig, hspace=0.50, wspace=0.35,
                     height_ratios=[2.5, 1.5, 1.5, 1.5])

# Pre-compute shared variables
act_dates    = [d for d, _, _ in actual_portfolio]
act_rets     = [r for _, r, _ in actual_portfolio]
fc_dates     = [d for d, _, _ in forecast_portfolio]
fc_rets      = [r for _, r, _ in forecast_portfolio]
anchor_date  = datetime.date(2026, 3, 13)
anchor_ret   = 0.0  # returns anchored at 0% on March 13

# ── Row 0: Portfolio return (actual vs forecast) ───────────────────────────
ax1 = fig.add_subplot(gs_layout[0, :])

if actual_portfolio:
    ax1.plot(act_dates, act_rets, "o-", color="black", linewidth=2.2,
             markersize=6, label="Actual portfolio return", zorder=5)

ax1.plot([anchor_date] + fc_dates,
         [anchor_ret] + fc_rets,
         "--", color="steelblue", linewidth=1.8, alpha=0.75,
         label="Model forecast return")

ax1.axhline(0, color="gray", linestyle=":", alpha=0.4, label="Entry (breakeven)")

# Shade actual vs forecast gap
if actual_portfolio:
    shared = [(d, ar, fc_by_date[d][0])
              for d, ar, _ in actual_portfolio if d in fc_by_date]
    if shared:
        for d_s, a_s, f_s in shared:
            color = "green" if a_s >= f_s else "red"
            ax1.fill_between([d_s, d_s], [f_s], [a_s], alpha=0.18, color=color)

# Mid-plan checkpoint vertical line
mid_date = datetime.date(2026, 3, 20)
ax1.axvline(mid_date, color="orange", linestyle="--", alpha=0.5, linewidth=1)
ax1.text(mid_date, ax1.get_ylim()[0] if actual_portfolio else -0.5,
         "  Mar 20\n  checkpoint", fontsize=7, color="orange", va="bottom")

ax1.set_title(
    "Portfolio Return: Actual vs Model Forecast  (Mar 16–27, 2026)\n"
    "Option A: 2382.HK / UNH / CVX / VWAGY / 0388.HK  ($2,000 each, equal-weight)",
    fontsize=11, fontweight="bold"
)
ax1.set_ylabel("Portfolio Return (%)", fontsize=10)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.2f}%"))
ax1.legend(fontsize=9)
ax1.tick_params(axis="x", rotation=20, labelsize=8)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.grid(axis="y", alpha=0.2)

# ── Row 1, cols 0–2: Per-ticker return bar chart ──────────────────────────
ax_bars = fig.add_subplot(gs_layout[1, :3])
if actual_portfolio:
    last_date, last_ret, last_bk = actual_portfolio[-1]
    fc_i_last = next((i for i, d in enumerate(forecast_dates) if d == last_date),
                     len(forecast_dates) - 1)
    pred_rets_bar = [(forecasts[t][fc_i_last] - prev_close[t]) / prev_close[t] * 100
                     for t in TICKERS_A]
    act_rets_bar  = [last_bk[t] for t in TICKERS_A]

    x = np.arange(len(TICKERS_A))
    w = 0.35
    bars_p = ax_bars.bar(x - w/2, pred_rets_bar, w, label="Forecast",
                         color="steelblue", alpha=0.8)
    bars_a = ax_bars.bar(x + w/2, act_rets_bar, w, label="Actual",
                         color="orange", alpha=0.85)

    for bar, r in zip(bars_p, pred_rets_bar):
        ypos = bar.get_height() + 0.04 if r >= 0 else bar.get_height() - 0.35
        ax_bars.text(bar.get_x() + bar.get_width() / 2, ypos,
                     f"{r:+.2f}%", ha="center", va="bottom",
                     fontsize=6.5, color="steelblue")
    for bar, r in zip(bars_a, act_rets_bar):
        ypos = bar.get_height() + 0.04 if r >= 0 else bar.get_height() - 0.35
        ax_bars.text(bar.get_x() + bar.get_width() / 2, ypos,
                     f"{r:+.2f}%", ha="center", va="bottom",
                     fontsize=7, fontweight="bold", color="darkorange")

    ax_bars.axhline(0, color="black", linewidth=0.8)
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(TICKERS_A, fontsize=9)
    ax_bars.set_title(f"Per-Ticker Return  (as of {last_date.strftime('%b %d')})",
                      fontsize=9, fontweight="bold")
    ax_bars.set_ylabel("Return (%)", fontsize=9)
    ax_bars.legend(fontsize=8)
    ax_bars.grid(axis="y", alpha=0.2)

# ── Row 1, cols 3–4: Stop-loss status ────────────────────────────────────
ax_sl = fig.add_subplot(gs_layout[1, 3:])
if actual_portfolio:
    last_date, last_ret, last_bk = actual_portfolio[-1]
    now_prices  = {t: entry[t] * (1 + last_bk[t] / 100) for t in TICKERS_A}
    # Normalize: show current price as % of stop-loss (1.0 = at stop, >1 = safe)
    sl_ratios   = [now_prices[t] / stop_loss[t] for t in TICKERS_A]
    sl_colors   = ["red" if r < 1.0 else "steelblue" for r in sl_ratios]
    bars_sl = ax_sl.barh(TICKERS_A, sl_ratios, color=sl_colors, alpha=0.8)
    ax_sl.axvline(1.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    for bar, r in zip(bars_sl, sl_ratios):
        safe_pct = (r - 1.0) * 100
        ax_sl.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                   f"{safe_pct:+.1f}% cushion", va="center", fontsize=7)
    ax_sl.set_title(f"Stop-Loss Cushion\n(as of {last_date.strftime('%b %d')})",
                    fontsize=9, fontweight="bold")
    ax_sl.set_xlabel("Price / Stop-Loss Level", fontsize=8)
    ax_sl.set_xlim(0.9, max(sl_ratios) + 0.08)
    ax_sl.grid(axis="x", alpha=0.2)

# ── Rows 2–3: Per-ticker price tracks ────────────────────────────────────
for idx, ticker in enumerate(TICKERS_A):
    row = 2 + idx // 3
    col = idx % 3 if idx < 3 else (idx - 3) * 2      # 0,1,2 then 0,2
    if idx == 3:
        col_span = slice(0, 2)
        ax = fig.add_subplot(gs_layout[row, col_span])
    elif idx == 4:
        col_span = slice(2, 5)
        ax = fig.add_subplot(gs_layout[row, col_span])
    else:
        ax = fig.add_subplot(gs_layout[row, col])

    color = COLORS[ticker]
    ccy   = CURRENCY[ticker]

    # Forecast line (anchored at March 13 close)
    fc_prices = [prev_close[ticker]] + forecasts[ticker]
    fc_x      = [anchor_date] + forecast_dates
    ax.plot(fc_x, fc_prices, "--", color=color, alpha=0.55,
            linewidth=1.5, label="Forecast")

    # Stop-loss line
    ax.axhline(stop_loss[ticker], color="red", linestyle=":", alpha=0.5,
               linewidth=1, label=f"Stop {ccy}{stop_loss[ticker]:.1f}")

    # Entry price line
    ax.axhline(entry[ticker], color=color, linestyle=":", alpha=0.35,
               linewidth=1, label=f"Entry {ccy}{entry[ticker]:.2f}")

    # Actual prices
    act_px = [(d, close_prices[ticker][d])
              for d in all_actual_dates if d in close_prices[ticker]]
    if act_px:
        ax_d = [x[0] for x in act_px]
        ax_p = [x[1] for x in act_px]
        ax.plot(ax_d, ax_p, "o-", color=color, linewidth=2,
                markersize=4, label="Actual")
        latest_price = ax_p[-1]
        latest_ret   = (latest_price - entry[ticker]) / entry[ticker] * 100
        ax.set_title(f"{ticker}  {latest_ret:+.2f}%  ({ccy}{latest_price:.2f})",
                     fontsize=9, fontweight="bold", color=color)
    else:
        ax.set_title(f"{ticker}  (awaiting data)",
                     fontsize=9, fontweight="bold", color=color)

    ax.set_ylabel(f"Price ({ccy})", fontsize=7.5)
    ax.tick_params(axis="x", rotation=30, labelsize=6.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.legend(fontsize=6.0, loc="best")
    ax.grid(alpha=0.15)

# ── Figure title ──────────────────────────────────────────────────────────
today_str = datetime.date.today().strftime("%B %d, %Y")
if actual_portfolio:
    last_date, last_ret, _ = actual_portfolio[-1]
    status = (f"Portfolio: {last_ret:+.2f}%  (${CAPITAL * (1 + last_ret/100):,.0f})  "
              f"as of {last_date.strftime('%b %d')}")
else:
    status = "Awaiting first trading day data (entry: March 16)"

fig.suptitle(
    f"Global 2-Week Live Tracker — Option A  (Updated {today_str})\n"
    f"2382.HK / UNH / CVX / VWAGY / 0388.HK  |  $10,000 ($2,000 each)  |  {status}",
    fontsize=12, fontweight="bold"
)

plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved: {OUT_PNG}")
