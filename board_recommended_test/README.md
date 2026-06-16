# Board-Recommended Paper Trade Test

**Initiated:** 2026-04-11
**Notional:** $1,000,000
**Status:** Active (dry-run mode)

## Purpose

This directory houses a lifecycle paper-trade simulation that tracks the
board-approved allocation across TimesFM regional finetuned models. The test
validates whether the OOS statistical edge observed in bootstrap evaluations
translates to positive P&L in a realistic, rule-governed portfolio before any
live deployment.  Board recommendation at C:\Users\ylchen\workspace\personas\board_meetings\timesfm_lsf_strategy_apr2026.md

## Board-Approved Allocation

| Cell | Region | Horizon | Weight | Strategy | Model |
|------|--------|---------|--------|----------|-------|
| EUROPE 120d | Europe | 120 days | 25% | LSF (Long/Short/Flat) | EUv2 |
| JAPAN 120d | Japan | 120 days | 20% | LSF (Long/Short/Flat) | hdecay1.0 |
| EUROPE 60d | Europe | 60 days | 15% | LSF (Long/Short/Flat) | EUv2 |
| CHINA 120d | China | 120 days | 10% | LF (Long/Flat only) | Glia2a |
| JAPAN 60d | Japan | 60 days | 10% | LSF (Long/Short/Flat) | hdecay1.0 |
| US 60d | US | 60 days | 10% | LF (Long/Flat only) | Run4 |
| Cash | -- | -- | 10% | -- | -- |

**Blocked cells:** US 120d (28.9% directional accuracy -- below threshold).

## Tickers Traded

### Europe (9 tickers)

Eval (held-out): UL, EADSF, VWAGY
Train (in-sample): ASML, SAP, NVO, AZN, SHEL, DEO

Both eval and train tickers are traded. The model was finetuned on train
tickers only; eval tickers test generalization.

### Japan (9 tickers)

Eval: 8035.T, 6902.T, 4661.T
Train: 7203.T, 6758.T, 9984.T, 8306.T, 4502.T, 7974.T

### China (9 tickers, LF strategy -- no shorts)

Eval: 0388.HK, 2382.HK, 1177.HK
Train: 0700.HK, 0941.HK, 2318.HK, 1299.HK, 0005.HK, 0001.HK

No-short blocklist applies to sovereign-adjacent names: 0388.HK (HKEX),
0005.HK (HSBC HK), 0941.HK (China Mobile). Under LF strategy these would
already be flat on SHORT signals; the blocklist is a hard override.

### US (12 tickers, LF strategy -- no shorts)

Eval: NVDA, UNH, GS, CVX, KO, BA
Train: AAPL, MSFT, AMZN, GOOGL, META, JPM

US 120d is blocked. Only the 60d cell trades.

## Governance Rules

| Rule | Value |
|------|-------|
| Single-name short cap | 3% of notional |
| Stop-loss (per position) | -10% |
| Max aggregate net-short exposure | 40% of notional |
| Borrow cost (annualized) | 1.5% |
| Dead zone enforcement | Per-cell optimal DZ from routing_table.json |
| No-short blocklist | 0388.HK, 0005.HK, 0941.HK |

## Current Positions (2026-04-11, dry-run)

38 open positions across 6 cells after initial rebalance:

| Cell | Positions | Long | Short | Allocation ($) |
|------|-----------|------|-------|----------------|
| EUROPE 120d | 8 | 4 (UL, ASML, SAP, NVO) | 3 (EADSF, SHEL, DEO) | $250,000 |
| JAPAN 120d | 9 | 4 (8035.T, 7203.T, 6758.T, 9984.T) | 4 (6902.T, 8306.T, 4502.T, 7974.T) | $200,000 |
| EUROPE 60d | 8 | 4 (UL, ASML, SAP, NVO) | 3 (EADSF, SHEL, DEO) | $150,000 |
| CHINA 120d | 4 | 4 (0388.HK, 0700.HK, 0941.HK, 2318.HK) | 0 | $100,000 |
| JAPAN 60d | 8 | 4 (8035.T, 7203.T, 6758.T, 9984.T) | 3 (6902.T, 4502.T, 7974.T) | $100,000 |
| US 60d | 5 | 5 (NVDA, CVX, KO, BA, JPM) | 0 | $100,000 |
| **Total** | **38** | **25** | **13** | **$900,000** |

Cash reserve: $100,000 (10%).

Note: VWAGY and 4661.T were opened but hit -10% stop-loss on the same
day and were closed (see journal.csv for details).

## Closed Trades (journal.csv)

4 trades closed on 2026-04-11, all via STOP-LOSS:

| Ticker | Cell | Signal | Entry | Exit | P&L |
|--------|------|--------|-------|------|-----|
| VWAGY | EUROPE 120d | SHORT | 87.34 | 110.11 | -$7,243 |
| 4661.T | JAPAN 120d | SHORT | 87.34 | 98.63 | -$2,874 |
| VWAGY | EUROPE 60d | SHORT | 87.34 | 110.11 | -$4,346 |
| 4661.T | JAPAN 60d | SHORT | 87.34 | 98.63 | -$1,437 |

Total realized loss: -$15,901

## Rebalance Schedule

Positions are horizon-aligned. Each cell only opens new positions when
the prior batch expires.

| Cell | Next Rebalance |
|------|----------------|
| EUROPE 120d | 2026-10-08 |
| JAPAN 120d | 2026-10-08 |
| EUROPE 60d | 2026-07-10 |
| CHINA 120d | 2026-10-08 |
| JAPAN 60d | 2026-07-10 |
| US 60d | 2026-07-10 |

## How to Run

All commands use the venv Python from the project root:

```bash
cd timesfm

# Initialize with $1M notional
.venv/Scripts/python.exe board_recommended_test/paper_trade.py init --notional 1000000

# Daily run (MTM -> stop-losses -> expiry -> rebalance)
.venv/Scripts/python.exe board_recommended_test/paper_trade.py run

# Dry-run mode (random prices, no model inference)
.venv/Scripts/python.exe board_recommended_test/paper_trade.py run --dry-run

# Show current portfolio status
.venv/Scripts/python.exe board_recommended_test/paper_trade.py status

# Show closed-trade journal
.venv/Scripts/python.exe board_recommended_test/paper_trade.py history
```

## Files

| File | Description |
|------|-------------|
| `paper_trade.py` | Lifecycle simulator script |
| `state.json` | Current portfolio state (positions, cells, cash) |
| `journal.csv` | Append-only log of all closed trades |
| `README.md` | This file |

## Configuration

All config is externalized to `sector_routing_config.yaml` in the project
root. This includes region definitions, ticker lists, blocked cells,
no-short blocklist, dead zone thresholds, and model checkpoints.
