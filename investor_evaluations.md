# Investor Evaluations — Sector Routing Results

Four legendary investors evaluate the TimesFM sector routing system.
Generated: 2026-04-01 (updated with routing results)

---

## Warren Buffett — The Oracle of Omaha

Well, friends, let me tell you what I see when I look at these numbers. And I want to be honest with you, because in our business, the worst thing you can do is fool yourself — and you're the easiest person to fool.

### The Business Proposition

What we have here is a forecasting system that takes a pretrained foundation model — think of it as a general-purpose engine — and fine-tunes it on regional stock data. Then a "sector router" decides, ticker by ticker, whether the fine-tuned model actually helps or whether you're better off with the original.

The pitch is: "We can forecast stock prices 10-20% more accurately than the base model at longer horizons."

### What the Numbers Tell Me

**The good news is real, but modest.** At 120 days, the routing system shaves roughly 1-3 percentage points off MAPE across all four regions. China goes from 12.16% to 9.47% — that's a 22% relative improvement. Europe from 13.46% to 11.49%. These are not nothing.

**But here's what keeps me up at night.** At short horizons — 5 and 14 days, where most trading actually happens — the improvements are tiny. US at 5 days: zero. Japan at 5 days: zero. You're telling me you built this whole apparatus and at the horizon where capital turns over fastest, you can't beat the free model?

That reminds me of something Charlie says: "Show me the incentive and I'll show you the outcome." The system is optimized for long-horizon MAPE, so naturally it does best there. But is that where the economic value lies? A 120-day forecast with 9-13% error is like telling me the temperature next July will be somewhere between 72 and 98 degrees. Technically true, practically useless for deciding whether to bring a jacket tomorrow.

### The Moat Assessment

**Pricing Power:** None. The underlying model is open-source from Google. Anyone can replicate this. You're selling a recipe that uses someone else's kitchen.

**Switching Costs:** Near zero. If a better foundation model comes out next quarter — and in AI, they come out like rabbits — your fine-tuned checkpoints become scrap metal.

**Network Effects:** None. More users don't make the forecasts better.

**Cost Advantage:** Modest. Fine-tuning costs roughly 2-8 hours of compute per region. A graduate student with a cloud GPU can replicate this over a weekend.

**Moat verdict: There isn't one.** The "sector routing" insight is genuinely interesting. But it's an observation, not a moat. Once you publish it, everyone knows it.

### The Banking Problem Is the Most Interesting Finding

The single most valuable piece of intelligence in this entire exercise isn't the routing system — it's the *pattern of failure.*

MUFG regresses by 6.92 percentage points. HSBC by 3.24. They're *complex, multi-factor entities* whose stock prices are driven by interest rates, regulatory decisions, credit cycles, and cross-border capital flows — things that no amount of historical price pattern-fitting will capture. The model is essentially trying to predict the weather by looking at yesterday's temperature, and it works fine for companies whose "weather" is fairly persistent. But banks? Banks are driven by yield curves, loan books, and regulatory capital requirements.

This tells me something important: **the model has learned price momentum and mean-reversion patterns, not business fundamentals.** That's fine for what it is, but let's not confuse pattern recognition with understanding.

### Statistical Honesty

**Only the US results are statistically significant at the aggregate level.** Japan, Europe, and China all failed. You ran 10,000 bootstrap resamples across 15 rolling windows, and three of your four markets can't clear the bar.

Fifteen rolling windows is a small sample. You're building a routing system on the statistical equivalent of flipping a coin a dozen times and declaring you've found a biased coin because you got 8 heads.

### Would I Invest?

**No.** Here's why:

1. **No moat.** Replicable, built on someone else's open-source model.
2. **Statistical fragility.** Three of four markets fail significance testing.
3. **The wrong problem.** Reducing MAPE from 12% to 9% at 120 days is going from "badly wrong" to "somewhat less badly wrong." That's not a business.
4. **Regime risk.** Everything tested in one market regime. What happens in the next crash?

### What I Would Do Instead

1. **Publish the finding** that fine-tuning helps momentum-driven tickers and hurts complex-factor tickers.
2. **Stop trying to squeeze more MAPE.** You've hit bedrock. The answer is not a bigger shovel.
3. **Ask the right question:** Can forecast accuracy be translated into risk-adjusted returns? That requires Sharpe ratios and drawdown statistics, not MAPE.

*That's my two cents — which, after inflation, is worth about a penny and a half.*

---

## Charlie Munger — The Abominable No-Man

Let me start where I always start: **by inverting.**

### What Would Guarantee This Fails in Production?

1. **The routing table is fit to the same data it's evaluated on.** You used 15 rolling windows to classify tickers into tiers, then measured improvement on those same 15 windows. This is not out-of-sample — it's a particularly sophisticated form of overfitting. The router *cannot help but improve* on the aggregate because it's selecting the better model *after observing which model was better.* I'd bet my subscription to the Daily Journal that your "routing alpha" shrinks by 40-60% on truly unseen windows.

2. **n=3 to n=6 eval tickers per region.** You're drawing sweeping conclusions about "sector-specific finetuning" from sample sizes that wouldn't pass a freshman statistics course.

3. **The routing decision changes over time, but you're treating it as static.** MUFG is -6.92pp today. What was it 6 months ago? If the answer is "we don't know," then you don't have a deployable system — you have a retrospective analysis dressed up as a strategy.

### The Mental Model Sweep

**Engineering (feedback systems):** A static routing table is an open-loop controller. In any non-stationary system — and financial markets are the *definition* of non-stationary — open-loop control degrades. The question isn't whether the routing table will become stale, but how fast.

**Biology (overfitting as autoimmunity):** Finetuning a 200M parameter model on 10 tickers is like training an immune system on 10 antigens. MUFG and HSBC aren't "consistently harmful" — they're tickers whose statistical properties differ enough from training data that the finetuned model's learned biases actively hurt. You haven't identified *what* the model learned that's harmful, which means you can't predict which new tickers will be harmed.

**Psychology (confirmation bias, narrative fallacy):** You've constructed a beautiful narrative — "finetuning is ticker-specific, not market-wide; banking/conglomerates consistently regress." This narrative fits the data perfectly because it was constructed *from* the data. A finding would be: *why* do banking tickers regress? Without mechanism, you have pattern-matching on 15 windows with 3-6 tickers — exactly the kind of thing that makes you feel smart right before it stops working.

**Economics (transaction costs of complexity):** Every layer of complexity has a maintenance cost. Compare this to "just use baseline." The baseline requires nothing. Your marginal improvement — *statistically insignificant* in 3 of 4 regions — must justify this ongoing complexity burden.

### The Lollapalooza Effect (Working Against You)

Multiple forces converge to make this look better than it is:

- **Lookahead bias in tier classification** (you know which model won before routing)
- **Small n** (3-6 tickers amplifies noise into "findings")
- **Multiple comparisons** (20 region-horizon combinations — at p=0.05, expect 1 significant by chance; you got exactly 1: US)
- **Survivorship in the "consistently harmful" list** (you only see tickers you tested)

These aren't independent problems. They compound. That's the lollapalooza working against you.

### What the Data Actually Says (If I'm Being Honest)

Strip away the routing theater:

> "We have weak evidence that a finetuned model improves 60-120d forecasts for most but not all tickers, with a few tickers where it actively hurts. We don't know why."

That's the honest version. Everything else is decoration.

### The Pre-Mortem: Three Years From Now, This Failed

You deployed the router. For 4-6 months it looked fine. Then market regime shifted and tier classifications drifted. MUFG became Tier 1. NVDA became Tier 2. Nobody noticed for three months. By the time you recalibrated, you'd underperformed baseline for a quarter, and the organizational credibility of the system was destroyed.

### My Verdict

If you must deploy something, deploy the finetuned model at 60d+ horizons and baseline at 5-14d. That's a two-line routing table. It captures 80% of the value with 5% of the complexity. And stop running experiments — you hit the ceiling at Run 4 and everything since has been rearranging furniture.

The most valuable thing I can tell you: **know when you're done.** You're done.

---

## Michael Burry — Scion Capital

Look. I'll tell you what I see.

### The routing is a put option on your own model's failures.

That's all it is. You built a model that hurts certain tickers. Then you built a router that says "don't use the model when it hurts." The improvement from routing over FT-only is almost entirely damage avoidance, not alpha generation.

Proof:

Japan. FT-only at 14d: 5.12% vs baseline 4.55%. Your finetuned model is **57 basis points worse** than doing nothing. Router "improves" by routing everything back to baseline. That's not a routing win. That's an admission the model broke something.

Japan 30d: same pattern. FT 6.94% vs baseline 5.76%. Router returns you to 5.73%. Congratulations — you spent compute to get back to zero.

### The real signal is in the asymmetry.

US: FT-only beats baseline at every horizon except 5d. Router adds marginal value. This is the only region where finetuning actually works. Routing improvement over FT-only is noise — 2 to 14 basis points. Within any reasonable CI.

Japan: FT-only is destructive at short horizons, helpful at long. Binary regime. The model learned something about 60d+ dynamics but corrupted short-horizon representations. Layers 17-19 can't serve both masters.

CN: The confession is right there in your data. n=3 showed +6.89pp. n=9 showed +2.34pp, not significant. **You were curve-fitting to three tickers.** The 60d flipping negative on expansion is the tell — the original three were sector-aligned with training data by accident.

### The structural problem nobody's discussing.

MUFG: -6.92pp. Worst single-ticker result across all regions.

MUFG is a Japanese megabank. Interest rate sensitive, cross-border exposure, BoJ policy dependent. Your training set is semiconductor-heavy and telecom-heavy. You trained a model on Softbank and Toyota and asked it to forecast a bank that moves on yield curve dynamics.

This isn't a model failure. It's a **specification error.**

The tier classification confirms it. Tier 1 winners cluster in sectors matching training data. Tier 2/3 losers are sector mismatches. Every single one.

**Your model is a sector bet disguised as a forecasting improvement.**

### What the bootstrap is actually telling you.

Only US passes aggregate significance. But US has the smallest improvements. The regions with the largest point estimates (CN +2.69pp, JP +2.18pp, EU +1.97pp at 120d) all fail significance.

Classic. Large effect + wide CI = small sample + high variance. The bootstrap is honest. Listen to it.

### Three things hidden in plain sight:

**1.** The "improvements" at 120d are mechanically larger because the denominator is larger. A 1pp improvement at 120d is structurally easier to achieve than 1pp at 5d. You're being flattered by horizon scaling.

**2.** Route vs FT-only improvement is nearly zero for US (0.02-0.14pp) but large for Japan (0.21-1.20pp). The router's value is **inversely proportional to the model's quality.** That should concern you.

**3.** CN 60d: n=3 showed improvement. n=9 flipped to -0.42pp. The model didn't learn "Chinese equity dynamics." It learned "HK insurance stock dynamics" and you called it China.

### What I'd actually do:

Stop. Ship US Run4 as-is. The 0.68pp is real and significant. Everything else is noise dressed up in bootstrap confidence intervals. The project has found its answer. It just doesn't like it.

---

## Jim Simons — Renaissance Technologies

### Signal Validation Audit: TimesFM Sector Routing

#### Stage 0: Signal Definition

**Claim:** A bootstrap-based classification system routes each ticker to either a finetuned or baseline TimesFM checkpoint per horizon, producing lower aggregate MAPE than either model alone.

This is a model selection signal, not a trading signal. The question is whether the classification is stable, generalizable, and statistically meaningful.

#### Stage 1: Data Audit

**1.1 Sample sizes are critically small**

| Region | Total tickers | Eval tickers | Train tickers | Rolling windows |
|--------|--------------|--------------|---------------|-----------------|
| US | 12 | 6 | 6 | 15 |
| Japan | 9 | 3 | 6 | 15 |
| Europe | 9 | 3 | 6 | 15 |
| China | 9 | 3 | 6 | 15 |

The effective number of independent observations is not 15 windows x 9 tickers = 135. The windows overlap heavily -- window k and window k+1 share all but 1 day of context and all but 1 day of forecast horizon. For a 120-day horizon, consecutive windows share 119/120 = 99.2% of their evaluation period. This means the 15 windows contain roughly **1.5-2 effective independent observations**, not 15.

The bootstrap procedure resamples over windows as if they were independent. They are not. The bootstrap CIs are therefore **dramatically too narrow**.

**1.2 Survivorship bias in eval set selection**

The China case is the smoking gun. With n=3 eval tickers, the headline was +6.89pp at 120d. Expanding to n=9 shrank it to +2.34pp, and the 60d result **flipped sign** from positive to -0.42pp (P(worse)=73%). This is textbook selection bias.

**1.3 Look-ahead bias in routing**

The routing table is built on the same 15 evaluation windows that are used to measure the "improvement." This is in-sample classification applied to in-sample data. There is no temporal separation between the classification period and the evaluation period.

The correct design would be: classify on windows 1-10, evaluate on windows 11-15. This was never done.

**1.4 Horizon multiplicity**

Each ticker is evaluated at 5 horizons independently. This creates 5x the opportunity for spurious classification. A ticker that is "T1" at 30d and "T2" at 60d is not exhibiting a stable signal -- it is exhibiting noise.

#### Stage 2: Statistical Significance

**2.1 Effective sample size**

With 15 windows and 120-day horizons overlapping by 119 days:
```
N_eff ≈ 15 * (1/120) * 2 ≈ 2 independent windows
```
With 9 tickers: ~18 effective ticker-window observations. The t-statistic should be computed against ~2 degrees of freedom, not 14.

**2.2 Bootstrap CI invalidity**

The bootstrap resamples window indices treating them as iid draws. With 99% overlap at 120d, the true 95% CI is likely 5-10x wider than reported. Most "significant" results would fail with proper serial correlation correction (Newey-West or block bootstrap with block size >= horizon).

**2.3 Multiple comparisons**

Total test count conservatively 80+ (12 tickers x 5 horizons, 4 regions x 5 horizons, plus ~40 experiments). Under Bonferroni correction at alpha=0.05: p < 0.000625 required. Under Harvey-Liu-Zhu (t > 3.0), most individual improvements fail.

#### Stage 3: Out-of-Sample Validation

**There is no out-of-sample validation.**

The routing table is built on windows 0-14 and evaluated on windows 0-14. This is equivalent to sorting stocks into quintiles by past returns and then measuring the past returns of those quintiles. Of course the sorted portfolios look good -- you selected them to look good.

The entire "Route vs BL" column in the results table is in-sample.

#### Stage 4: Multiple Comparisons

40+ experiments across the project. The probability that at least one of 40 independent tests produces p < 0.05 under the null is 1 - (0.95)^40 = 87%.

The pattern of "bigger improvement at longer horizons" is suspicious -- longer horizons have fewer effective independent windows and wider true CIs.

#### Stage 5: Economic Mechanism

**Stability of tier classification** reveals troubling instability:
- 6902.T (Denso): T1 at 30d but T2 at 60d and 120d
- 0388.HK (HKEX): T2 at 120d but T3 at most other horizons
- A ticker that helps at 30d and hurts at 60d is exhibiting estimation noise, not a stable signal

**The hyperparameter knife-edge:** Only one narrow configuration (lr=1e-4, wd=0.01, freeze 17/20, cosine+5ep warmup, 10 tickers, stride=32) avoids NaN. This is not a robust optimization landscape.

#### Verdict

**What is real:**
1. Finetuning helps some tickers and hurts others (per-ticker heterogeneity is genuine)
2. The idea of model selection is sound

**What is not real:**
1. The claimed improvements are in-sample (zero OOS validation)
2. The bootstrap CIs are invalid (99% window overlap)
3. Multiple comparisons are uncorrected (80+ tests)
4. The China n=3 result is textbook selection bias
5. Tier classification is unstable across horizons

**Would I trade this?** No. Not as presented.

**What would make this convincing:**
1. **Block bootstrap** with block size >= forecast horizon
2. **Temporal out-of-sample:** Build routing table on data through date T, evaluate on data from T+1 forward
3. **Expand eval sets** to 20+ tickers per region
4. **Apply Bonferroni correction** across all tests actually run
5. **Report Sharpe ratios**, not MAPE

The mathematics here is solid in design but undermined by the data limitations. Nine tickers and fifteen overlapping windows is not a dataset -- it is an anecdote.

---

## Orchestrator Note: Temporal OOS Data Exists (Partial Rebuttal)

The investors were given `sector_router.py` results, which classify and evaluate on
the **same** 15 windows (in-sample). However, the project has separate scripts that
**do** implement temporal out-of-sample validation:

1. **`selective_ensemble_eval.py` (US):** 15 selection / 15 eval windows. The US
   selective ensemble's +0.28pp result is genuinely OOS. This corroborates Run4's
   value but uses a different routing method (per-ticker best model, not bootstrap tiers).

2. **`horizon_selective_eval_china.py` (China):** 7 selection / 8 eval windows.
   The "honest split" results in `glia_summary.md` (Glia2a +0.31pp at 5d through
   +6.89pp at 120d) use only eval windows. This is temporal OOS for the n=3 eval set.

3. **Training ticker evaluation:** The 6 training tickers per region are evaluated on
   held-out time windows (no overlap with training data). These constitute a form of
   cross-validation, though not fully independent.

**What this means for the critiques:**

| Critique | Validity |
|----------|----------|
| `sector_router.py` "what-if" MAPE is in-sample | **Valid** — the routing table numbers are in-sample |
| No temporal OOS exists anywhere in the project | **Overstated** — honest-split scripts exist and corroborate |
| Bootstrap CIs too narrow (99% window overlap at 120d) | **Valid** — 1-day step size creates near-complete overlap |
| n=3 eval tickers insufficient | **Valid** — China n=9 expansion confirmed this |
| Tier classification unstable across horizons | **Valid** — not addressed by honest-split scripts |

**Recommended fix:** Add `--honest-split` flag to `sector_router.py` that classifies
on windows 0 to N/2 and computes "what-if" MAPE only on windows N/2+1 to N.

---

## Cross-Investor Consensus

All four investors independently converge on these points:

| Finding | Buffett | Munger | Burry | Simons |
|---------|---------|--------|-------|--------|
| Routing improvement is in-sample | Yes | Yes | Yes | Yes |
| Sample sizes inadequate | Yes | Yes | Yes | Yes |
| Only US statistically significant | Yes | Yes | Yes | Yes |
| Router hedges damage, not generates alpha | - | - | Yes | - |
| Sector-specificity finding worth publishing | Yes | Yes | Yes | - |
| Project has hit ceiling — stop experimenting | Yes | Yes | Yes | Yes |
| No moat / not a deployable business | Yes | Yes | - | - |
| Need temporal OOS validation | - | Yes | - | Yes |
| Need block bootstrap for honest CIs | - | - | - | Yes |
| Need 20+ tickers per region | - | - | - | Yes |
| Evaluate using Sharpe, not MAPE | Yes | - | - | Yes |

### Actionable Recommendations (consensus)

1. **Ship US Run4 as-is** — the only statistically significant result
2. **Stop running experiments** — ceiling hit, diminishing returns
3. **If deploying routing:** use horizon-based (FT for 60d+, baseline for 5-14d) — captures 80% value at 5% complexity
4. **Before trusting routing tables:** implement temporal OOS validation and block bootstrap
5. **Expand eval sets** to 20+ tickers before making deployment decisions
6. **Publish the sector-specificity finding** as a research contribution
7. **Evaluate with Sharpe ratios** to connect forecast accuracy to economic value
