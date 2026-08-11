# GSM-Alpha

A PyTorch-Lightning implementation of 东北证券 2024-06-03,
《GSM-Alpha：提取时序特征的统一框架 —— 机器学习系列之五》.

Two things are implemented:

* **GSM** — the *Generalized Signature Method* of report section 2 (Morrill,
  Fermanian, Kidger & Lyons 2021), the four-stage framework
  `rescale ∘ transform ∘ window ∘ augment` that all the signature-method variants
  in the literature fall out of. Every option in tables 1–7 is available, not
  just the one the report ends up using.
* **GSM-Alpha** — the stock-selection network of section 3: GSM feature
  extraction on 5-minute and daily bars, indicator mixing, cross-stock attention,
  and the weighted-correlation objective, retrained on the section 3.2 rolling
  yearly schedule.

```
gsm_alpha/
  signature/     lyndon.py  torch_backend.py  backend.py  gsm.py
  data/          sources.py  preprocess.py  labels.py  cache.py  datamodule.py
  models/        gsm_alpha.py  loss.py
  config.py  train.py  evaluate.py  cli.py
configs/default.yaml     the report's own setting
env/create_env.sh        the environment, and why every pin is there
scripts/export_labels.py the residual-return training target, exported to parquet
tests/                   88 tests, including signatory-vs-fallback parity
```

---

## 1. The signatory constraint

`signatory` is a C++ extension compiled against a specific PyTorch, and its last
release (1.2.6.1.9.0) targets **torch 1.9.0**, which publishes no wheels beyond
Python 3.9. That single pin propagates through the whole stack, and
`env/create_env.sh` documents each consequence inline:

| pin | why |
|---|---|
| python 3.9 | torch 1.9 has no 3.10+ wheels |
| torch 1.9.0+cpu | the version signatory 1.2.6.1.9.0 builds against |
| pip < 24.1 | pytorch-lightning 1.6.5 ships metadata newer pip rejects |
| setuptools 59.5 | torch 1.9's tensorboard shim reads `distutils.version`, gone in 60 |
| torchmetrics 0.9.3 | 1.x requires torch ≥ 1.10 |
| numpy < 2 | torch 1.9 was built against the numpy 1.x C API |

Two ordering traps are handled in the script: signatory needs torch importable
*before* it builds (`--no-build-isolation`), and installing pytorch-lightning
drags in a modern torch, so torch is force-reinstalled and re-pinned afterwards.

```bash
bash env/create_env.sh gsm     # ~5 minutes, compiles signatory from source
conda activate gsm
```

**The project does not actually require signatory.**
`gsm_alpha/signature/torch_backend.py` is a from-scratch signature and
log-signature transform in plain PyTorch — signature by Chen's identity folded
divide-and-conquer over the path's segments, log-signature by the tensor-algebra
logarithm read off at Lyndon-word positions (signatory's `mode="words"`). It is
differentiable, runs on any PyTorch including 2.x, and
`tests/test_signature_backend.py` pins it to signatory's output to 1e-10 across
every `(channels, depth, length)` combination the report uses. `backend: auto`
prefers signatory when it imports and falls back silently when it does not, so
you can develop on a modern interpreter and train on the pinned one.

This is verified rather than asserted. Beyond the in-suite parity tests, the
same 800-dimensional GSM output was computed twice on identical input — once
under python 3.9 / torch 1.9 / signatory, once under python 3.13 / torch 2.13 /
the fallback — and agreed to **1.6e-13**.

signatory is nonetheless roughly two orders of magnitude faster, which matters:
at depth 5 over 3 channels it turns ~2 000 stock-samples per second of 20-day
5-minute path on 8 threads. Use it for cache builds; the fallback is for
portability and for checking.

---

## 2. Data contract

The pipeline reads exactly **two parquet files**, and nothing else about the
deployment is assumed. Point `configs/default.yaml` at your own and it runs.

**Daily** — columns `open, high, low, close, volume`, keyed by `date` (a date)
and `stable_id` (an integer security id). The key may be a pandas MultiIndex or
two ordinary columns.

**Intraday** — the same five columns keyed by `date`, `time` (a time of day) and
`stable_id`. Every date must carry the same intraday grid. Reading is much
faster when the file is written one row group per date, which the reader detects
and exploits; it is not required.

Missing observations may be absent rows or `NaN`; both are handled.

The defaults point at this workspace's copies:

```
/tq/scratch/zhangchang/data/cn_equity/pv/daily_ohlcv_20160101_20251231.parquet
/tq/scratch/zhangchang/data/cn_equity/pv/intraday/rq_5m_ohlcv_20160101_20251231.parquet
```

Two facts about that particular data the reader works out for itself, rather
than hard-coding:

* The intraday bar labels are **right-edge**, so the `09:30` and `13:00` labels
  are empty by construction (`13:00` would span the lunch break). They are
  dropped, leaving **48 bars a day**, and a 20-day window is 960 steps.
* The grid is probed from **eight dates spread across the file**, not one.
  Probing only the first date would have been a silent disaster: 2016-01-04 is
  the first circuit-breaker day and halted at 13:34, so every afternoon bar
  would have been deleted from the entire history.

---

## 3. Running it

```bash
conda activate gsm

# 1. Precompute per-date model inputs (one streaming pass over both panels).
python -m gsm_alpha.cli build-cache --stage features --threads 32

# 2. Rolling yearly retrain; writes runs/default/factor.parquet
python -m gsm_alpha.cli train

# 3. Rank IC / ICIR / quintile long-short, the report's own metrics
python -m gsm_alpha.cli evaluate

# resolved config, live signature backend, cache coverage
python -m gsm_alpha.cli info
```

Any config key can be overridden from the command line, repeatably:

```bash
python -m gsm_alpha.cli --set train.max_epochs=5 --set data.universe_top_n=500 train
```

### Why there is a cache

A 20-day intraday window is 960 bars per name, and consecutive sample dates
share 19 of their 20 days. Building windows inside the training loop would
re-decode the same parquet row group twenty times. `build-cache` walks dates in
order with a rolling buffer, reads each date exactly once, and writes one `.npz`
per date. The rolling schedule then consumes each date about four times (four
fits overlap it), and every ablation re-consumes all of them.

Because the report's chosen augmentation — coordinate projections — has **no
parameters**, its GSM output is fixed and can be cached directly, so GSM
disappears from the training loop entirely. A *learnable* augmentation
(`multi_headed_stream_preserving`, the section 3.1 alternative) cannot be, and
the builder refuses `--stage features` for it with an explanatory error; build
`--stage windows` and train `--kind windows` instead, which runs GSM inside the
forward pass.

The build is resumable — it skips dates already present — so it can be
interrupted and restarted freely, and extended later by moving `end_date`.

**Rough cost**, measured on this workspace (185 cores, CPU only, signatory):
137 sample dates over 5.5 years at a 300-name universe took ~5 minutes,
including streaming all 1 350 intervening dates. Scaling that, a full daily
cache over 2016–2025 at 1 500 names is on the order of an hour and ~23 GB
(1 600 float32 per stock-date). `train_day_stride`, `universe_top_n` and the
date range are the three knobs that move it.

---

## 4. What the model does

### GSM (section 2)

```
z_{i,j} = (ρ_post ∘ S_N ∘ ρ_pre ∘ W^j ∘ φ^i)(x)
```

| stage | implemented | report's choice |
|---|---|---|
| augment `φ` | none, time, basepoint, invisibility-reset, lead-lag, coordinate projection (singletons/pairs/triplets), random projection, learnt projection, multi-headed stream-preserving | coordinate pairs → time → basepoint |
| window `W` | global, sliding, expanding, hierarchical dyadic | global |
| transform `S_N` | signature, log-signature, any depth | log-signature, depth 5 |
| rescale `ρ` | none, pre (`(N!)^{1/N}`), post (`k!`) | none |

Under the report's setting, 5 OHLCV channels become C(5,2)=10 pair-streams of
`(t, xᵢ, xⱼ)` with a prepended origin; each yields 80 depth-5 log-signature
channels, so **each branch produces 800 features** per stock per date.

The augmentations are not decoration. Without basepoint, a price path shifted by
a constant has an *identical* signature; without the time channel, two paths that
differ only in sampling density are indistinguishable. Both are properties you
want for handwriting recognition and emphatically do not want for prices —
`tests/test_gsm.py` asserts each invariance and then asserts that the
corresponding augmentation breaks it.

### The network (section 3.1, figure 4)

```
per stock:  GSM → Linear → [LayerNorm → Linear → GELU → Linear] ⊕ → h        (per branch)
            concat(h_minute, h_daily)
per day:    MultiheadAttention over the stocks → LayerNorm ⊕ → Linear → factor
```

A batch is **one trading day's entire cross section**. That is what makes stock
mixing well defined — each name attends over its peers, which is where the
table 9 information gain comes from — and attention rather than an MLP because
the number of listed names changes daily.

### The objective (section 3.2)

Negative *weighted* correlation, weights exponentially decaying in the rank of
the **predicted** factor value with half-life = batch size / 2. Concentrating
the fit on the top of the book is what the report means by reducing the 空头效应.
The weights come from a ranking, so they are piecewise constant and carry no
gradient; they are detached, and the model learns by moving predictions rather
than by gaming its own weights.

### The rolling schedule (section 3.2)

One model per prediction year, fitted on the four preceding years: the first
three are training, the fourth validation, and each split's final month is
dropped so a 20-day forward label never crosses a split boundary. Early stopping
patience 30, max 100 epochs. Concatenating the years gives a factor series where
every value came from a model that never saw its date.

---

## 5. Deviations from the report, and why

These are the places where the report and this data could not both be honoured.
All are config knobs, and all default as noted.

**1. Label neutralisation is opt-in** (see §7 for how to switch it on). The
report's target is the industry- and market-cap-neutralised, cross-sectionally
standardised 20-day forward return. An OHLCV panel contains neither industry nor
market cap, so the target is pluggable in two ways, and with neither the label is
only standardised — `LabelBuilder` logs a warning saying so. This matters for
interpreting results: the report attributes its factor's *low* market-cap
correlation (table 11) specifically to having neutralised the training target, so
without one expect a stronger size tilt than it reports.

**2. `price_zscore: joint`.** Section 3.2 says prices are divided by the
sample's last close and then z-scored along time. Those two steps are only
jointly meaningful if the z-score shares one mean and standard deviation across
the four price channels — a *per-channel* z-score is invariant to a positive
scalar divisor, which would make the division a no-op and would also destroy the
level relationship between the channels (a bar's high would stop sitting above
its close). `joint` is the reading under which step 3 does something;
`per_channel` is available, and `tests/test_data.py` demonstrates both facts.

**3. `train_day_stride: 5`.** The report samples every trading day. The label is
a 20-day forward return, so consecutive daily samples overlap by 19/20 and carry
very little independent information, while costing 5× the compute. Set to `1` to
reproduce the report exactly.

**4. `universe_top_n: 1500`.** The report uses the whole market. A top-N
liquidity screen (trailing 60-day median dollar volume, computed only from
information available on the date) keeps a first cache build to a sane size. Set
to `null` for the full market. Note this cuts against the grain of the reported
result: table 14 shows the factor is *strongest* in small-cap pools (国证2000
Rank IC 13.37% vs 沪深300 6.58%), so a liquidity screen should be expected to
understate it.

**5. No index-enhancement backtest.** Section 3.4's portfolios need a
constrained optimiser and a risk model (2% weight cap, 5% tracking error, ±2%
industry bands, 0.3σ style bands) that this project does not carry.
`gsm_alpha/evaluate.py` reports Rank IC, ICIR, quintile long-short and the
per-year breakdown — the section 3.3 tables — and stops there.

**6. Price adjustment is whatever the input file contains.** The preprocessing
z-scores each sample, so an unadjusted split or ex-dividend gap inside a window
is a real distortion. Roughly 0.006% of daily returns in the bundled panel
exceed ±50%, which is consistent with either reading; if your panel is
unadjusted, adjust it upstream.

---

## 6. Tests

```bash
python -m pytest tests/ -q        # 75 tests, ~1s
```

The suite is written to catch the failures that actually happen here rather than
to cover lines. Notably:

* **Backend parity.** The pure-PyTorch signature and log-signature are compared
  against signatory across every `(d, depth, length)` combination, plus a
  `gradcheck`. This caught two real bugs during development — a divide-and-conquer
  fold that paired non-adjacent path segments, and a recursion that produced
  `dx^{⊗k}/k` instead of `dx^{⊗k}/k!`. Both passed a shape check happily.
* **Mathematical properties**, independent of signatory: Chen's identity, the
  single-segment tensor exponential, reparametrisation invariance.
* **Augmentation semantics**: that basepoint really does break translation
  invariance, that lead-lag staggers correctly, that the composed report setting
  yields exactly 10 streams of 3 channels with a zero first row.
* **Split arithmetic**: that no fitted date reaches into the predicted year and
  that each split's final month is dropped, i.e. that the 20-day label cannot
  leak across the boundary.
* **Objective**: that flat weights reduce to Pearson, that the half-life is what
  it claims, that a perfect prediction minimises it, and that stock mixing lets
  one stock's input change another stock's output (and that disabling it stops
  that happening).

---

## 7. The training target: using a risk-model residual return

The most faithful way to reproduce the report's target on this platform is not to
neutralise anything yourself — it is to train on a residual return the data lake
already computes. The tq zoo's `resp_res_*` family is the report's idea done more
thoroughly: the return residualised against a full RQ risk model (country,
industry, and the entire style block including size), then shifted forward.

```bash
# on a machine WITH the tq data lake
python scripts/export_labels.py --out /data/gsm/labels_res20d.parquet \
    --start 2016-01-01 --end 2025-12-31
```

That writes one 66 MB parquet — 8.06 M rows, 2 391 trading days, ~3 370 names per
date — in the same `(date, stable_id, value)` contract as everything else. Ship
it with the panels; the training box never needs tq, DataAlchemy or the lake.

```yaml
labels:
  forward_label_path: /data/gsm/labels_res20d.parquet
  forward_label_column: resp_res_rq_cnltb_1500to1500_20d
  forward_label_scale: 0.0001
```

### Which residual — a correction worth knowing

`res_rq_cntrb` is the house default for factor evaluation, but **it is only
materialised out to `_10d`**. At 15d/20d/30d/40d the family on disk is
`res_rq_cnltb`. So at the report's 20-day horizon `res_rq_cntrb` is not
available, and you want:

| | model | horizons on disk |
|---|---|---|
| `res_rq_cntrb` | CNTR + BJSE — RQ **trading** (short-horizon) | 1d … 10d |
| `res_rq_cnltb` | CNLT + BJSE — RQ **long-term** | 1d … 40d, incl. **20d** |

This is not a downgrade. CNLT is the long-horizon model, so it is the
better-matched choice for a 20-day holding period anyway. If you would rather
keep `cntrb`, set `labels.horizon: 10` and use
`resp_res_rq_cntrb_1500to1500_10d`.

The `1500to1500` slot is close-to-close, which matches the report's plain 20-day
forward return; the other eight slots (`930to930`, `1000to1000`, …) are available
if you want the label to match a specific execution time.

### Alignment and units, verified

`resp_*` is **already forward-shifted**, so nothing shifts it again — the builder
raises if you also pass `exposures_path`, since residualising a residual would be
wrong. Checked against the price panel:

| label at `t` vs total 20d return starting | correlation |
|---|---|
| `t − 20` | +0.02 |
| **`t`** | **+0.80** |
| `t + 20` | +0.04 |

It is the forward return for its own row's date, and the correlation is below 1.0
exactly because the risk factors have been stripped out. The series is in **basis
points** (measured: 1 unit ≈ 1.08e-4 of return), which is irrelevant to training —
the label is cross-sectionally standardised — but `forward_label_scale: 0.0001`
makes the evaluator print real percentages.

### Score it against what you trained on

A model fitted on residual returns deliberately ignores the style and industry
beta that dominates the cross section of *total* returns, so scoring it on total
returns understates it. `evaluate --against-label` scores against the same series:

```bash
python -m gsm_alpha.cli evaluate --against-label
```

The same factor from the smoke run, scored both ways — note the Sharpe, which is
what actually changes:

| scored against | Rank IC | ICIR | L/S annual | L/S Sharpe |
|---|---|---|---|---|
| total close-to-close 20d | 0.070 | 1.96 | 22.4% | 1.26 |
| residual 20d | 0.059 | 2.33 | 18.5% | **1.79** |

---

## 8. Running on a rented cloud box

The code moves as-is. Copy the repo plus three files, and nothing else:

| file | size | from |
|---|---|---|
| `daily_ohlcv_*.parquet` | 148 MB | your data lake |
| `rq_5m_ohlcv_*.parquet` | 6.2 GB | your data lake |
| `labels_res20d.parquet` | 66 MB | `scripts/export_labels.py` |

Then edit the three paths in `configs/default.yaml`. Nothing else is machine
specific — no tq, no DataAlchemy, no data lake.

### Split the work: cache on CPU, train on GPU

The pipeline is already built for this. `build-cache` runs the signatures and
writes plain per-date files; `train` reads those and never touches GSM. So the
two stages can live on different machines with different stacks:

| stage | hardware | needs signatory? |
|---|---|---|
| `build-cache` | CPU, many cores | **yes** (or the slower fallback) |
| `train` / `evaluate` | GPU | **no** — features are precomputed |

That split is what makes a modern GPU usable at all: torch 1.9 cannot run on
sm_89/sm_90 cards, but the training box only needs a current torch and the
pure-python backend, which it never even calls in feature mode.

**Full-market training is the part that needs the GPU, and CPU will not do it.**
Measured here, a training step at 4 300 names takes 970 ms on 32 threads, 956 ms
on 64, and 992 ms on 128 — it **does not scale with cores at all**. The reason is
visible in a breakdown: the stock-mixing attention is 99% of the forward pass and
achieves only ~240 GFLOP/s, because a 4 300 x 4 300 x 4-head score matrix is
296 MB and the kernel is memory-bandwidth bound, not compute bound. More cores do
not add bandwidth. A GPU's HBM does, by an order of magnitude, and `precision: 16`
halves the traffic again — which is why this workload is a good GPU candidate
even though the model is small.

One epoch at full market, measured on CPU:

| | 32 threads | 64 | 128 |
|---|---|---|---|
| stride 1 | 12.6 min | 12.5 min | 12.9 min |
| stride 5 | 2.5 min | 2.5 min | 2.6 min |

**Measure before you commit.** `benchmark` times a real step on whatever box you
are on and projects the whole schedule from it:

```bash
python -m gsm_alpha.cli --config configs/paper.yaml benchmark --device cuda --precision 16
python -m gsm_alpha.cli --config configs/paper.yaml benchmark --threads 64   # CPU
```

It takes seconds and tells you the hours. Run it first.

### Keep the GPU fed

Once a step is milliseconds, per-batch disk reads become the bottleneck — a
full-market day-batch is ~28 MB. `data.preload` (default `auto`) loads a whole
split into RAM once, after which every epoch is pure compute and DataLoader
workers are turned off automatically. Budget: a 4-year training split at full
market is **~27 GB**, so `preload_max_gb: 32` and a box with ~48 GB of RAM.

`feature_dtype: float16` would halve that and is tempting — but do not: depth-5
log-signatures reach ~264 000 and float16 overflows at 65 504. The build now
**refuses the cast** rather than write silent infinities, which is how this was
caught; before the guard it produced `inf` features, `NaN` predictions and a
crash 20 minutes in.

### CPU thread count is not automatic

`torch_threads` defaults to 0, meaning torch's own default — which is derived
from the reported CPU count and is wrong in many containers. This box reports 1
core while having 185, so CPU training would silently run single-threaded and
take 8x longer. **Set `train.torch_threads` explicitly whenever training on CPU.**

### Two environments, one per stage

```bash
bash env/create_env.sh      gsm       # CPU cache box: pinned, builds signatory
bash env/create_env_gpu.sh  gsm-gpu   # GPU train box: current torch, no pins
```

The GPU script deliberately does **not** install signatory, and does not need
to: `train` reads cached features and never calls the signature code. That is
what sidesteps the pin entirely — torch 1.9's newest CUDA build is cu111, which
will not run on an sm_89/sm_90 card (4090, L40S, H100) at all.

Tested, not hoped for: the **entire suite, including the rolling-fit integration
test, passes unmodified on Python 3.13 / torch 2.13 / Lightning 2.6.5** with no
signatory present, and the fallback produces GSM features identical to
signatory's to 1.6e-13.

If your card is older (V100, A100 — sm_70/sm_80) you can instead use the pinned
stack everywhere: swap the torch lines in `env/create_env.sh` to
`torch==1.9.0+cu111` and rebuild signatory against it.

### `configs/paper.yaml` runs unchanged on either box

It asks for `precision: 16` and `accelerator: auto`, which is what you want on a
GPU. On a CPU box the same file still works: `resolve_hardware` downgrades to 32
and logs why. That matters because Lightning 1.6 *raises* for `precision=16` on
CPU, and Lightning 2.x silently substitutes bfloat16 — so neither the crash nor
the false timing can reach you.

Set `accelerator: gpu` explicitly if you would rather the run fail loudly than
fall back to CPU on a box where the driver is missing.

### A sensible first run

```bash
# ~1 hour, ~23 GB of cache
python -m gsm_alpha.cli build-cache --threads 32

python -m gsm_alpha.cli train
python -m gsm_alpha.cli evaluate --against-label
```

Start with `universe_top_n: 1500` and `train_day_stride: 5` to get an end-to-end
result, then set them to `null` and `1` for the full-market, every-day run the
report describes. Watch the disk: the cache is ~1 600 float32 per stock-date, so
the full market at stride 1 is several times the 23 GB figure above.

---

## 9. Reproducing the report's numbers

`configs/paper.yaml` is the report's setting: full market, every trading day,
20-day industry- and market-cap-neutralised target, factor period ending
2024-05, early stopping 30 / max 100 epochs.

```bash
python scripts/export_exposures.py --out /data/gsm/exposures.parquet
python -m gsm_alpha.cli --config configs/paper.yaml build-cache --threads 32
python -m gsm_alpha.cli --config configs/paper.yaml train
python -m gsm_alpha.cli --config configs/paper.yaml evaluate \
    --rebalance monthly --groups 5 --neutralize-factor /data/gsm/exposures.parquet
```

### 2018 and 2019 cannot be reproduced

The report rolls yearly from 2018 on a 4-year lookback, so its 2018 fit trains on
2014–2016 and validates on 2017. **Five-minute bars in this data lake begin
2016-01-04** — there is no 2014 or 2015 intraday history to fall back on — and
after the 60-day warmup the first usable sample is 2016-04-01.

| predict year | needs training data from | status |
|---|---|---|
| 2018 | 2014-01 | **impossible** |
| 2019 | 2015-01 | **impossible** |
| 2020 | 2016-01 | partial — 2016 starts in April, ~8% of the training set lost |
| 2021 → | 2017-01 | fully clean |

`first_predict_year: 2020` is the shipped compromise; set it to `2021` if you
want every fit strictly clean. Either way, **compare against the report's
per-year tables 12 and 13, not its headline aggregate**, which covers
2018-01…2024-05 and yours will not.

### The report's per-year numbers, for the years you can reach

Table 12, GSM-Alpha **after** factor neutralisation — compare with
`evaluate --neutralize-factor`:

| year | Rank IC | ICIR | 多头年化超额 | 多空年化 | 多空 Sharpe |
|---|---|---|---|---|---|
| 2020 | 12.84% | 2.48 | 13.98% | 38.53% | 4.34 |
| 2021 | 10.80% | 1.60 | 12.91% | 33.92% | 3.88 |
| 2022 | 10.02% | 2.49 | 8.08% | 31.77% | 4.99 |
| 2023 | 12.47% | 2.55 | 12.95% | 32.51% | 6.19 |
| 2024¹ | 10.49% | 1.82 | 7.51% | 42.98% | 4.98 |

Table 13, **un-neutralised** — compare with plain `evaluate`. This arm needs no
exposures file at all, because the report's other two preprocessing steps
(去极值 and 标准化) are monotone and so change neither Rank IC nor the quintile
buckets:

| year | Rank IC | ICIR | 多头年化超额 | 多空年化 | 多空 Sharpe |
|---|---|---|---|---|---|
| 2020 | 15.85% | 2.67 | 22.11% | 49.54% | 3.93 |
| 2021 | 10.71% | 1.10 | 11.52% | 32.99% | 2.36 |
| 2022 | 12.31% | 1.27 | 11.03% | 36.07% | 3.31 |
| 2023 | 12.75% | 1.28 | 8.53% | 25.68% | 2.56 |
| 2024¹ | 11.92% | 1.03 | 17.64% | 57.55% | 4.54 |

¹ January–May only.

The direction of the neutralisation effect is already visible in the smoke run —
Rank IC falls, stability rises — matching the report's table 10 (12.19% / ICIR
2.26 neutralised versus 13.33% / ICIR 1.62 raw):

| smoke factor | Rank IC | ICIR | L/S Sharpe |
|---|---|---|---|
| raw | 0.070 | 1.96 | 1.26 |
| neutralised | 0.050 | 2.26 | 1.99 |

### Remaining differences to keep in mind

* **Benchmark.** The report's 多头年化超额 is against an equal-weighted all-A
  basket; `evaluate` reports `top_minus_equal_weight`, which is the same idea
  computed within the scored universe.
* **Universe filters.** The report excludes 北交所 and ST/\*ST names. This
  pipeline does not, because the OHLCV contract carries neither flag. Add them
  as exposure-style columns if it matters to you.
* **Price adjustment.** As in §5, whatever the input panel contains.
* **Label breadth.** `configs/paper.yaml` uses industry + log market cap, which
  is what the report says. `scripts/export_labels.py`'s risk-model residual is
  the stronger target but is *not* what the report trained on — do not mix the
  two when the goal is a like-for-like comparison.

### What it will cost

Measured on this workspace (32 threads, CPU only, signatory), per trading date:
5-minute decode 0.065 s, normalisation 0.39 s, GSM 0.40 s at a 1 500-name
universe. The stock-mixing attention is **O(stocks²)**, which is what makes the
full market expensive: a training step is 132 ms at 1 500 names, 460 ms at
3 000, and 916 ms at 4 300.

| universe | stride | cache | min/epoch | 30 ep | 60 ep | 100 ep |
|---|---|---|---|---|---|---|
| 1 500 | 5 | 0.6 h | 0.3 | 1.8 h | 3.0 h | 4.6 h |
| 1 500 | 1 | 0.6 h | 1.7 | 6.6 h | 12.6 h | 20.6 h |
| 4 300 | 5 | 1.6 h | 2.4 | 10.0 h | 18.3 h | 29.5 h |
| **4 300 (paper)** | **1** | **1.6 h** | **11.9** | **43 h** | **85 h** | **141 h** |

Totals are wall clock for the cache plus all rolling fits at that epoch count;
early stopping (patience 30, max 100) usually lands in the middle column. Cache
size is ~6.4 KB per stock-date: 23 GB at 1 500 names, **67 GB at full market**.

So the faithful run is on the order of **three to four days on 32 cores**, and
the cache is a one-time cost shared by every later ablation. A practical
sequence is to do `configs/default.yaml` first for an end-to-end result in about
three hours, then commit to `configs/paper.yaml`.
