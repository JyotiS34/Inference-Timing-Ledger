# Inference Timing Ledger

A dependency-free CSV schema and validator for recording, exchanging, and sanity-checking
LLM inference timing measurements. Reference model: **Qwen/Qwen2.5-0.5B-Instruct**.

|  |  |
|---|---|
| **Author** | Jyotirmoy Saikia — jyotirmoysaikia05153@gmail.com |
| **Attribution** | "Jyotirmoy Saikia: inference timing ledger" |
| **Schema version** | 1.0 |
| **Validator** | `validate_ledger.py` — Python 3.8+, standard library only |
| **License** | MIT |

## Contents

| File | What it is |
|---|---|
| `inference_timing_ledger.csv` | The timing-results template: 36 columns + 7 synthetic example rows |
| `validate_ledger.py` | Schema and sanity validator (CLI, `--strict`, `--json`) |
| `examples/ledger_invalid_example.csv` | 17 deliberately broken rows used to verify the validator rejects bad data |
| `.github/workflows/ci.yml` | Runs the validator on every push |

## Quick start

```console
$ python3 validate_ledger.py inference_timing_ledger.csv --strict
inference timing ledger validator (schema v1.0)
file: inference_timing_ledger.csv
data rows checked: 7

no issues found

result: VALID - 0 error(s), 0 warning(s)
```

No installation, no dependencies: any Python 3.8+ works.

## Design goals

- **One row = one measured configuration** (device × precision × workload × batch size).
- Everything a reviewer needs to judge a timing claim: methodology (warmup,
  synchronization), the full latency distribution, peak memory, and exact software versions.
- Strict enough to catch the usual benchmarking mistakes, lenient enough for
  heterogeneous hardware (CPU, CUDA, Apple Silicon, NPUs).
- Zero dependencies, usable in CI.

## Schema (v1.0)

36 columns. Required means the cell must be non-empty; empty cells are only allowed
where noted. Latency is recorded in **milliseconds**; memory in **MB/GB as named in
the column**.

### 1 · Run identity and provenance

| Column | Type | Required | Notes |
|---|---|---|---|
| `schema_version` | string | yes | must be `1.0` |
| `run_id` | string | yes | unique, no whitespace; recommended pattern `[A-Za-z0-9][A-Za-z0-9._-]*`, e.g. `qwen05b-a10g-bf16-i512-o128-b01-001` |
| `recorded_at_utc` | timestamp | yes | ISO-8601 UTC, `YYYY-MM-DDTHH:MM:SSZ` |
| `recorded_by` | string | yes | person who ran the measurement |
| `contact_email` | string | yes | contact for questions about the row |

### 2 · Model

| Column | Type | Required | Notes |
|---|---|---|---|
| `model_id` | string | yes | Hub-style `org/name`, e.g. `Qwen/Qwen2.5-0.5B-Instruct` |
| `model_revision` | string | yes | commit SHA or tag the run pinned (`main` acceptable; a full SHA is better) |

### 3 · Device

| Column | Type | Required | Notes |
|---|---|---|---|
| `device_type` | enum | yes | `cpu` `gpu` `mps` `npu` `tpu` `other` |
| `device_name` | string | yes | e.g. `NVIDIA A10G`, `Intel Xeon Gold 6338`, `Apple M2 Pro` |
| `device_count` | int ≥ 1 | yes | number of accelerators used |
| `device_memory_total_gb` | float > 0 | recommended | total memory of one device; lets readers interpret `peak_memory_mb` |

### 4 · Precision

| Column | Type | Required | Notes |
|---|---|---|---|
| `precision` | enum | yes | `fp32` `fp16` `bf16` `fp8` `int8` `int4` |
| `quantization_method` | enum | yes | `none` `bitsandbytes` `gptq` `awq` `other` |

### 5 · Workload and timing methodology

| Column | Type | Required | Notes |
|---|---|---|---|
| `input_len_tokens` | int ≥ 1 | yes | prompt length in tokens |
| `output_len_tokens` | int ≥ 0 | yes | generated tokens per request; `0` = prefill-only row |
| `batch_size` | int ≥ 1 | yes | requests per timed iteration |
| `warmup_runs` | int ≥ 0 | yes | untimed warmup iterations (≥ 1 recommended on gpu/mps) |
| `timed_runs` | int ≥ 1 | yes | timed iterations behind the distribution (≥ 10 recommended) |
| `sync_method` | enum | yes | `device_sync` = `torch.cuda.synchronize()` / `torch.mps.synchronize()` around each timed region · `cuda_events` = CUDA event pairs · `host_timer_only` = host wall-clock without device sync (warns on gpu) · `not_applicable` = CPU-only timing |

### 6 · Latency distribution (ms, per timed iteration for the whole batch)

| Column | Type | Required | Notes |
|---|---|---|---|
| `latency_ms_min` … `latency_ms_max` | float > 0 | yes | `min`, `p50`, `mean`, `p90`, `p95`, `p99`, `max`; must satisfy min ≤ p50 ≤ p90 ≤ p95 ≤ p99 ≤ max and min ≤ mean ≤ max |
| `latency_ms_stddev` | float ≥ 0 | yes | standard deviation across `timed_runs` |

### 7 · Throughput and memory

| Column | Type | Required | Notes |
|---|---|---|---|
| `throughput_output_tokens_per_sec` | float ≥ 0 | yes | `batch_size × output_len_tokens ÷ (latency_ms_mean / 1000)`; `0` when `output_len_tokens = 0`; cross-checked by the validator |
| `peak_memory_mb` | float > 0 | yes for gpu/mps | peak accelerator memory during timing, e.g. `torch.cuda.max_memory_allocated() / 1024**2`; may be empty on cpu |

### 8 · Software

| Column | Type | Required | Notes |
|---|---|---|---|
| `inference_engine` | string | yes | e.g. `pytorch-transformers`, `vllm`, `llama.cpp` |
| `python_version` | string | yes | e.g. `3.11.9` |
| `torch_version` | string | recommended | empty only when the engine does not use torch |
| `transformers_version` | string | recommended | ditto |
| `cuda_version` | string | gpu runs | empty on cpu/mps |
| `os_version` | string | yes | e.g. `Ubuntu 22.04.5 LTS`, `macOS 15.1` |

### 9 · Free text

| Column | Type | Required | Notes |
|---|---|---|---|
| `notes` | string | no | anything unusual about the run |

### Coverage of the requested fields

| Requested | Columns |
|---|---|
| device | `device_type`, `device_name`, `device_count`, `device_memory_total_gb` |
| input length | `input_len_tokens` (+ `output_len_tokens` to separate prefill from decode) |
| batch | `batch_size` |
| precision | `precision`, `quantization_method` |
| warmup | `warmup_runs` |
| sync | `sync_method` |
| latency distribution | `latency_ms_min/p50/mean/p90/p95/p99/max/stddev` |
| peak memory | `peak_memory_mb` |
| software versions | `inference_engine`, `python_version`, `torch_version`, `transformers_version`, `cuda_version`, `os_version` |

Additions beyond the requested list, and why: `schema_version` (schema evolution),
`run_id` (stable reference), `recorded_at_utc` / `recorded_by` / `contact_email`
(provenance), `model_id` / `model_revision` (exact artifact measured), `timed_runs`
(sample size behind the percentiles), `throughput_output_tokens_per_sec` (capacity
metric, checked for consistency), `notes` (free context).

## Measurement recommendations

- Warm up before timing (≥ 5 iterations on GPU); never include the first iterations
  (lazy init, kernel autotuning, allocator warm-up).
- Synchronize the device around every timed region (`torch.cuda.synchronize()` /
  `torch.mps.synchronize()`) or use CUDA events. Host-side timers alone under-report
  asynchronous kernels — the validator warns about it.
- Reset peak-memory statistics after warm-up (`torch.cuda.reset_peak_memory_stats()`),
  then read `torch.cuda.max_memory_allocated() / 1024**2` for `peak_memory_mb`.
- Use as many timed iterations as practical (≥ 100 when feasible) and report the whole
  distribution, not just the mean — tail percentiles are usually what users feel.
- Pin what you can (fixed input/output lengths, deterministic settings) and record
  everything you cannot pin in `notes`.
- One configuration per row. To correct a published row, append a new row with a new
  `run_id` and a note instead of editing history.

## Throughput vs. single-request latency

Two different questions, two different columns:

- **Latency** (`latency_ms_*`) answers *"how long does one request wait?"* — the
  wall-clock time from submission to completion, in milliseconds, experienced by a
  single user.
- **Throughput** (`throughput_output_tokens_per_sec`) answers *"how many tokens per
  second can the system deliver in aggregate?"* — a capacity metric, defined as
  `batch_size × output_len_tokens ÷ (latency_ms_mean / 1000)`.

Three rules to keep them apart:

1. **Never divide latency by batch size and call the result latency.** With static
   batching, every request in a batch finishes when the timed iteration ends, so each
   request's completion latency equals the iteration latency. `latency / batch_size`
   is *amortized service time* — a throughput in disguise.
2. **Batching trades latency for throughput.** Worked example from the synthetic rows 1
   and 2 (same model, device, precision and workload; only `batch_size` differs):

   | run_id suffix | batch_size | latency (mean) | throughput |
   |---|---|---|---|
   | `…-b01-001` | 1 | 852.7 ms | 150.1 tok/s |
   | `…-b08-001` | 8 | 1983.4 ms | 516.3 tok/s |

   Throughput grows ~3.4×, but every individual request finishes ~2.3× slower. Quoting
   only one of the two numbers is how benchmarks mislead.
3. **Never compare throughput across different batch sizes — or throughput against
   latency — without stating the configuration.** A `tok/s` figure from a batch-64 run
   says nothing about what a single interactive user will feel.

Rows with `output_len_tokens = 0` time prefill only; rows with `output_len_tokens > 0`
mix prefill and decode — compare only like with like.

## The validator

Implements **20 error checks** (validation fails) and **17 warning checks**
(data-quality nudge); the tables below list every rule.

The invalid fixture in `examples/` triggers all 14 row-level error checks and
15 of the 17 warning checks (it happens not to trip `W_CUDA_UNEXPECTED` or
`W_RECOMMENDED_EMPTY`). The six error checks it cannot trigger are file- or
header-level (`E_UNREADABLE`, `E_EMPTY_FILE`, `E_CSV_PARSE`,
`E_DUPLICATE_COLUMN`, `E_MISSING_COLUMN`, `E_UNKNOWN_COLUMN`): the validator
stops right after a broken header, so a single fixture can never demonstrate
both the header-level and the row-level rules.

```console
$ python3 validate_ledger.py <ledger.csv> [--strict] [--json]
```

| Exit code | Meaning |
|---|---|
| 0 | valid (warnings allowed unless `--strict`) |
| 1 | validation failed |
| 2 | file could not be read or parsed |

`--strict` treats warnings as failures (recommended in CI); `--json` prints a
machine-readable report (`valid`, `error_count`, `warning_count`, `findings[]`).

### Error checks (validation fails)

| Code | Catches |
|---|---|
| `E_MISSING_COLUMN` / `E_UNKNOWN_COLUMN` / `E_DUPLICATE_COLUMN` | header does not match schema v1.0 exactly |
| `E_FIELD_COUNT` | row has more/fewer fields than the header |
| `E_EMPTY_FILE` / `E_UNREADABLE` / `E_CSV_PARSE` | file-level problems (exit code 2) |
| `E_EMPTY_REQUIRED` | a required column is empty |
| `E_BAD_INT` / `E_BAD_FLOAT` | non-numeric values in numeric columns (also rejects `nan`/`inf`) |
| `E_BAD_ENUM` | value outside the allowed set for `device_type`, `precision`, `quantization_method`, `sync_method` |
| `E_BAD_TIMESTAMP` | not ISO-8601 UTC, or not a real date/time |
| `E_RUN_ID_WHITESPACE` / `E_DUPLICATE_RUN_ID` | malformed or reused `run_id` |
| `E_RANGE` | value outside its allowed range (e.g. `batch_size` < 1, negative lengths, non-positive latency) |
| `E_SCHEMA_VERSION` | `schema_version` is not `1.0` |
| `E_LATENCY_ORDER` / `E_MEAN_OUT_OF_RANGE` | percentile ordering violated, or mean outside [min, max] |
| `E_THROUGHPUT_WITHOUT_OUTPUT` | throughput reported although `output_len_tokens` is 0 |
| `E_MEMORY_EXCEEDS_TOTAL` | `peak_memory_mb` exceeds `device_memory_total_gb` |

### Warning checks (data-quality nudge)

| Code | Catches |
|---|---|
| `W_SYNC_MISMATCH` | gpu timed without device sync; cpu with sync; `cuda_events` on mps |
| `W_NO_WARMUP` | gpu/mps run with 0 warmup iterations |
| `W_FEW_TIMED_RUNS` | fewer than 10 timed runs (tail percentiles unreliable) |
| `W_ZERO_STDDEV` / `W_SPREAD_WITH_SINGLE_RUN` | statistically implausible distributions |
| `W_THROUGHPUT_MISMATCH` | stored throughput differs > 5% from the recomputed value |
| `W_THROUGHPUT_MISSING` | throughput is 0 although tokens were generated |
| `W_NO_PEAK_MEMORY` / `W_MEMORY_TOTAL_MISSING` | memory columns empty on gpu/mps |
| `W_CUDA_MISSING` / `W_CUDA_UNEXPECTED` | `cuda_version` set on non-gpu rows or missing on gpu rows |
| `W_QUANT_METHOD` | `int8`/`int4`/`fp8` precision with `quantization_method = none` |
| `W_RECOMMENDED_EMPTY` | recommended software columns left empty |
| `W_WHITESPACE` / `W_RUN_ID_PATTERN` / `W_EMAIL_FORMAT` / `W_MODEL_ID_FORMAT` | cosmetic/format drift |

## Verification on synthetic data

All rows currently in `inference_timing_ledger.csv` are **synthetic examples** (see
below). The validator was tested against them and against the deliberately broken
fixture:

```console
$ python3 validate_ledger.py inference_timing_ledger.csv --strict
data rows checked: 7
result: VALID - 0 error(s), 0 warning(s)

$ python3 validate_ledger.py examples/ledger_invalid_example.csv
data rows checked: 17
ERROR  row 1 (fixture-bad-enum-01).precision: 'float16' is not an allowed value ...  [E_BAD_ENUM]
ERROR  row 3 (fixture-latency-order-03).latency_ms_p90: latency_ms_p50 (841.5) must be <= latency_ms_p90 (780.0)  [E_LATENCY_ORDER]
ERROR  row 7 (fixture-memory-exceeds-07).peak_memory_mb: peak_memory_mb (40000.0 MB) exceeds device_memory_total_gb (24.0 GB = 24576 MB)  [E_MEMORY_EXCEEDS_TOTAL]
WARN   row 13 (fixture-sync-warnings-13).sync_method: sync_method 'host_timer_only' on a gpu: ...  [W_SYNC_MISMATCH]
... (full list in the run output; 16 errors and 17 warnings in total, covering all 14 row-level error rules and 15 of the 17 warning rules)
result: INVALID - 16 error(s), 17 warning(s)
```

CI (`.github/workflows/ci.yml`) re-runs both checks on every push: the template must
pass `--strict`, and the validator must reject the invalid fixture **with the expected
16 errors and 17 warnings** (asserted via `--json`), so a silently weakened fixture or
validator cannot slip through.

## About the synthetic example rows

Per the agreed task terms, the example rows are synthetic: **no model weights were
downloaded and no accelerator time was used** to produce this template. They exist to
exercise every schema branch (gpu/cpu/mps, quantized and full precision, batch scaling,
CUDA-event timing, prefill-only rows) and to give the validator realistic data to chew
on. Each labels itself as synthetic in its `notes` column. Delete them and add real
measurements — the schema and validator are ready for them.

## Adding a row

1. Open `inference_timing_ledger.csv` in any editor or spreadsheet tool; keep the
   header row untouched.
2. Fill every non-optional column; leave optional columns empty only when they
   genuinely do not apply (e.g. `cuda_version` on a CPU row).
3. Run `python3 validate_ledger.py inference_timing_ledger.csv --strict` before
   committing. CI will re-check it after the push.

## License and contact

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Jyotirmoy Saikia.

Questions about the schema, the validator, or a specific row:
**Jyotirmoy Saikia** — jyotirmoysaikia05153@gmail.com
