# Choosing the SageMaker instance

**Decision (2026-10-05):**
- **Training jobs** (milestones v1–v6 and the live run): **ml.m6i.2xlarge**, run with `--workers 4 --threads 2`.
- **Feature processing job:** **ml.m5.xlarge**.

**Estimated total cost: under $1.** The choice came from cheap local probes and the existing run logs; we did not run the full jobs to compare instances.

## What was measured

All probes ran on the M4 Mac (Python 3.10, LightGBM, same code as commit d96173b). Peak RAM is resident memory (RSS), sampled every 0.5 s with psutil or read from `ru_maxrss`. Each probe took under a minute.

| Probe | What it ran | Peak RAM |
|---|---|---|
| One training worker, worst case | The largest single fit: production / `lag_7_day_diff`, 74 columns, 1,189,880 rows (the last live-run retrain window plus the 400k augmentation rows). Only 20 boosting rounds, since RAM is set by the data, not the number of rounds. | **2.73 GiB** |
| Main process, worst case | Live run: raw tables loaded, then one day's features built | **2.17 GiB** (raw tables 1.62) |
| Feature build, 3 months | `build_features` on 272k rows | 1.80 GiB (raw 1.38) |
| Feature build, 6 months | `build_features` on 552k rows | 2.53 GiB (raw 1.61) |

Two estimates follow from these probes:
- **Feature build, all 2.02 M rows (extrapolated, not measured):** memory above the raw tables grows about linearly, roughly 1.8 GiB per million rows. That gives about 5 GiB in total. We budget **7 GiB**. *Wrong for v1–v5 at the time; see the correction below.*
- **Training RAM with W parallel workers:** about **2.2 + 2.73 · W GiB, plus 1 GiB** for the OS.

**Compute needed** comes from the fit logs of the existing local runs (`fits.csv`, 2 threads per fit). Nothing was re-run.

| | v1 | v2 | v3 | v4 | v5 | v6 | live run |
|---|---|---|---|---|---|---|---|
| Fit-seconds | 151 | 1,301 | 1,527 | 1,950 | 2,022 | 2,005 | ≈ 2,200 |

That is about 11,200 fit-seconds, plus about 9 minutes of work that cannot run in parallel: evaluation, and the daily scoring in the live run.

## Correction (2026-10-06): the full feature build, measured

The first AWS run failed: the processing job (ml.m5.xlarge, 16 GiB) ran out of memory on its first table. The estimate
above came from v6's settings, which cast the features to float16; v1–v5 then kept float64. Measured afterwards, one
full-period table:

| Table | Where | Peak RAM |
|---|---|---|
| v6 (float16) serving table | x86 container (`memory.peak` of its cgroup) | **5.8 GiB** |
| v6 (float16) serving table | Mac (`/usr/bin/time -l`) | 4.5 GiB resident, 7.2 GiB footprint |
| v1 (float64) serving table | Mac (`/usr/bin/time -l`) | 9.6 GiB resident, **15.8 GiB** footprint |

Decision: all milestones now use v6's float16 cast (`config._FP16`; recorded in each model's `notebook_fidelity` tag), so
every table is a float16 table and the processing job stays on ml.m5.xlarge (about 2.7× headroom over 5.8 GiB).

## Estimate per instance

**How the estimates were made:**
- Prices are SageMaker on-demand in us-east-1, from the AWS Pricing API on 2026-10-05.
- **The time estimates assume that one x86 vCPU pair is 1.5× slower than one M4 core.** That factor is a guess, so the times are ±50%.

| Instance | vCPU / RAM | $/h | Workers × threads | RAM needed | Est. time | Est. cost |
|---|---|---|---|---|---|---|
| ml.m5.xlarge | 4 / 16 GiB | 0.230 | 2 × 2 | ~9 GiB | ~2.5 h | ~$0.60 |
| **ml.m6i.2xlarge** | 8 / 32 GiB | 0.461 | 4 × 2 | ~14 GiB | ~1.4 h | **~$0.65** |
| ml.m5.2xlarge | 8 / 32 GiB | 0.461 | 4 × 2 | ~14 GiB | ~1.4 h | ~$0.65 |
| ml.c6i.2xlarge | 8 / 16 GiB | 0.408 | 3 × 2 (4 would be too tight) | ~12 GiB | ~1.8 h | ~$0.72 |
| ml.c5.4xlarge | 16 / 32 GiB | 0.816 | 8 × 2 | ~25 GiB (tight) | ~0.8 h | ~$0.66 |
| ml.m7i.4xlarge | 16 / 64 GiB | 0.968 | 8 × 2 | ~25 GiB | ~0.8 h | ~$0.78 |

**Feature processing job:** ml.m5.xlarge (16 GiB, $0.23/h for processing). That gives about 2× headroom over the 7 GiB budget. Its run time is minutes, so its cost is cents.

## Why ml.m6i.2xlarge

- **Total cost barely depends on instance size.** The work is mostly independent fits, so twice the vCPUs costs twice as much per hour but takes about half the time. The real choice is between RAM headroom and wall time.
- **ml.m6i.2xlarge leaves room to spare.** It uses less than half its RAM (14 of 32 GiB). It finishes in about 1.4 h, against 2.5 h on m5.xlarge, for about $0.05 more.
- **It beats m5.2xlarge on CPU at the same price.** It is a newer Intel generation for the same $0.461/h.
- **The 16-vCPU options are oversized or tight:**
  - c5.4xlarge would need about 25 of its 32 GiB.
  - m7i.4xlarge, the earlier pick, has 64 GiB when about 25 is needed.
- **No Graviton (ARM).** SageMaker training does not offer ARM instances, and the image is built for x86.

**After the first cloud job:** the training job's CloudWatch metrics (MemoryUtilization, CPUUtilization) and its billable seconds will replace the estimates above.
