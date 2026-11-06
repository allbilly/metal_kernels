# AGENTS.md

Rules for agents working in this repo.

## Scope: change only what was asked

- Make the **smallest** edit that satisfies the request. If the user says "modify 2 lines", modify 2 lines.
- A question is not a request to act: answer it, and do not run builds, benchmarks, traces, or tests unless asked. Reading and inspecting files is fine.
- Do not rewrite, restructure, reformat, renumber, or "improve" files you were not asked to touch.
- Do not fix unrelated bugs, typos, or style issues you notice along the way. Mention them instead, or leave them alone.
- Resolve `TODO`/`FIXME` markers in files you edit; if one is out of scope, mention it instead of leaving it stale.
- Do not add sections, tables, examples, notes, or headings that the user did not request.
- Do not invent files, scripts, commands, or measurements; do only what the user asked.
- Do not rename things (files, symbols, prose terms) unless asked.
- Preserve the existing voice, casing, punctuation, and trailing whitespace. This README is intentionally lowercase and informal — keep it that way.
- Pad Markdown table cells so each column is visually aligned in the raw source, including rows you did not edit.
- After any table edit, re-check every table block in the file (header, separator, all rows) and fix ragged columns before finishing.
- Default to one table per engine. If merging tables, keep each engine's rows in its own contiguous block (diagonal `—` pattern); don't interleave engines or collapse multiple kernels into one row.
- Keep each measurement in one table; reference it from prose instead of duplicating the numbers.
- Tables carry their own provenance: label rows `measured` vs `modelled`, and put assumptions in numbered notes (`1.`, `2.`) under the table, not prose paragraphs.
- Show time and rate together (`ms` and `TFLOP/s`), so every speed in the table is checkable arithmetic.
- Mark rows excluded from a total with `†` and define it in the notes.
- Lead a table with one line; no rationale paragraph before it.
- Every constant names its measurement (script or command) and the observed range; use the best measured value, not a rounded-up one, and say which numbers are nominal (e.g. 68.3 GB/s) vs measured (58.2).
- Rebasing a constant means updating every derived number in `README.md` and `README_full.md`: tables, percentages in prose, and unit-table SOL rows.
- After rebasing, `rg` the old value in both files to catch prose and duplicated sections.
- No duplicate section numbers or engine sections across `README.md` and `README_full.md`; moving a section deletes it from the source and closes the numbering gap.
- In paired comparisons, give explicit wins for both sides (e.g. `0/10`), not `—` for the baseline.
- If a request is ambiguous about scope, ask before expanding it. Never assume "update X" means "rewrite X".
- Prefer simple commands over explanatory prose when the user asks for commands.
- Keep optimization sections (e.g. §1.5) to command blocks and result tables; skip the rationale prose. One table per measurement.
- A section titled `optimize X` must contain an actual code/patch change and a measured A/B; otherwise title it `analyze X` or `profile X`.
- Keep new explanations to one sentence; put details in tables or commands.
- Activate the repo venv (`source /Users/yeren/Desktop/metal_kernels/.venv/bin/activate`) and use `python` in README commands; do not write `.venv/bin/python`.
- Use the repo `.venv` for torch/MPS comparisons; do not create a second environment.
- Document kernel dumps as copy-paste commands per engine in `README_full.md` §1.3; keep the explanation short.
- Set temperature to 0 directly in benchmark commands when the tool exposes that option; do not explain it separately.
- Keep README install/test commands in order: oMLX, mlx_lm; other engines are in `README_full.md`.
- Do not add bare kernel-name lists; give the measured share or the relevant code instead.
- Never drop a measured `ms`/time column or total when editing a table; keep the time next to its rate.

## Debugging

- Debug in three layers: observations (measured, each with its source), hypotheses (each tied to a named observation), and the test that proves or disproves it. Never present a hypothesis as a finding.
- Every hypothesis names its deciding test, the expected result on each side, and the cost; a hypothesis with no deciding test is a guess and stays out of the docs.
- When a test disproves or fails to support a hypothesis, retract the claim in the doc and say what killed it; do not leave it as `likely` or delete the evidence silently.
- When a kernel has no obvious source line, trace the Python caller in the real engine process — wrap `nn.Linear.__call__` with a stack recorder or attach a debugger — and group by `(x.shape, weight.shape, caller)`; a kernel name alone does not prove the route, and identical per-token op inventories mean the difference is kernel selection or measurement.
- Set-diff kernel names from paired traces (`scripts/kernels.py traces/A --csv --all | cut -d, -f1 | sort -u`, then `comm -23 omlx mlx`) and map every oMLX-only name to its JIT/source and route gate before counting it as decode cost; prefill kernels (`fa256`, `gdn_prework`) leak into short windows and inflate per-token rates.

## Verify before claiming

- Confirm facts with commands (`sw_vers`, `sysctl`, `--version`) instead of guessing.
- Dump the kernels that actually ran (`llama.cpp GGML_METAL_CAPTURE_COMPUTE`, MLX `mx.metal.start_capture`, `gpucapture`, tinygrad `DEBUG=2`) and `rg` those names in the source; don't guess from static source greps.
- Report only what you actually ran and observed.
- Don't state hardware throughput numbers (FP16/bf16 peak, TFLOP/s, 2x ratios) without a source or a measurement on this machine; cite it or say it is unmeasured.
- Don't write `likely`/`probably` for a measured difference; show the cause (matched warmup, kernel trace, or patch ablation) or say it is unexplained.
- Warm every engine once before a paired prefill sample; a cold first prefill includes kernel compilation and understates tok/s.
- A custom kernel is not faster until an A/B shows it at the context that matters (short decode vs long context).
- Report GEMM SOL against the FLOP-weighted shape mix, not the fastest single shape; list the fastest shape and calculated ALU bound separately.
- Keep bf16/fp16 and MLX/torch separate; M1 has no BF16 matrix units, so Qwen inference is bf16 storage with fp32 math, fp16 is comparison-only (`sweep_gemv.py --dtype`, `torch_mps.py`), and a dtype or backend switch needs an e2e correctness/stability A/B.

## Profiling and disk

- Before a GPU run, check `/tmp/metal_kernels_gpu.lock` with `lsof`; if another process holds it, wait. Don't add a lock helper or build locking into the benchmark script.
- Profile hotspots from the CLI: patched `Metal System Trace` + `Metal GPU Counters` via `xctrace`, then export the shader and counter tables; hotspot profiling needs no Xcode.
- A `.gputrace` snapshots model buffers (5.0 GB llama.cpp, 1.7 GB mlx_lm, 1.8 GB ZMLX; vllm 5.9 GB). Parse and `rm -rf` in the same run, including `gpucapture`'s `$TMPDIR/Python_*.gputrace` copy; keep ≥6 GB free; don't `open` those in Xcode on 8 GB — the replay OOMs.
- Shader Timeline + counters write GBs to `$TMPDIR`. Use a time limit, keep ~20 GB free, and delete leftover `instruments*.ktrace` after an interrupted run.
- Exact per-kernel counters still need Xcode Profile, but on a standalone harness with MB-sized buffers.

## Adding an engine

- Build in a temp dir or venv with the documented command; run paired ABBA 16-token decodes at `-t 0` (5 rounds = 10 pairs) and measure prefill too; record median prefill/decode tok/s, ratio/wins, peak RSS, and exact versions/commit.
- If the upstream claim does not reproduce, report the measured range and the difference.
- Clean build caches and traces, then check `df -h`.
