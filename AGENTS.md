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
