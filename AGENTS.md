# AGENTS.md

Rules for agents working in this repo.

## Scope: change only what was asked

- Make the **smallest** edit that satisfies the request. If the user says "modify 2 lines", modify 2 lines.
- Do not rewrite, restructure, reformat, renumber, or "improve" files you were not asked to touch.
- Do not fix unrelated bugs, typos, or style issues you notice along the way. Mention them instead, or leave them alone.
- Resolve `TODO`/`FIXME` markers in files you edit; if one is out of scope, mention it instead of leaving it stale.
- Do not add sections, tables, examples, notes, or headings that the user did not request.
- Do not rename things (files, symbols, prose terms) unless asked.
- Preserve the existing voice, casing, punctuation, and trailing whitespace. This README is intentionally lowercase and informal — keep it that way.
- Pad Markdown table cells so each column is visually aligned in the raw source, including rows you did not edit.
- After any table edit, re-check every table block in the file (header, separator, all rows) and fix ragged columns before finishing.
- Default to one table per engine. If merging tables, keep each engine's rows in its own contiguous block (diagonal `—` pattern); don't interleave engines or collapse multiple kernels into one row.
- Keep each measurement in one table; reference it from prose instead of duplicating the numbers.
- If a request is ambiguous about scope, ask before expanding it. Never assume "update X" means "rewrite X".
- Prefer simple commands over explanatory prose when the user asks for commands.
- Document kernel dumps as copy-paste commands per engine, like section 1.2; keep the explanation short.
- Set temperature to 0 directly in benchmark commands when the tool exposes that option; do not explain it separately.
- Keep §1.1–§1.3 command blocks in GitHub-star order: ollama, llama.cpp, tinygrad, oMLX, mlx_lm, ZML, vllm-metal, ZMLX, agave. Put a new engine in that order.
- Do not add bare kernel-name lists; give the measured share or the relevant code instead.

## Verify before claiming

- Confirm facts with commands (`sw_vers`, `sysctl`, `--version`) instead of guessing.
- Dump the kernels that actually ran (`llama.cpp GGML_METAL_CAPTURE_COMPUTE`, MLX `mx.metal.start_capture`, `gpucapture`, tinygrad `DEBUG=2`) and `rg` those names in the source; don't guess from static source greps.
- Report only what you actually ran and observed.
- A custom kernel is not faster until an A/B shows it at the context that matters (short decode vs long context).

## Profiling and disk

- Profile from the CLI: patched `Metal System Trace` + `Metal GPU Counters` via `xctrace`, then export the shader and counter tables. `gputrace` only parses `.gputrace` dumps; profiling needs no Xcode.
- A `.gputrace` snapshots model buffers (5.0 GB llama.cpp, 1.7 GB mlx_lm, 1.8 GB ZMLX; vllm 5.9 GB). Parse and `rm -rf` in the same run, including `gpucapture`'s `$TMPDIR/Python_*.gputrace` copy; keep ≥6 GB free; don't `open` those in Xcode on 8 GB — the replay OOMs.
- Shader Timeline + counters write GBs to `$TMPDIR`. Use a time limit, keep ~20 GB free, and delete leftover `instruments*.ktrace` after an interrupted run.
- Exact per-kernel counters still need Xcode Profile, but on a standalone harness with MB-sized buffers.

## Adding an engine

- Build in a temp dir or venv with the documented command; run 5× 16-token decodes at `-t 0`; record the 5-run average decode tok/s, prefill time, peak RSS, and exact versions/commit.
- If the upstream claim does not reproduce, report the measured range and the difference.
- Clean build caches and traces, then check `df -h`.
