# README

This is a repo learning metal kernels on m1 air 8gb macOS 27.2.
I want to write sth like the [yalm blog by andrewkchan](https://andrewkchan.dev/posts/yalm.html) 

The plan
1. profile qwen3.5 0.8B metal kernels from oMLX / mlx_lm (other engines: `README_full.md`)
2. PR the faster kernel to the slower engine
3. Bypass apple code with AGXForge(https://github.com/sbryngelson/agxforge)

# 1.1 Install oMLX / mlx_lm

```bash
uv venv --python 3.13
source .venv/bin/activate
uv tool install hf
hf download mlx-community/Qwen3.5-0.8B-bf16

# mlx_lm
uv pip install 'mlx-lm==0.32.2'

# oMLX (GUI install: open the .dmg and drag to Applications)
wget https://github.com/jundot/omlx/releases/download/v0.7.3/oMLX-0.7.3-macos26-27.dmg
```

# 1.2 Test run

Low Power Mode roughly halves Metal decode on this M1 Air (`pmset -g | grep lowpowermode`). Disable it before benchmarking: `sudo pmset -a lowpowermode 0`; restore with `1`.

Take one sample at a time, then run ABBA rounds: `mlx_lm, oMLX, oMLX, mlx_lm`. Compare each adjacent pair as a ratio; the ratio cancels thermal/DVFS drift. Five rounds gives ten pairs.

```bash
# one mlx_lm prefill sample (10k)
mlx_lm.generate --model mlx-community/Qwen3.5-0.8B-bf16 -p "$(cat benchmark_prompt_10k.txt)" -m 1 --temp 0

# one mlx_lm decode sample (1k prefix)
mlx_lm.generate --model mlx-community/Qwen3.5-0.8B-bf16 -p "$(head -c 5465 benchmark_prompt_10k.txt)" -m 16 --temp 0

# one oMLX prefill sample
OMLX=/Applications/oMLX.app/Contents/MacOS/omlx-cli
$OMLX serve --port 8005 --headless &
for i in {1..120}; do curl -s -H 'Authorization: Bearer 1234' -o /dev/null http://127.0.0.1:8005/v1/models && break; sleep 1; done
curl -s -H 'Authorization: Bearer 1234' -H 'Content-Type: application/json' \
  http://127.0.0.1:8005/v1/completions \
  --data-binary "$(jq -n --arg prompt "$(cat benchmark_prompt_10k.txt)" '{model:"mlx-community--Qwen3.5-0.8B-bf16",prompt:$prompt,max_tokens:1,temperature:0}')" \
  | jq -r '.usage.prompt_tokens_per_second'

# one oMLX decode sample (1k prefix)
curl -s -H 'Authorization: Bearer 1234' -H 'Content-Type: application/json' \
  http://127.0.0.1:8005/v1/completions \
  --data-binary "$(jq -n --arg prompt "$(head -c 5465 benchmark_prompt_10k.txt)" '{model:"mlx-community--Qwen3.5-0.8B-bf16",prompt:$prompt,max_tokens:16,temperature:0}')" \
  | jq -r '.usage.generation_tokens_per_second'
```

| Engine | Model             | Prefill tok/s |  wins | Decode tok/s |  wins |
|--------|-------------------|--------------:|------:|-------------:|------:|
| mlx_lm | Qwen3.5-0.8B BF16 |         996.7 | 10/10 |         36.3 |  5/10 |
| oMLX   | Qwen3.5-0.8B BF16 |         952.7 |  0/10 |         35.5 |  5/10 |

Warm up first, both prefill and decode are paired medians; prefill uses the 10k prompt and decode its first 1k tokens; mlx_lm prefill wins 10/10, decode is a tie.

# 1.3 mlx_lm qwen3.5 0.8b prefill

Prefill trace uses `benchmark_prompt_10k.txt` (10000 tokens).

```bash
P=$(cat benchmark_prompt_10k.txt)
mlx_lm.generate --model mlx-community/Qwen3.5-0.8B-bf16 -p "$P" -m 1 --temp 0 > /dev/null 2>&1 &
PID=$!
sleep 3
scripts/profile.sh --attach "$PID" --seconds 30 --kernels --out traces/prefill-baseline
```

| kernel                  | intervals % | what it does                          |
|-------------------------|------------:|---------------------------------------|
| `steel_gemm`            |        76.4 | q/k/v/o and MLP GEMM                  |
| ↳ `nt bm64_bn32_bk32`   |        67.1 | same GEMMs, nt 64×32 tile             |
| ↳ `nn bm64_bn64_bk16`   |         9.3 | same GEMMs, nn 64×64 tile             |
| `gdn`                   |        10.1 | GDN mixer                             |
| `sigmoid+mul`           |         3.5 | fused gating                          |
| rest                    |         3.1 | small kernels                         |
| `select`                |         2.3 | attention mask/index                  |
| copies                  |         2.1 | layout/cast copies                    |
| `depthwise_conv`        |         1.0 | GDN conv                              |
| `gemv`                  |         0.9 | one decode token                      |
| `rms`                   |         0.8 | RMSNorm                               |
| **total**               |       100.0 | rounded                               |

Prefill is compute-bound: `steel_gemm` runs at 1.22 TFLOP/s, 88% of the 1.39 TFLOP/s bf16 GEMM SOL and 30% of the 4.06 TFLOP/s MMA peak; `steel_gemm` is 76.4% of GPU kernel time, and the rest is GDN/attention. Casting the same weights to fp16 raises `steel_gemm` to 1.52 TFLOP/s (88% of the 1.73 fp16 GEMM SOL, 37% of MMA) and the full prefill from 996.7 to ~1175 tok/s (+18%).

| metric                        |    bf16 |  fp16 est. |   MMA SOL |
|-------------------------------|--------:|-----------:|----------:|
| `steel_gemm` TFLOP/s          |    1.22 |       1.52 |      4.06 |
| GEMM SOL (weighted shape mix) |    1.39 |       1.73 |         — |
| % of GEMM SOL                 |     88% |        88% |         — |
| % of MMA SOL                  |     30% |        37% |      100% |
| 10k prefill tok/s             |   996.7 |       1175 |         — |

`fp16 est.` assumes casting the same weights; GEMM SOL is the FLOP-weighted best over the nine model shapes at M=1011 (fastest single shape 1.55, 248320×1024); MMA SOL is the register-resident `simdgroup_matrix<float>` peak at 28k threads, so 4.06 is a compute-only ceiling, and the fp16 gain is cheaper bf16→fp32 conversion.

# 1.4 mlx_lm qwen3.5 0.8b decode

```bash
python benchmarks/decode_trace.py > /tmp/decode.log 2>&1 &
PID=$!
while ! grep -q READY /tmp/decode.log; do sleep 1; done
scripts/profile.sh --attach "$PID" --seconds 20 --kernels --out traces/timeline-baseline
```

Trace session: 32.5 tok/s (30.75 ms/token), 64 decode tokens. The paired engine A/B is a tie (§1.2).
The trace prompt was the first 5,465 bytes of the 10k file.

| kernel             | intervals % | ms/token | what it does                                        |
|--------------------|------------:|---------:|-----------------------------------------------------|
| `gemv`             |        97.5 |    26.41 | BF16 GEMV for decode linear layers                  |
| ↳ `bm8`            |        49.5 |    13.41 | 8 simdgroup rows per threadgroup                    |
| ↳ `bm4`            |        48.0 |    13.00 | 4 simdgroup rows                                    |
| ↳ `bm1`            |         0.0 |     0.00 | 1 simdgroup row (small matrices)                    |
| `sdpa_vector`      |         1.6 |     0.43 | single-token attention over KV cache (head dim 256) |
| rest               |         0.7 |     0.18 | small kernels: argmax, copies, elementwise          |
| `gated_delta_step` |         0.1 |     0.04 | one step of the gated delta rule (linear attention) |
| `rms`              |         0.1 |     0.02 | RMSNorm                                             |
| `sigmoid+mul`      |         0.1 |     0.02 | fused sigmoid gate × value                          |
| host/queue gap     |           — |     3.66 | —                                                   |
| **total**          |       100.0 |    30.75 |                                                     |

| metric   | paired A/B | measured SOL |
|----------|-----------:|-------------:|
| tok/s    |       36.3 |         39.3 |
| ms/token |      27.55 |        25.42 |
| GB/s     |       54.6 |         59.2 |
| % SOL    |        92% |         100% |

Measured SOL = 1.505 GB/token ÷ 59.2 GB/s = 25.42 ms/token → 39.3 tok/s; 1.505 GB is the 0.752B-param text model at 2 bytes, and 59.2 GB/s is the measured streaming read from a 512 MB MLX buffer (87% of the 68.3 GB/s nominal), so the SOL is conservative.

# 1.5 omlx qwen3.5 0.8b prefill

Prefill trace uses `benchmark_prompt_10k.txt` (10000 tokens).

```bash
OMLX=/Applications/oMLX.app/Contents/MacOS/omlx-cli
$OMLX serve --port 8005 --headless &
for i in {1..120}; do curl -s -H 'Authorization: Bearer 1234' -o /dev/null http://127.0.0.1:8005/v1/models && break; sleep 1; done
curl -s -H 'Authorization: Bearer 1234' -H 'Content-Type: application/json' \
  http://127.0.0.1:8005/v1/completions \
  --data-binary "$(jq -n --arg p "$(cat benchmark_prompt_10k.txt)" '{model:"mlx-community--Qwen3.5-0.8B-bf16",prompt:$p,max_tokens:1,temperature:0}')" \
  | jq -r '.usage.prompt_tokens_per_second'
```

| kernel                  | intervals % | what it does                          |
|-------------------------|------------:|---------------------------------------|
| `steel_gemm`            |        76.3 | q/k/v/o and MLP GEMM                  |
| ↳ `nt bm64_bn32_bk32`   |        68.1 | same GEMMs, nt 64×32 tile             |
| ↳ `nn bm64_bn64_bk16`   |         8.2 | same GEMMs, nn 64×64 tile             |
| gdn                     |        10.2 | GDN mixer                             |
| `sigmoid+mul`           |         3.1 | fused gating                          |
| rest                    |         2.6 | small kernels                         |
| copies                  |         2.1 | layout/cast copies                    |
| `select`                |         1.7 | attention mask/index                  |
| `fa256`                 |         1.6 | attention                             |
| `depthwise_conv`        |         1.0 | GDN conv                              |
| `rms`                   |         0.9 | RMSNorm                               |
| `gemv`                  |         0.4 | one decode token                      |
| **total**               |       100.0 | rounded                               |

Warm-to-warm, the 10k interval totals differ by 1.5%; the 12% gap only appears when mlx_lm's cold first prefill is compared with a warm oMLX sample (§1.2).

Prefill is compute-bound: `steel_gemm` runs at 1.24 TFLOP/s, 89% of the 1.39 TFLOP/s bf16 GEMM SOL and 31% of the 4.06 TFLOP/s MMA peak; `steel_gemm` is 76.3% of GPU kernel time, and the rest is GDN/attention. Casting the same weights to fp16 raises `steel_gemm` to 1.55 TFLOP/s (90% of the 1.73 fp16 GEMM SOL, 38% of MMA) and the full prefill from 952.7 to ~1123 tok/s (+18%).

| metric                        |    bf16 |  fp16 est. |   MMA SOL |
|-------------------------------|--------:|-----------:|----------:|
| `steel_gemm` TFLOP/s          |    1.24 |       1.55 |      4.06 |
| GEMM SOL (weighted shape mix) |    1.39 |       1.73 |         — |
| % of GEMM SOL                 |     89% |        90% |         — |
| % of MMA SOL                  |     31% |        38% |      100% |
| 10k prefill tok/s             |   952.7 |       1123 |         — |

`fp16 est.` assumes casting the same weights; GEMM SOL is the FLOP-weighted best over the nine model shapes at M=1011 (fastest single shape 1.55, 248320×1024); MMA SOL is the register-resident `simdgroup_matrix<float>` peak at 28k threads, so 4.06 is a compute-only ceiling, and the fp16 gain is cheaper bf16→fp32 conversion.

# 1.6 omlx qwen3.5 0.8b decode

```bash
OMLX=/Applications/oMLX.app/Contents/MacOS/omlx-cli
$OMLX serve --port 8005 --headless &
PID=$!
curl -s --retry 120 --retry-connrefused --retry-delay 1 \
  -H 'Authorization: Bearer 1234' -H 'Content-Type: application/json' \
  http://127.0.0.1:8005/v1/completions \
  --data-binary "$(jq -n --arg p "$(head -c 5465 benchmark_prompt_10k.txt)" '{model:"mlx-community--Qwen3.5-0.8B-bf16",prompt:$p,max_tokens:512,temperature:0}')" > /dev/null &
sleep 12   # server + model load + prefill
scripts/profile.sh --attach "$PID" --seconds 3 --kernels --out traces/omlx-decode
```

Trace session: 33.0 tok/s (30.3 ms/token). The paired engine A/B is a tie (§1.2).

| kernel                    | intervals % | ms/token | what it does                          |
|---------------------------|------------:|---------:|---------------------------------------|
| `gemv`                    |        99.2 |    30.06 | BF16 GEMV for decode linear layers    |
| ↳ `bm8`                   |        47.8 |    14.48 | 8 simdgroup rows per threadgroup      |
| ↳ `bm4`                   |        48.4 |    14.67 | 4 simdgroup rows                      |
| ↳ `bm1`                   |         3.0 |     0.91 | 1 simdgroup row (small matrices)      |
| `omlx_qwen35_fa256_*` †   |         0.3 |     0.09 | oMLX flash attention (head dim 256)   |
| `vn_copyfloat32float16` † |         0.2 |     0.06 | vectorized fp32→fp16 copy/cast        |
| `rmsbfloat16`             |        0.03 |     0.01 | RMSNorm                               |
| rest                      |         0.3 |     0.09 | small kernels: GDN, RoPE, elementwise |
| **total**                 |       100.0 |    30.30 |                                       |

† Not observed in §1.4.

| metric   | paired A/B | measured SOL |
|----------|-----------:|-------------:|
| tok/s    |       35.5 |         39.3 |
| ms/token |      28.17 |        25.42 |
| GB/s     |       53.4 |         59.2 |
| % SOL    |        90% |         100% |

Measured SOL = 1.505 GB/token ÷ 59.2 GB/s = 25.42 ms/token → 39.3 tok/s; 1.505 GB is the 0.752B-param text model at 2 bytes, and 59.2 GB/s is the measured streaming read from a 512 MB MLX buffer (87% of the 68.3 GB/s nominal), so the SOL is conservative.

# 1.7 analyze mlx_lm

1.2 shows mlx_lm at 36.3 tok/s vs oMLX at 35.5;

GEMV is 97.5% of kernel time; the table below is a nominal sensitivity (68.3 GB/s = 45.4 tok/s), while §1.4/§1.6 use the measured 59.2 GB/s roofline.

| GEMV gain | GB/s | ms/token | tok/s |    e2e |
|----------:|-----:|---------:|------:|-------:|
|        0% | 54.6 |    27.55 |  36.3 |      — |
|        2% | 55.6 |    27.08 |  36.9 |  +1.8% |
|        4% | 56.6 |    26.60 |  37.6 |  +3.6% |
|        6% | 57.6 |    26.13 |  38.3 |  +5.4% |
|      6.9% | 58.1 |    25.92 |  38.6 |  +6.3% |

```bash
python benchmarks/gemv_chain.py --passes 10
```

| tensors | weights |   median | GB/s | note                                             |
|--------:|--------:|---------:|-----:|--------------------------------------------------|
|     187 | 1.50 GB | 25.85 ms | 58.2 | all 9 GEMV shapes; independent, no attention/GDN |

`custom64` ties MLX (0.997, 5/10), and the chain reaches 58.2 GB/s (85% of peak); a 5% kernel win is +4.6% e2e, and the only native lever is forcing bm4 vs bm8, below 2-3% we stop.

The same shapes in fp16 reach 1.73 TFLOP/s weighted (§1.3/§1.5), so 1.39 is a bf16-kernel ceiling, not the M1 limit.

MegaQwen's best e2e result is +1.2% and its MLP block is ~1%, with fusion capped at 2.4%, so megakernels realistically give +1-2% (36.4-36.7 tok/s), not 39.3 — reaching that needs all non-GEMV work and the host gap to vanish.

# 1.8 analyze omlx

oMLX is not slower: 35.5 vs 36.3 tok/s is a 0.977 paired ratio with 5/10 wins, i.e. noise.

The trace difference is GEMV 99.2% vs 97.5%, mostly bm1 at 3.0% vs 0%, plus fa256 0.3% and copy 0.2% against MLX's sdpa 1.5%.

Adding kernels, porting MLX's GEMV, or tuning the current one is not justified: both use the same bm8/bm4 families at 90–92% of measured SOL, and bm1 is only 3% of time; quantize before touching these.

# 1.9 analyze tinygrad

tinygrad BEAM=0 is 16.8 tok/s and BEAM=4 is 25.2, against mlx_lm at 36.3.

| variant         | tok/s | ms/token | GB/s | %SOL |
|-----------------|------:|---------:|-----:|-----:|
| tinygrad BEAM=0 |  16.8 |    59.52 | 25.3 |   43 |
| tinygrad BEAM=4 |  25.2 |    39.68 | 37.9 |   64 |
| mlx_lm          |  36.3 |    27.55 | 54.6 |   92 |
| measured SOL    |  39.3 |    25.42 | 59.2 |  100 |

BEAM=4 is 64% of measured SOL, so it needs +44% to match mlx_lm and +56% to reach 39.3.

BEAM search already tunes its kernels, and latest commits (f5ac546, 45e6a26, bd683c4) fail decode with an LLVM IR error, so there is no newer number and no GEMV port/tune worth doing.
