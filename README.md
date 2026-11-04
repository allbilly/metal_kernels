# README

This is a repo learning metal kernels on m1 air 8gb macOS 27.2.
I want to write sth like the [yalm blog by andrewkchan](https://andrewkchan.dev/posts/yalm.html) 

The plan
1. profile qwen3.5 0.8B metal kernels from oMLX / mlx_lm (other engines: `README_full.md`)
2. PR the faster kernel to the slower engine
3. Bypass apple code with AGXForge(https://github.com/sbryngelson/agxforge)

## 1.1 Install oMLX / mlx_lm

```bash
uv venv --python 3.13
source .venv/bin/activate
uv tool install hf
hf download mlx-community/Qwen3.5-0.8B-bf16

# mlx_lm
uv pip install 'mlx==0.32.4' 'mlx-lm==0.32.2'

# oMLX (GUI install: open the .dmg and drag to Applications)
wget https://github.com/jundot/omlx/releases/download/v0.7.4.dev1/oMLX-0.7.4.dev1-macos26-27.dmg
```

## 1.2 Test run

Low Power Mode roughly halves Metal decode on this M1 Air (`pmset -g | grep lowpowermode`). Disable it before benchmarking: `sudo pmset -a lowpowermode 0`; restore with `1`.

Take one sample at a time, then run ABBA rounds: `mlx_lm, oMLX, oMLX, mlx_lm`. Compare each adjacent pair as a ratio; the ratio cancels thermal/DVFS drift. Five rounds gives ten pairs.

```bash
# one mlx_lm prefill sample (10k)
mlx_lm.generate --model mlx-community/Qwen3.5-0.8B-bf16 --ignore-chat-template \
  -p "$(cat benchmark_prompt_10k.txt)" -m 1 --temp 0

# one mlx_lm decode sample (1k prefix)
mlx_lm.generate --model mlx-community/Qwen3.5-0.8B-bf16 --ignore-chat-template \
  -p "$(head -c 5465 benchmark_prompt_10k.txt)" -m 16 --temp 0

# one oMLX prefill sample
OMLX=/Applications/oMLX.app/Contents/MacOS/omlx-cli
$OMLX serve --port 8005 --headless &
PID=$!
for i in {1..120}; do curl -sf -H 'Authorization: Bearer 1234' -o /dev/null http://127.0.0.1:8005/v1/models && break; sleep 1; done
curl -s -H 'Authorization: Bearer 1234' -H 'Content-Type: application/json' \
  http://127.0.0.1:8005/v1/completions \
  --data-binary "$(jq -n --arg prompt "$(cat benchmark_prompt_10k.txt)" '{model:"mlx-community--Qwen3.5-0.8B-bf16",prompt:$prompt,max_tokens:1,temperature:0}')" \
  | jq -r '.usage.prompt_tokens_per_second'

# one oMLX decode sample (1k prefix)
curl -s -H 'Authorization: Bearer 1234' -H 'Content-Type: application/json' \
  http://127.0.0.1:8005/v1/completions \
  --data-binary "$(jq -n --arg prompt "$(head -c 5465 benchmark_prompt_10k.txt)" '{model:"mlx-community--Qwen3.5-0.8B-bf16",prompt:$prompt,max_tokens:16,temperature:0}')" \
  | jq -r '.usage.generation_tokens_per_second'
kill $PID
```

| Engine | Model             | Prefill tok/s |  wins | Decode tok/s |  wins |
|--------|-------------------|--------------:|------:|-------------:|------:|
| mlx_lm | Qwen3.5-0.8B BF16 |         954.0 |  9/10 |         36.9 | 10/10 |
| oMLX   | Qwen3.5-0.8B BF16 |         912.0 |  1/10 |         35.4 |  0/10 |

Warm up first, both prefill and decode are paired medians; prefill uses the 10k prompt and decode its first 1k tokens; mlx_lm wins prefill 9/10 and decode 10/10 (oMLX 0.7.4.dev1).

## 1.3 mlx_lm qwen3.5 0.8b prefill

```bash
P=$(cat benchmark_prompt_10k.txt)
mlx_lm.generate --model mlx-community/Qwen3.5-0.8B-bf16 --ignore-chat-template -p "$P" -m 1 --temp 0 > /dev/null 2>&1 &
PID=$!
sleep 3
scripts/profile.sh --attach "$PID" --seconds 30 --kernels --out traces/prefill-baseline
```

| kernel                  | intervals % | what it does                          |
|-------------------------|------------:|---------------------------------------|
| `steel_gemm`            |        76.4 | q/k/v/o and MLP GEMM                  |
| ↳ `nt bm64_bn32_bk32`   |        67.1 | x @ w.T, 64×32: q/k/v/o, MLP up/gate  |
| ↳ `nn bm64_bn64_bk16`   |         9.3 | x @ w, 64×64: MLP down_proj           |
| `gdn`                   |        10.1 | GDN mixer                             |
| `sigmoid+mul`           |         3.5 | fused gating                          |
| rest                    |         3.1 | small kernels                         |
| `select`                |         2.3 | attention mask/index                  |
| copies                  |         2.1 | layout/cast copies                    |
| `depthwise_conv`        |         1.0 | GDN conv                              |
| `gemv`                  |         0.9 | one decode token                      |
| `rms`                   |         0.8 | RMSNorm                               |
| **total**               |       100.0 | rounded                               |

So 76.4% time spent on GEMM, lets see what are the shapes ran.

|    M |    K |      N |   count | bf16 ms | bf16 TFLOP/s |
|-----:|-----:|-------:|--------:|--------:|-------------:|
| 2048 | 1024 |   6144 |      18 |   16.35 |         1.58 |
| 2048 | 1024 |   4096 |       6 |   10.95 |         1.57 |
| 2048 | 1024 |   3584 |      48 |    9.65 |         1.56 |
| 2048 | 3584 |   1024 |      24 |    9.76 |         1.54 |
| 2048 | 1024 |   2048 |      18 |    5.70 |         1.51 |
| 2048 | 2048 |   1024 |      24 |    5.74 |         1.50 |
| 2048 | 1024 |    512 |      12 |    1.73 |         1.24 |
| 2048 | 1024 |     16 |      36 |    0.43 |         0.16 |
|  1 † | 1024 | 248320 |       1 |       — |            — |
|  mix |    — |      — |     186 |   0.656 |        1.517 |
† The M=1 row is the LM head, run once per step as a GEMV, excluded from the SOL.

M=2048 is mlx_lm's default `--prefill-step-size`, so the 10k prefill is 4×2048 + 1807 + 1 tokens. Tuning it does not help: 4096 is 0.990 (3/10) and 10000 is 0.993 (2/6), 1024 was inconclusive under throttle, and 10000 costs 4.7 vs 3.1 GB peak.

We also got the weighted bf16 TFLOP 1.517 as our best measured GEMM, with mix rate Σ count·2·M·N·K ÷ Σ count·ms.

But we only measured 1.24 TFLOP/s during inference: the same FLOPs over the time `steel_gemm` actually takes in the model (76.4% × 1.048 ms/token), i.e. 82% of the isolated 1.517.
The comparison is like-for-like, same shapes, counts, dtype and M, so 1.24/1.517 = 82% is kernel efficiency in situ; 1.517 is the best measured rate, not a hardware bound.

| unit                   | TFLOP/s | best GEMM | % of best GEMM | 10k prefill tok/s |
|------------------------|--------:|----------:|---------------:|------------------:|
| bf16 `steel_gemm`      |    1.24 |     1.517 |            82% |             954.0 |
| bf16 CPU GEMM          |    0.44 |     1.010 |            44% |                 — |

If the model weight is released in fp16, the same shapes run at:

| unit                   | TFLOP/s | best GEMM | % of best GEMM | 10k prefill tok/s |
|------------------------|--------:|----------:|---------------:|------------------:|
| fp16 est. `steel_gemm` |    1.52 |     1.858 |            82% |              1110 |

`fp16 est.` assumes casting the same weights; the CPU row is the same eight shapes on `mx.cpu`, where bf16 converts to fp32 and the 1.01 TFLOP/s fp32 GEMM is its best measured; MMA 4.06 is a register-resident bound, not a GEMM target, and the fp16 gain is cheaper bf16→fp32 conversion.

1.517 is not a hardware SOL: it is `steel_gemm` alone, with no attention, GDN or host gap. The hardware bound is the 4.06 register-resident MMA, which never reads weights, so no real GEMM reaches it; a better kernel than `steel_gemm` could exist, but nothing measured here beats 1.517. Holding the measured 76.4% GEMM share and 0.247 ms/token of non-GEMM work:

| GEMM TFLOP/s | what it is                    | 10k prefill tok/s |
|-------------:|-------------------------------|------------------:|
|         1.24 | `steel_gemm` in the model     |               954 |
|        1.517 | best measured GEMM            |              1106 |
|         4.06 | register-resident MMA         |              2029 |
|            — | no GEMM at all                |              4040 |

tok/s = 1000 ÷ (0.247 + 0.996/rate) ms, so the whole prefill is capped near 2,030 tok/s by the MMA bound and 4,040 by everything that is not GEMM.

## 1.4 mlx_lm qwen3.5 0.8b decode

```bash
python benchmarks/decode_trace.py > /tmp/decode.log 2>&1 &
PID=$!
while ! grep -q READY /tmp/decode.log; do sleep 1; done
scripts/profile.sh --attach "$PID" --seconds 20 --kernels --out traces/timeline-baseline
```

Trace session: 32.5 tok/s (30.75 ms/token), 64 decode tokens. The paired engine A/B is 36.9 vs 35.4 tok/s (§1.2).
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
| tok/s    |       36.9 |         39.3 |
| ms/token |      27.09 |        25.42 |
| GB/s     |       55.6 |         59.2 |
| % SOL    |        94% |         100% |

Measured SOL = 1.505 GB/token ÷ 59.2 GB/s = 25.42 ms/token → 39.3 tok/s; 1.505 GB is the 0.752B-param text model at 2 bytes, and 59.2 GB/s is the measured streaming read from a 512 MB MLX buffer (87% of the 68.3 GB/s nominal), so the SOL is conservative.

The 187 tensors behind that 1.505 GB:

| shape (N×K)  | tensors |     GB |
|--------------|--------:|-------:|
| 248320×1024  |       1 |  0.509 |
| 3584×1024    |      48 |  0.352 |
| 6144×1024    |      18 |  0.227 |
| 1024×3584    |      24 |  0.176 |
| 1024×2048    |      24 |  0.101 |
| 2048×1024    |      18 |  0.076 |
| 4096×1024    |       6 |  0.050 |
| 512×1024     |      12 |  0.013 |
| 16×1024      |      36 |  0.001 |
| **total**    | **187** |  1.505 |

## 1.5 omlx qwen3.5 0.8b prefill

Prefill trace uses `benchmark_prompt_10k.txt` (10,000 tokens).

```bash
OMLX=/Applications/oMLX.app/Contents/MacOS/omlx-cli
$OMLX serve --port 8005 --headless &
PID=$!
for i in {1..120}; do curl -sf -H 'Authorization: Bearer 1234' -o /dev/null http://127.0.0.1:8005/v1/models && break; sleep 1; done
curl -s -H 'Authorization: Bearer 1234' -H 'Content-Type: application/json' \
  http://127.0.0.1:8005/v1/completions \
  --data-binary "$(jq -n --arg p "$(cat benchmark_prompt_10k.txt)" '{model:"mlx-community--Qwen3.5-0.8B-bf16",prompt:$p,max_tokens:1,temperature:0}')" \
  | jq -r '.usage.prompt_tokens_per_second'
kill $PID
```

| kernel                  | intervals % | what it does                          |
|-------------------------|------------:|---------------------------------------|
| `steel_gemm`            |        76.6 | q/k/v/o and MLP GEMM                  |
| ↳ `nt bm64_bn32_bk32`   |        63.9 | x @ w.T, 64×32: q/k/v/o, MLP up/gate  |
| ↳ `nn bm64_bn64_bk16`   |        12.7 | x @ w, 64×64: MLP down_proj           |
| `gdn`                   |         8.4 | GDN mixer                             |
| `looped_softmax`        |         4.0 | attention softmax                     |
| `sigmoid+mul`           |         2.2 | fused gating                          |
| `select`                |         2.1 | attention mask/index                  |
| copies                  |         1.7 | layout/cast copies                    |
| `gemv`                  |         1.3 | one decode token                      |
| `depthwise_conv`        |         0.8 | GDN conv                              |
| `rms`                   |         0.7 | RMSNorm                               |
| `fa256`                 |         0.7 | attention                             |
| rest                    |         1.5 | sdpa, RoPE, elementwise               |
| **total**               |       100.0 | rounded                               |

| unit                   | TFLOP/s | best GEMM | % of best GEMM | 10k prefill tok/s |
|------------------------|--------:|----------:|---------------:|------------------:|
| bf16 `steel_gemm`      |    1.19 |     1.517 |            78% |             912.0 |
| bf16 CPU GEMM          |    0.44 |     1.010 |            44% |                 — |

If the model weight is released in fp16, the same shapes run at:

| unit                   | TFLOP/s | best GEMM | % of best GEMM | 10k prefill tok/s |
|------------------------|--------:|----------:|---------------:|------------------:|
| fp16 est. `steel_gemm` |    1.46 |     1.858 |            78% |              1061 |

`fp16 est.` assumes casting the same weights; the CPU row is the same eight shapes as §1.3 on `mx.cpu`, where bf16 converts to fp32 and the 1.01 TFLOP/s fp32 GEMM is its best measured; MMA 4.06 is a register-resident bound, not a GEMM target, and the fp16 gain is cheaper bf16→fp32 conversion.

## 1.6 omlx qwen3.5 0.8b decode

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
kill $PID
```

Kernel mix from a 0.7.4.dev1 decode trace; ms/token scaled to the paired 35.4 tok/s. The A/B is 35.4 vs 36.9 (§1.2).

| kernel                    | intervals % | ms/token | what it does                          |
|---------------------------|------------:|---------:|---------------------------------------|
| `gemv`                    |        98.0 |    27.70 | BF16 GEMV for decode linear layers    |
| ↳ `bm8`                   |        47.7 |    13.48 | 8 simdgroup rows per threadgroup      |
| ↳ `bm4`                   |        47.2 |    13.34 | 4 simdgroup rows                      |
| ↳ `bm1`                   |         3.1 |     0.88 | 1 simdgroup row (small matrices)      |
| `sdpa_vector`             |         1.6 |     0.45 | single-token attention over KV cache  |
| rest                      |         0.4 |     0.12 | small kernels: rms, copies, argmax    |
| **total**                 |       100.0 |    28.27 |                                       |

| metric   | paired A/B | measured SOL |
|----------|-----------:|-------------:|
| tok/s    |       35.4 |         39.3 |
| ms/token |      28.27 |        25.42 |
| GB/s     |       53.2 |         59.2 |
| % SOL    |        90% |         100% |

Measured SOL = 1.505 GB/token ÷ 59.2 GB/s = 25.42 ms/token → 39.3 tok/s; 1.505 GB is the 0.752B-param text model at 2 bytes, and 59.2 GB/s is the measured streaming read from a 512 MB MLX buffer (87% of the 68.3 GB/s nominal), so the SOL is conservative.

## 1.7 analyze mlx_lm

1.2 shows mlx_lm at 36.9 tok/s vs oMLX at 35.4;

GEMV is 97.5% of kernel time; the table below is a nominal sensitivity (68.3 GB/s = 45.4 tok/s), while §1.4/§1.6 use the measured 59.2 GB/s roofline.

| GEMV gain | GB/s | ms/token | tok/s |    e2e |
|----------:|-----:|---------:|------:|-------:|
|        0% | 55.6 |    27.09 |  36.9 |      — |
|        2% | 56.5 |    26.62 |  37.6 |  +1.7% |
|        4% | 57.6 |    26.15 |  38.2 |  +3.6% |
|        6% | 58.6 |    25.69 |  38.9 |  +5.4% |
|      6.9% | 59.1 |    25.48 |  39.2 |  +6.3% |

```bash
python benchmarks/gemv_chain.py --passes 10
```

| tensors | weights |   median | GB/s | note                                             |
|--------:|--------:|---------:|-----:|--------------------------------------------------|
|     187 | 1.50 GB | 25.85 ms | 58.2 | all 9 GEMV shapes; independent, no attention/GDN |

`custom64` ties MLX (0.997, 5/10), and the chain reaches 58.2 GB/s (85% of peak); a 5% kernel win is +4.6% e2e, and the only native lever is forcing bm4 vs bm8, below 2-3% we stop.

The same shapes in fp16 reach 1.858 TFLOP/s weighted (§1.3/§1.5), so 1.517 is a bf16-kernel ceiling, not the M1 limit.

MegaQwen's best e2e result is +1.2% and its MLP block is ~1%, with fusion capped at 2.4%, so megakernels realistically give +1-2% (37.3-37.6 tok/s), not 39.3 — reaching that needs all non-GEMV work and the host gap to vanish.

## 1.8 analyze omlx

oMLX 0.7.4.dev1 is slower: 35.4 vs 36.9 tok/s is a 0.958 ratio with 0/10 wins.

The trace difference is GEMV 98.0% vs 97.5%, mostly bm1 at 3.1% vs 0%; 0.7.4.dev1 dropped its fa256 decode kernel and now uses sdpa_vector 1.6%, same as MLX's 1.5%.

Adding kernels, porting MLX's GEMV, or tuning the current one is not justified: both use the same bm8/bm4 families at 90–94% of measured SOL, and bm1 is only 3% of time; quantize before touching these.

## 1.9 analyze tinygrad

tinygrad BEAM=0 is 16.8 tok/s and BEAM=4 is 25.2, against mlx_lm at 36.9.

| variant         | tok/s | ms/token | GB/s | %SOL |
|-----------------|------:|---------:|-----:|-----:|
| tinygrad BEAM=0 |  16.8 |    59.52 | 25.3 |   43 |
| tinygrad BEAM=4 |  25.2 |    39.68 | 37.9 |   64 |
| mlx_lm          |  36.9 |    27.09 | 55.6 |   94 |
| measured SOL    |  39.3 |    25.42 | 59.2 |  100 |

BEAM=4 is 64% of measured SOL, so it needs +46% to match mlx_lm and +56% to reach 39.3.

BEAM search already tunes its kernels, and latest commits (f5ac546, 45e6a26, bd683c4) fail decode with an LLVM IR error, so there is no newer number and no GEMV port/tune worth doing.
