# README

This is a repo learning metal kernels on m1 air 8gb macOS 27.2.
I want to write sth like the [yalm blog by andrewkchan](https://andrewkchan.dev/posts/yalm.html)

The plan
1. profile qwen3.5 0.8B metal kernels from oMLX / mlx_lm (other engines: `README_full.md`)
2. PR the faster kernel to the slower engine
3. Investigate native GPU code generation beyond Metal (AGXForge targets M5 and is a reference, not an M1 path)

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

Take one sample at a time, then run ABBA rounds: `mlx_lm, oMLX, oMLX, mlx_lm`. Compare the two cross-engine pairs per round (mlx1/omlx1 and omlx2/mlx2) as ratios; the ratio cancels thermal/DVFS drift. Five rounds gives ten pairs.

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

Warm up first, both prefill and decode are paired medians; prefill uses the 10k prompt and decode its first 5,465 bytes (1,012 tokens; oMLX drops the trailing newline and sees 1,011); mlx_lm wins prefill 9/10 and decode 10/10 (oMLX 0.7.4.dev1).

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
| ↳ `nn bm64_bn64_bk16`   |         9.3 | x @ w, 64×64 tile                     |
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

1. † The M=1 row is the LM head, run once per step as a GEMV, excluded from the baseline.
2. `ms` is per dispatch at M=2048; `mix` is the 10k prompt (4×2048 + 1807) and the baseline.

If we can make GEMM faster or remove other work,

| Scenario                       |      GEMM time | GEMM TFLOP/s | Other work |     Prefill speed |
|--------------------------------|---------------:|-------------:|-----------:|------------------:|
| GEMM + other work (measured)   |       0.801 ms |         1.24 |   0.248 ms |         954 tok/s |
| Faster GEMM + other work       |       0.656 ms |         1.52 |   0.248 ms |       1,106 tok/s |
| Modelled hardware + other work |       0.381 ms |         2.62 |   0.248 ms |       1,592 tok/s |
| Modelled hardware only         |       0.381 ms |         2.62 |          0 |       2,627 tok/s |

1. The 0.656 ms is the measured isolated GEMM mix and 0.801 ms is inferred from the measured prefill;
2. the 0.381 ms row is modelled, assuming Apple's 2.62 TFLOP/s fp32 rating (8 cores × 16 execution units × 8 fp32 lanes × 2 (FMA) × 1.278 GHz), not a verified bf16 maximum. `scripts/model_prefill_max.py` applies that peak to the eight Qwen GEMM shapes at M=2048 with tile padding and wave occupancy; padding costs 0.12% of FLOPs, and at 8-32 concurrent tiles every shape fills the machine, so the wave term is zero; only an implausible 256-tile capacity would cost another 1%.

## 1.4 mlx_lm qwen3.5 0.8b decode

```bash
python benchmarks/decode_trace.py > /tmp/decode.log 2>&1 &
PID=$!
while ! grep -q READY /tmp/decode.log; do sleep 1; done
scripts/profile.sh --attach "$PID" --seconds 20 --kernels --out traces/timeline-baseline
```

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

1. Trace session: 32.5 tok/s (30.75 ms/token), 64 decode tokens.
2. The trace prompt was the first 5,465 bytes of the 10k file; the paired A/B is 36.9 vs 35.4 tok/s (§1.2).

| metric   | paired A/B | measured SOL |
|----------|-----------:|-------------:|
| tok/s    |       36.9 |         38.7 |
| ms/token |      27.09 |        25.86 |
| GB/s     |       55.6 |         58.2 |
| % SOL    |        95% |         100% |

1. Measured SOL = 1.505 GB/token ÷ 58.2 GB/s = 25.86 ms/token → 38.7 tok/s; 1.505 GB is the 0.752B-param text model at 2 bytes.
2. 58.2 GB/s is the read rate: the 1.5 GB GEMV chain (`benchmarks/gemv_chain.py`) measures 56.8-58.2 GB/s and a 512 MB read (`scripts/mem_bw.py`) 53.6; 58.2 is the best measured (85% of the 68.3 GB/s nominal).

## 1.5 omlx qwen3.5 0.8b prefill

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
| ↳ `nn bm64_bn64_bk16`   |        12.7 | x @ w, 64×64 tile                     |
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

1. Kernel mix from a 0.7.4.dev1 prefill trace at 10,000 tokens.
2. `steel_gemm` is 76.6% of kernel time, 1.19 TFLOP/s against the 1.517 isolated mix.

If we can make GEMM faster or remove other work,

| Scenario                       |      GEMM time | GEMM TFLOP/s | Other work |     Prefill speed |
|--------------------------------|---------------:|-------------:|-----------:|------------------:|
| GEMM + other work (measured)   |       0.837 ms |         1.19 |   0.259 ms |         912 tok/s |
| Faster GEMM + other work       |       0.656 ms |         1.52 |   0.259 ms |       1,093 tok/s |
| Modelled hardware + other work |       0.381 ms |         2.62 |   0.259 ms |       1,562 tok/s |
| Modelled hardware only         |       0.381 ms |         2.62 |          0 |       2,627 tok/s |

1. The 0.656 ms is the isolated GEMM mix from §1.3; 0.837 ms and 0.259 ms are inferred from the 912 tok/s prefill.
2. The 0.381 ms row is modelled from Apple's 2.62 TFLOP/s fp32 rating, as in §1.3.

### 1.5.1 optimize omlx prefill

| Scenario                       |      GEMM time | GEMM TFLOP/s | Other work |     Prefill speed |
|--------------------------------|---------------:|-------------:|-----------:|------------------:|
| mlx_lm (ref, §1.3)             |       0.801 ms |         1.24 |   0.248 ms |         954 tok/s |
| GEMM + other work (measured)   |       0.837 ms |         1.19 |   0.259 ms |         912 tok/s |
| Faster GEMM + other work       |       0.656 ms |         1.52 |   0.259 ms |       1,093 tok/s |
| Modelled hardware + other work |       0.381 ms |         2.62 |   0.259 ms |       1,562 tok/s |
| Modelled hardware only         |       0.381 ms |         2.62 |          0 |       2,627 tok/s |

We know from 1.2 that mlxlm prefill kernel is faster than omlx.
Lets
1. profile both prefill kernels to understand both
2. understand by omlx prefill kernels are slower, what are the exact time used
3. Draft better omlx prefill kerenls inspired by mlxlm example
4. Test those in omlx run
