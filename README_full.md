# README

This is a repo learning metal kernels on m1 air 8gb macOS 27.2.
I want to write sth like the [yalm blog by andrewkchan](https://andrewkchan.dev/posts/yalm.html)

The plan
1. profile qwen3.5 0.8B metal kernels from ollama / llama.cpp / tinygrad / oMLX / mlx.lm / ZML / vllm-metal / ZMLX / agave (sorted by GitHub stars)
2. PR the faster kernel to the slower engine
3. Bypass apple code with AGXForge(https://github.com/sbryngelson/agxforge)

# 1.1 Install ollama / llama.cpp / tinygrad / oMLX / mlx.lm / ZML / vllm-metal / ZMLX / agave

```bash
# ollama
brew install ollama

uv venv --python 3.13
source .venv/bin/activate
uv pip install huggingface-hub

# llama.cpp
brew install llama.cpp
hf download ggml-org/Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-BF16.gguf --local-dir ~/.cache/gguf

# tinygrad
uv pip install 'tinygrad @ git+https://github.com/tinygrad/tinygrad.git@ce9c84605e94de9667fb578c1ff3fa8678725a87'

# oMLX (GUI install: open the .dmg and drag to Applications)
wget https://github.com/jundot/omlx/releases/download/v0.7.3rc1/oMLX-0.7.3rc1-macos26-27.dmg

# mlx_lm
uv pip install 'mlx-lm==0.32.2'
hf download mlx-community/Qwen3.5-0.8B-bf16 --local-dir ~/.cache/models/Qwen3.5-0.8B-bf16

# ZML / LLMD (Metal)
brew install zml/zml/llmd
hf download Qwen/Qwen3.5-0.8B

# vllm-metal
brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
brew trust vllm-project/vllm-metal
brew install vllm-project/vllm-metal/vllm-metal

# ZMLX
uv pip install 'zmlx[lm]'

# agave (Zig 0.17, https://github.com/maci0/agave)
brew install zig
git clone --depth 1 https://github.com/maci0/agave
cd agave
zig build -Denable-debug=false -Denable-bench=false \
  -Denable-gemma4=false -Denable-diffusion-gemma=false -Denable-qwen4exp=false -Denable-qwen4-exp=false \
  -Denable-deepseek4=false -Denable-llama4=false -Denable-dflash2=false \
  -Denable-vulkan=false -Denable-cuda=false -Denable-rocm=false -Denable-webgpu=false
cd ..

# mps-linear-attn (PyTorch/MPS kernel patch, not an engine)
git clone https://github.com/mpsops/mps-linear-attention
uv pip install torch==2.15.0 transformers setuptools wheel
sed -i '' "s/'-std=c++17', '-O3'/'-std=c++20', '-O3', '-Wno-invalid-specialization'/" mps-linear-attention/setup.py
SDKROOT=$(xcrun --show-sdk-path) CC=/usr/bin/clang CXX=/usr/bin/clang++ \
  uv pip install --no-build-isolation -e mps-linear-attention
```

Torch 2.15.0 lives in `.venv`; the MPS comparison scripts are in `experimental/benchmarks/`.

# 1.2 Test run

Low Power Mode roughly halves Metal decode on this M1 Air (`pmset -g | grep lowpowermode`). Disable it before benchmarking: `sudo pmset -a lowpowermode 0`; restore with `1`.

```bash
# ollama (BF16 default)
MODEL=qwen3.5:0.8b-bf16
ollama serve &
for i in {1..120}; do curl -s -o /dev/null http://127.0.0.1:11434/api/tags && break; sleep 1; done
ollama pull "$MODEL"
for i in {1..5}; do
  curl -s http://127.0.0.1:11434/api/generate \
    -d "$(jq -n --arg model "$MODEL" --arg prompt "$(cat benchmark_prompt_10k.txt)" '{model:$model,prompt:$prompt,stream:false,options:{num_predict:16,temperature:0}}')" \
    | jq -r '.eval_count / (.eval_duration/1e9)'
done

# llama.cpp
for i in {1..5}; do
  /usr/bin/time -l llama-cli -m ~/.cache/gguf/Qwen3.5-0.8B-BF16.gguf -f benchmark_prompt_10k.txt -n 16 --temp 0 -ngl all --perf --no-display-prompt --simple-io
done

# llama.cpp CPU
for i in {1..5}; do
  /usr/bin/time -l llama-cli -m ~/.cache/gguf/Qwen3.5-0.8B-BF16.gguf -f benchmark_prompt_10k.txt -n 16 --temp 0 -ngl 0 --perf --no-display-prompt --simple-io
done

# tinygrad, restart with desired BEAM value and matching cache (beam-1.db, beam-2.db, ...)
BEAM=0 CACHEDB=~/.cache/tinygrad/beam-0.db python -m tinygrad.llm --model ~/.cache/gguf/Qwen3.5-0.8B-BF16.gguf --serve 8002 &
for i in {1..900}; do curl -s -o /dev/null http://127.0.0.1:8002/v1/models && break; sleep 1; done
for i in {1..5}; do
  curl http://127.0.0.1:8002/v1/chat/completions \
    -H 'Content-Type: application/json' \
    --data-binary "$(jq -n --arg prompt "$(cat benchmark_prompt_10k.txt)" '{model:"Qwen3.5-0.8B",messages:[{role:"user",content:$prompt}],max_tokens:16,temperature:0}')"
done

# mlx-lm
for i in {1..5}; do
  mlx_lm.generate --model ~/.cache/models/Qwen3.5-0.8B-bf16 -p "$(cat benchmark_prompt_10k.txt)" -m 16 --temp 0
done

# ZML / LLMD (8GB: fraction 0.6, 2048 context, local checkpoint)
SNAP=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*/ | head -1)
llmd --model="$SNAP" --model-name=Qwen3.5-0.8B \
  --max-context-len=2048 --token-batch-size=512 --prefill-chunk-size=128 \
  --batch-size=1 --gpu-memory-fraction=0.6 --cache-memory-fraction=0.5 \
  --listen=127.0.0.1:8004 &
for i in {1..600}; do curl -s -o /dev/null http://127.0.0.1:8004/v1/models && break; sleep 1; done
for i in {1..5}; do
  curl http://127.0.0.1:8004/v1/chat/completions \
    -H 'Content-Type: application/json' \
    --data-binary "$(jq -n --arg prompt "$(cat benchmark_prompt_10k.txt)" '{model:"Qwen3.5-0.8B",messages:[{role:"user",content:$prompt}],max_tokens:16,temperature:0}')"
done

# vllm-metal
MTL_CAPTURE_ENABLED=1 VLLM_METAL_KV_COMMIT_PROBE=0 vllm serve mlx-community/Qwen3.5-0.8B-bf16 --port 8003 --max-model-len 2048 --gpu-memory-utilization 0.95 &
for i in {1..300}; do curl -s -o /dev/null http://127.0.0.1:8003/v1/models && break; sleep 1; done
for i in {1..5}; do
  curl http://127.0.0.1:8003/v1/completions \
    -H 'Content-Type: application/json' \
    --data-binary "$(jq -n --arg prompt "$(cat benchmark_prompt_10k.txt)" '{model:"mlx-community/Qwen3.5-0.8B-bf16",prompt:$prompt,max_tokens:16,temperature:0}')"
done
# decode tok/s = 1 / (sum / count)
curl -s http://127.0.0.1:8003/metrics | grep request_time_per_output_token_seconds_

# zmlx: runs stock mlx_lm first ("baseline", no patches), then the same model
# with zmlx.patch applied ("patched"); table averages the 5 gen tok/s of each
python -m zmlx.validate ~/.cache/models/Qwen3.5-0.8B-bf16 --max-tokens 16 --runs 5 --prompt "$(cat benchmark_prompt_10k.txt)"

# agave
agave/zig-out/bin/agave ~/.cache/gguf/Qwen3.5-0.8B-BF16.gguf --backend metal --ctx-size 2048 \
  -n 16 -t 0 --seed 42 "$(cat benchmark_prompt_10k.txt)"

# mps-linear-attn (drop the import for baseline; 23.2 -> 27.6 tok/s)
python -c 'import time,torch; import mps_linear_attn; from transformers import Qwen3_5ForConditionalGeneration,AutoTokenizer; p="Qwen/Qwen3.5-0.8B"; t=AutoTokenizer.from_pretrained(p,trust_remote_code=True); m=Qwen3_5ForConditionalGeneration.from_pretrained(p,trust_remote_code=True,dtype=torch.float16).to("mps"); i=t("hi",return_tensors="pt").to("mps"); m.generate(**i,max_new_tokens=4,do_sample=False); torch.mps.synchronize(); s=time.time(); m.generate(**i,max_new_tokens=64,do_sample=False); torch.mps.synchronize(); print(64/(time.time()-s))'
```

| Engine          | Model                  | Decode tok/s (5-run avg)     | Startup/search | Runtime memory | bandwidth used (68.3 GB/s) |
|-----------------|------------------------|------------------------------|----------------|----------------|----------------------------|
| mlx_lm          | Qwen3.5-0.8B BF16      | 33.3 tok/s                   | -              | 2.32 GB        | ~78.0%                     |
| oMLX            | Qwen3.5-0.8B BF16      | 33.0 tok/s                   | -              | 2.49 GB        | ~77.3%                     |
| llama.cpp       | Qwen3.5-0.8B BF16 GGUF | 31.5 tok/s                   | -              | 2.70 GB        | ~73.8%                     |
| Ollama          | Qwen3.5-0.8B BF16 GGUF | 30.6 tok/s                   | -              | 1.77 GB        | ~67.2%                     |
| vllm-metal      | Qwen3.5-0.8B BF16      | 28.0 tok/s                   | ~15s           | ~3.6 GB        | ~65.6%                     |
| ZMLX            | Qwen3.5-0.8B BF16      | 25.4 patched / 34.5 baseline | -              | 2.31 GB        | ~59.5% / ~80.4%            |
| tinygrad BEAM=4 | Qwen3.5-0.8B BF16 GGUF | 25.2 tok/s                   | ~10m 45s       | 2.73 GB        | ~59.0%                     |
| ZML llmd        | Qwen3.5-0.8B BF16      | 24.5 tok/s                   | ~34s           | ~5.2 GB        | ~57.4%                     |
| tinygrad BEAM=3 | Qwen3.5-0.8B BF16 GGUF | 23.2 tok/s                   | ~6m 45s        | 2.67 GB        | ~54.3%                     |
| tinygrad BEAM=0 | Qwen3.5-0.8B BF16 GGUF | 16.8 tok/s                   | -              | 1.94 GB        | ~39.4%                     |
| tinygrad BEAM=2 | Qwen3.5-0.8B BF16 GGUF | 15.0 tok/s                   | 4m 09s         | 2.60 GB        | ~35.1%                     |
| Agave           | Qwen3.5-0.8B BF16 GGUF | 13.7 tok/s                   | -              | 1.79 GB        | ~30.1%                     |
| tinygrad BEAM=5 | Qwen3.5-0.8B BF16 GGUF | 13.2 tok/s                   | ~12m 54s       | 2.79 GB        | ~30.9%                     |
| tinygrad BEAM=1 | Qwen3.5-0.8B BF16 GGUF | 12.8 tok/s                   | 2m 14s         | 2.57 GB        | ~30.0%                     |
| llama.cpp CPU   | Qwen3.5-0.8B BF16 GGUF | 2.2 tok/s                    | -              | ~2.0 GB        | ~4.8%                      |


llama.cpp 0.7.0 b11669: 31.5 tok/s average; `-fa on` 30.5 and `-fa off` 27.9, so the default `auto` stays.

tinygrad latest (f5ac546, 45e6a26, bd683c4) fails decode on M1 with an LLVM IR error after prefill; table numbers are ce9c846.

TensorFold v1.2.0 rejects Qwen3.5-0.8B: `tensorfold info` reports `qwen3_5` full precision, 1.61 GiB, 3.61 GiB floor, but `serve` fails with `UnsupportedQwenGeometry`. Its native Qwen engine supports Qwen3.5-2B, not 0.8B.

Luminal (1cbb917) is unsupported: its `qwen3` adapter rejects Qwen3.5-0.8B (`model_type=qwen3_5_text`), and its Metal backend has no BF16.

# 1.3 Dump the metal kernels

```bash
go install github.com/tmc/gputrace/cmd/gputrace@latest

# ollama (MTL_CAPTURE_ENABLED=1, then gpucapture the llama-server runner; kernel names are UUIDs)
MTL_CAPTURE_ENABLED=1 ollama serve &
for i in {1..120}; do curl -s -o /dev/null http://127.0.0.1:11434/api/tags && break; sleep 1; done
curl -s http://127.0.0.1:11434/api/generate \
  -d "$(jq -n --arg prompt "$(cat benchmark_prompt_10k.txt)" '{model:"qwen3.5:0.8b-bf16",prompt:$prompt,stream:false,options:{num_predict:4,temperature:0}}')" > /dev/null
gpucapture start -p $(gpucapture list | awk '/llama-server/ {print $1; exit}') -c 1 -o /tmp/ollama.gputrace &
GP=$!
sleep 1
curl -s http://127.0.0.1:11434/api/generate \
  -d "$(jq -n --arg prompt "$(cat benchmark_prompt_10k.txt)" '{model:"qwen3.5:0.8b-bf16",prompt:$prompt,stream:false,options:{num_predict:16,temperature:0}}')" > /dev/null
wait $GP
gputrace kernels /tmp/ollama.gputrace
rm -rf /tmp/ollama.gputrace "$TMPDIR"/llamaserver_*.gputrace

# llama.cpp (GGML_METAL_CAPTURE_COMPUTE=2 captures the 2nd graph = first decode step; graph 1 is prefill)
MTL_CAPTURE_ENABLED=1 GGML_METAL_CAPTURE_COMPUTE=2 \
  llama-cli -m ~/.cache/gguf/Qwen3.5-0.8B-BF16.gguf -p hi \
  -n 2 --temp 0 -ngl all --perf --no-display-prompt --simple-io
gputrace kernels /tmp/perf-metal-*.gputrace
rm -rf /tmp/perf-metal-*.gputrace

# tinygrad (first run waits ~2 min for the BEAM=1 search, then prints kernel names)
DEBUG=2 BEAM=1 CACHEDB=~/.cache/tinygrad/beam-1.db python -m tinygrad.llm --model ~/.cache/gguf/Qwen3.5-0.8B-BF16.gguf --benchmark 1 2>&1 | grep '\*\*\* METAL'

# oMLX (gpucapture can't attach; xctrace does. Needs MST-shader.tracetemplate from §1.4.)
# API key: ~/.omlx/settings.json (1234 here)
OMLX=/Applications/oMLX.app/Contents/MacOS/omlx-cli
$OMLX serve --model-dir ~/.cache/models --port 8005 --headless &
SRV=$!
for i in {1..120}; do curl -s -H 'Authorization: Bearer 1234' -o /dev/null http://127.0.0.1:8005/v1/models && break; sleep 1; done
xcrun xctrace record --template MST-shader.tracetemplate --instrument 'Metal GPU Counters' \
  --attach $SRV --time-limit 10s --output /tmp/omlx.trace &
XP=$!
sleep 2
curl -s -H 'Authorization: Bearer 1234' -H 'Content-Type: application/json' \
  http://127.0.0.1:8005/v1/completions \
  --data-binary "$(jq -n --arg prompt "$(cat benchmark_prompt_10k.txt)" '{model:"Qwen3.5-0.8B-bf16",prompt:$prompt,max_tokens:32,temperature:0}')" > /dev/null
wait $XP
xcrun xctrace export --input /tmp/omlx.trace \
  --xpath '//trace-toc/run[@number="1"]/data/table[@schema="metal-shader-profiler-intervals"]' > /tmp/omlx_shader.xml
rm -rf /tmp/omlx.trace

# mlx_lm (one decode step; 1.7 GB, delete after)
MTL_CAPTURE_ENABLED=1 python -c 'import mlx.core as mx; from mlx_lm import load, stream_generate; from mlx_lm.sample_utils import make_sampler; model, tokenizer = load("mlx-community/Qwen3.5-0.8B-bf16"); prompt = open("benchmark_prompt_10k.txt").read(); gen = stream_generate(model, tokenizer, prompt, max_tokens=3, sampler=make_sampler(temp=0)); next(gen); mx.metal.start_capture("/tmp/mlx_decode.gputrace"); next(gen); mx.metal.stop_capture()'
gputrace kernels /tmp/mlx_decode.gputrace
rm -rf /tmp/mlx_decode.gputrace

# ZML llmd (MTL_CAPTURE_ENABLED=1 + gpucapture)
MTL_CAPTURE_ENABLED=1 llmd --model=$HOME/.cache/models/Qwen3.5-0.8B --model-name=Qwen3.5-0.8B \
  --max-context-len=2048 --token-batch-size=512 --prefill-chunk-size=128 --batch-size=1 \
  --gpu-memory-fraction=0.6 --cache-memory-fraction=0.5 --listen=127.0.0.1:8004 &
for i in {1..600}; do curl -s -o /dev/null http://127.0.0.1:8004/v1/models && break; sleep 1; done
curl -s http://127.0.0.1:8004/v1/chat/completions -H 'Content-Type: application/json' \
  --data-binary "$(jq -n --arg prompt "$(cat benchmark_prompt_10k.txt)" '{model:"Qwen3.5-0.8B",messages:[{role:"user",content:$prompt}],max_tokens:4,temperature:0}')" > /dev/null
gpucapture start -p $(gpucapture list | awk '/llmd/ {print $1; exit}') -c 1 -o /tmp/llmd.gputrace &
GP=$!
sleep 1
curl -s http://127.0.0.1:8004/v1/chat/completions -H 'Content-Type: application/json' \
  --data-binary "$(jq -n --arg prompt "$(cat benchmark_prompt_10k.txt)" '{model:"Qwen3.5-0.8B",messages:[{role:"user",content:$prompt}],max_tokens:16,temperature:0}')" > /dev/null
wait $GP
gputrace kernels /tmp/llmd.gputrace
rm -rf /tmp/llmd.gputrace "$TMPDIR"/llmd_*.gputrace

# vllm-metal: MTL_CAPTURE_ENABLED=1 makes it capturable; gpucapture triggers the trace.
# one command buffer ≈ 6 GB (4.2 GB weights); raise -c for more dispatches.
MTL_CAPTURE_ENABLED=1 VLLM_METAL_KV_COMMIT_PROBE=0 vllm serve mlx-community/Qwen3.5-0.8B-bf16 --port 8003 --max-model-len 2048 --gpu-memory-utilization 0.95 &
for i in {1..300}; do curl -s -o /dev/null http://127.0.0.1:8003/v1/models && break; sleep 1; done
# warm up so the worker shows GPU time in `gpucapture list`
curl -s http://127.0.0.1:8003/v1/completions -H 'Content-Type: application/json' \
  --data-binary "$(jq -n --arg prompt "$(cat benchmark_prompt_10k.txt)" '{model:"mlx-community/Qwen3.5-0.8B-bf16",prompt:$prompt,max_tokens:4,temperature:0}')" > /dev/null
gpucapture start -p $(gpucapture list | awk '/Python/ && $4 > 0 {print $1; exit}') -c 1 -o /tmp/vllm.gputrace &
GP=$!
sleep 1
curl -s http://127.0.0.1:8003/v1/completions -H 'Content-Type: application/json' \
  --data-binary "$(jq -n --arg prompt "$(cat benchmark_prompt_10k.txt)" '{model:"mlx-community/Qwen3.5-0.8B-bf16",prompt:$prompt,max_tokens:16,temperature:0}')" > /dev/null
wait $GP
gputrace kernels /tmp/vllm.gputrace
rm -rf /tmp/vllm.gputrace "$TMPDIR"/Python_*.gputrace

# vllm's own profiler (docs/profiling.md): POST /start_profile + /stop_profile with
# MTL_CAPTURE_ENABLED=1 set before the process starts writes a .gputrace. Bound it with
# --gpu-memory-utilization 0.1 and max_tokens=10: ~2.2 GB, ~49 s, capture is 50-100x slower.
# Xcode's Profile pane replays the capture; without the KV shrink it reallocates ~22 GiB
# and locks up. gpucapture above is for kernel names, not per-kernel timing.
# Their doc says xctrace cannot name MLX kernels; that is the stock template. With the
# shader-timeline patch + counters (§1.4) the intervals are named (gemv_*, sdpa_vector).

# sources
ls /opt/homebrew/Cellar/vllm-metal/*/libexec/lib/python3.12/site-packages/vllm_metal/metal/kernels_v2/

# zmlx (same, with the patch applied; 1.8 GB, delete after)
MTL_CAPTURE_ENABLED=1 python -c 'import mlx.core as mx; from mlx_lm import load, stream_generate; from mlx_lm.sample_utils import make_sampler; from zmlx.patch import patch; model, tokenizer = load("mlx-community/Qwen3.5-0.8B-bf16"); patch(model); prompt = open("benchmark_prompt_10k.txt").read(); gen = stream_generate(model, tokenizer, prompt, max_tokens=3, sampler=make_sampler(temp=0)); next(gen); mx.metal.start_capture("/tmp/zmlx_decode.gputrace"); next(gen); mx.metal.stop_capture()'
gputrace kernels /tmp/zmlx_decode.gputrace
rm -rf /tmp/zmlx_decode.gputrace

# agave: no in-process capture hook; --profile is op timing. For a Metal trace use xctrace --launch (§1.4).
# agave (per-op profile; no .gputrace)
agave/zig-out/bin/agave ~/.cache/gguf/Qwen3.5-0.8B-BF16.gguf --backend metal \
  -n 16 -t 0 --profile "hi"
```

| engine     | dump               | capture output               | size   |
|------------|--------------------|------------------------------|--------|
| ollama     | `gpucapture -c 1`  | 20 UUIDs (unlabeled)         | 2.0 GB |
| llama.cpp  | `.gputrace`        | 19 names / 682 dispatches    | 5.0 GB |
| tinygrad   | `DEBUG=2`          | 8 fused names                | —      |
| oMLX       | `xctrace --attach` | 102 shaders / 4566 intervals | 77 MB  |
| mlx_lm     | `.gputrace`        | 44 names / 611 dispatches    | 1.7 GB |
| ZML        | `gpucapture -c 1`  | 195 names / 715 dispatches   | 2.0 GB |
| vllm-metal | `gpucapture -c 1`  | 8 names / 1 dispatch         | 5.9 GB |
| ZMLX       | `.gputrace`        | 43 names / 611 dispatches    | 1.8 GB |
| agave      | `--profile`        | 10 op rows                   | —      |

Parse and delete in the same run; sizes are per capture. Counts are capture-scoped (one graph, one `-c 1` buffer, or 32 tokens), not per-token and not comparable; rank by time % in §1.4.

# 1.4 Profile

The two kernel tables need `xctrace`, not the §1.3 `.gputrace` (no timings); commands are below.

| Engine          | tok/s ↑ | ms/token ↓ | command buffers ↓ | dispatches ↓ | unique kernels |
|-----------------|---------|------------|-------------------|--------------|----------------|
| ZMLX baseline   | 34.5    | 29.0       | 21                | 611          | 44             |
| mlx_lm          | 33.3    | 30.0       | 21                | 611          | 44             |
| oMLX            | 33.0    | 30.3       | 1                 | 48           | 48             |
| vllm-metal      | 28.0    | 35.7       | 1                 | 68           | 68             |
| ZMLX patched    | 25.4    | 39.4       | 25                | 611          | 43             |
| llama.cpp       | 23.9    | 41.8       | 2                 | 671          | 19             |
| tinygrad BEAM=1 | 12.8    | 78.1       | n/a               | 8            | 8              |

`↑` higher is better, `↓` lower is better; `unique kernels` is descriptive.

| kernel                                        | oMLX % | mlx_lm % | llama.cpp % |
|-----------------------------------------------|--------|----------|-------------|
| `gemv_bfloat16_bm8_*`                         | 47.8   | 45.5     | —           |
| `gemv_bfloat16_bm4_*`                         | 48.4   | 45.2     | —           |
| `gemv_bfloat16_bm1_*`                         | 3.0    | 1.9      | —           |
| `rmsbfloat16`                                 | 0.03   | 1.0      | —           |
| `omlx_qwen35_fa256_*`                         | 0.3    | —        | —           |
| `vn_copyfloat32float16`                       | 0.2    | —        | —           |
| oMLX rest                                     | 0.3    | —        | —           |
| `sigmoid+mul` (MLX fused)                     | —      | 2.2      | —           |
| `custom_kernel_gated_delta_step_packed_btree` | —      | 1.6      | —           |
| `sdpa_vector_*`                               | —      | 0.8      | —           |
| mlx_lm rest                                   | —      | 1.8      | —           |
| `kernel_mul_mv_bf16_f32_4`                    | —      | —        | 42.4        |
| `kernel_mul_mv_ext_bf16_f32_r1_2`             | —      | —        | 35.3        |
| `kernel_mul_mv_mma_*`                         | —      | —        | 18.3        |
| `kernel_get_rows_f32`                         | —      | —        | 1.6         |
| `kernel_gated_delta_net_f32_4`                | —      | —        | 0.7         |
| `kernel_ssm_conv_f32_f32_batched_4`           | —      | —        | 0.2         |
| llama.cpp rest                                | —      | —        | 1.5         |

Values are % of GPU time (oMLX/llama: Shader Timeline intervals; mlx_lm: PC samples). Ollama, vllm-metal, ZMLX, ZML, tinygrad: no per-kernel timing yet.

oMLX is -0.3 tok/s (~-1%) vs mlx_lm in the table, despite a smaller non-matmul tail (0.8% vs 7.5%);
its extra kernels are cheap at 32 tokens (`fa256` 0.3%, gdn 0.1%, copy 0.2%).
`fa256` was re-tested at 4k: with `--no-cache` and a 4048-token prompt, oMLX
and mlx_lm both decode at ~29 tok/s, so the short-context tail does not become
a 4k win. The 36.1 tok/s in `experimental/kernels/omlx_benchmark.txt` is an
earlier run that did not reproduce here (32.6 avg over 5 runs, 2026-11-01).

| 4k context (4048 prompt, 32 decode) | prefill tok/s | decode tok/s |
|-------------------------------------|--------------:|-------------:|
| oMLX `--no-cache`                   | 438           | 28.7         |
| mlx_lm                              | 452           | 29.0         |

Medians of 5 alternating runs. Prefill varied 432-477 tok/s under sustained
load (isolated runs reached 629-821), so it is not a stable ranking; decode is
the comparable number.

The stock `Metal System Trace.tracetemplate` ships with the Shader Timeline switch off, so a trace recorded with it yields encoders but an empty `metal-shader-profiler-intervals` table. Patch a copy once (below); `--instrument 'Metal GPU Counters'` is what fills the shader intervals at record time. Attach after the model has loaded, or the window covers weight loading and has no shader intervals.

```bash
python scripts/trace_template.py enable-shader-timeline \
  '/Applications/Xcode.app/Contents/Applications/Instruments.app/Contents/Packages/GPU.instrdst/Contents/Templates/Metal System Trace.tracetemplate' \
  -o MST-shader.tracetemplate

# record a 1-token prefill + 2 decode steps (keeps prefill out of the mix)
xcrun xctrace record --template MST-shader.tracetemplate --instrument 'Metal GPU Counters' \
  --output llama_decode.trace --time-limit 20s --launch -- \
  llama-cli -m ~/.cache/gguf/Qwen3.5-0.8B-BF16.gguf -p hi \
  -n 2 --temp 0 -ngl all --perf --no-display-prompt --simple-io

# per-kernel % GPU time
xcrun xctrace export --input llama_decode.trace \
  --xpath '//trace-toc/run[@number="1"]/data/table[@schema="metal-shader-profiler-intervals"]' > shader.xml

# ALU / Buffer Read / LLC / Compute Occupancy / GPU Read Bandwidth
xcrun xctrace export --input llama_decode.trace \
  --xpath '//trace-toc/run[@number="1"]/data/table[@schema="gpu-counter-value"]' > counters.xml
```

`scripts/profile.sh` wraps the record/export above; `scripts/kernels.py` prints the per-kernel table, `scripts/timeline.py` renders the encoder/shader Gantt as HTML, and `scripts/counters.py` joins counters to shader intervals for an approximate ALU/buffer/LLC/occupancy table (use `--counters --keep`):

```bash
scripts/profile.sh --attach $SRV --seconds 5
scripts/kernels.py traces/<dir>
scripts/timeline.py traces/<dir> --out timeline.html
scripts/profile.sh --attach $SRV --seconds 3 --counters --keep
scripts/counters.py traces/<dir>
```

For a kernel-level A/B, `python experimental/benchmarks/rmsnorm.py` compares `mx.fast.rms_norm` with unfused, `mx.compile`, and custom Metal kernels, then optionally measures full-model tok/s; `python experimental/benchmarks/gemv.py` does the same for a Qwen decode GEMV. `experimental/benchmarks/megaqwen.py` times the fused MegaQwen MLP block, and `--e2e` patches a Qwen layer for a paired tok/s A/B; method and results are in `experimental/benchmarks/README.md` and `experimental/megaqwen/README.md`.

| engine    | matmul % | measured by                                     |
|-----------|----------|-------------------------------------------------|
| oMLX      | 99.2     | Shader Timeline intervals (gemv bm4+bm8+bm1)    |
| llama.cpp | 96.0     | Shader Timeline intervals (mul_mv bf16+ext+mma) |
| mlx_lm    | 92.5     | PC samples (gemv bm8+bm4+bm1)                   |

PC samples = `gpu-shader-profiler-sample` PC hits: sample counts, not durations. oMLX/llama shares are Shader Timeline interval durations.

Stalls in ms ≈ limiter % × kernel duration (Shader Timeline `duration`). For gaps and CPU→GPU latency, export `metal-application-command-buffer-submissions` and `metal-command-buffer-completed`.

# 1.5 Static kernel profile

[metal-profiler](https://github.com/imperatormk/metal-profiler) compiles a standalone `.metal` file, pulls the AGX binary, disassembles it, and estimates instruction cost, register occupancy, dependency stalls, and loop cost. It is static: it does not replace the §1.4 counters or timeline.

```bash
git clone --recurse-submodules https://github.com/imperatormk/metal-profiler ~/.cache/metal-profiler
cd ~/.cache/metal-profiler
PYTHONPATH=. python3 -m metal_profiler.metal_profile kernel.metal -f kernel
```

It needs a standalone `.metal` wrapper; an `mx.fast.metal_kernel` source string is not a file. For the MegaQwen bm4 down kernel it reported 273 instructions, 52 live GPRs, ALU-bound (232 cy F+SCIB vs 18 cy IC), 56% SCIB/control, and 9 loads before a ~165 cy wait. Hoisting the repeated `(row0+t)*K` address math out of the loop made the bm4 down kernel faster than `nn.Linear`; the bitwise A/B is in `experimental/benchmarks/mlp_bm4.py` and the integrated e2e result is in `experimental/megaqwen/README.md`.

Caveats: the occupancy line is a register-ceiling lookup, not achieved occupancy. It scored the one-row-per-threadgroup GEMV at 100% where `xctrace` measured 71.9%, and the bm4 down kernel launches only 2048 threads — 28.6% of the 7-core capacity — so its occupancy ceiling has nothing to do with registers. Its cycle numbers come from Mesa's own "crude cycle model for G13G", which counts ALU issue slots and treats memory, waits and control as zero. Use it to compare two versions of the same kernel, not to predict time. It needs the `applegpu` submodule, and the profiled `.metal` file must match the generated kernel source.

# 1.6 analyze llama.cpp

llama.cpp is 31.5 tok/s, so it is 80% of the 39.3 measured SOL and needs +25% to reach 39.3; default `-fa auto` beat on/off, so profile `mul_mv` next.

# 1.7 analyze ollama

Ollama is 77.9% of SOL and runs the llama.cpp Metal runner, so the same `mul_mv` work applies and it needs +28% to SOL.

# 1.8 analyze vllm-metal

vllm-metal is 65.6% of SOL and needs +53% to SOL; it is MLX-based, so start with dispatch/batching overhead and attention, not a new GEMV.

# 1.9 analyze ZMLX

ZMLX baseline is 80.4% of SOL while the patched build is 59.5%, so keep the baseline and do not apply the patch; it needs +24% to SOL.

# 1.10 analyze ZML

ZML is 57.4% of SOL and needs +74% to SOL; its custom Zig/Metal path needs broad kernel work, not one GEMV.

# 1.11 analyze agave

Agave is 30.1% of SOL and needs +212% to SOL, so fix the basic decode path before any micro-optimization.

# 1.12 analyze tinygrad

tinygrad BEAM=4 is 59.0% of SOL and needs +69% to SOL; BEAM search already tunes its kernels, and latest commits fail decode with the LLVM error, so there is no newer number to chase.

# 1.13 analyze mlx_lm

mlx_lm is 36.3 tok/s, 92% of the 39.3 measured SOL; GEMV is 97.6% of kernel time and already near DRAM, so the remaining work is bm selection or quantization (README §1.5).

# 1.14 analyze oMLX

oMLX is 35.5 tok/s, 90% of the 39.3 measured SOL; it is tied with mlx_lm, so porting or tuning its GEMV is not justified before quantization (README §1.6).

# 1.15 backup

```bash
python experimental/benchmarks/gemv.py --qwen --runs 10
```

| variant    | source                                             | implementation                                                     |
|------------|----------------------------------------------------|--------------------------------------------------------------------|
| `MLX`      | MLX 0.32.4 `gemv_bfloat16_bm8/bm4`                 | plain `x @ w.T`                                                    |
| `custom64` | KERNEL_SOURCE in `experimental/benchmarks/gemv.py` | one threadgroup per output row, 64 threads, shared-memory tree sum |

Sweep, min of 10 queued runs:

| weight N×K  | bm  | calls/token |  MLX GB/s | custom64 GB/s |
|-------------|-----|------------:|----------:|--------------:|
| 248320×1024 | bm8 |           1 |      63.1 |      **65.0** |
| 6144×1024   | bm8 |          18 |      71.4 |      **71.6** |
| 4096×1024   | bm8 |           6 |      78.7 |      **79.4** |
| 3584×1024   | bm4 |          48 |  **94.7** |          89.0 |
| 1024×3584   | bm4 |          24 |  **94.2** |          83.4 |
| 2048×1024   | bm4 |          18 | **116.0** |          96.9 |
| 1024×2048   | bm4 |          24 | **113.9** |          97.5 |
| 512×1024    | bm4 |          12 |  **61.0** |          27.0 |
| 16×1024     | bm1 |          36 |       4.3 |       **6.9** |

248320×1024 is the only shape whose benchmark is actually reading from DRAM; the smaller rows reuse one matrix across runs, so their GB/s is cache-inflated and their 1-2% winners do not translate to a token. The gate is a rotated-buffer chain or full-model A/B.

```bash
python benchmarks/gemv_chain.py --passes 10
```

| tensors | weights |   median | GB/s | note                                |
|--------:|--------:|---------:|-----:|-------------------------------------|
|     187 | 1.50 GB | 25.85 ms | 58.2 | all 9 GEMV shapes; no attention/GDN |

Roofline: 1.505 GB/token; 68.3 GB/s nominal = 45.4 tok/s, 59.2 GB/s measured = 39.3 tok/s; 36.3 tok/s is 92% of measured. GEMV sensitivity: README.md §1.7.

```bash
python experimental/benchmarks/gemv.py --k 1024 --n 248320 --paired --rounds 10 --warmup-seconds 2
```
Paired ABBA, 1024/248320, 2 s warmup — negative control:

| variant  | ratio vs MLX | wins |
|----------|-------------:|-----:|
| custom64 | 0.997        | 5/10 |
