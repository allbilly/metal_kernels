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
wget https://github.com/jundot/omlx/releases/download/v0.7.4.dev1/oMLX-0.7.4.dev1-macos26-27.dmg

# mlx_lm
uv pip install 'mlx==0.32.4' 'mlx-lm==0.32.2'
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
git clone --depth 1 --branch v0.13.0 https://github.com/maci0/agave ~/Desktop/agave
cd ~/Desktop/agave
zig build -Denable-debug=false -Denable-bench=false \
  -Denable-gemma4=false -Denable-diffusion-gemma=false -Denable-qwen4exp=false -Denable-qwen4-exp=false \
  -Denable-deepseek4=false -Denable-llama4=false -Denable-dflash2=false \
  -Denable-vulkan=false -Denable-cuda=false -Denable-rocm=false -Denable-webgpu=false
cd -

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

# llama.cpp (0.9.0 rejects a prompt longer than -c; 1,012-token prefix at ctx 2048)
for i in {1..5}; do
  /usr/bin/time -l llama-cli -m ~/.cache/gguf/Qwen3.5-0.8B-BF16.gguf \
    -p "$(head -c 5465 benchmark_prompt_10k.txt)" -n 16 --temp 0 -ngl all -c 2048 \
    --perf --no-display-prompt --simple-io
done

# llama.cpp CPU
for i in {1..5}; do
  /usr/bin/time -l llama-cli -m ~/.cache/gguf/Qwen3.5-0.8B-BF16.gguf \
    -p "$(head -c 5465 benchmark_prompt_10k.txt)" -n 16 --temp 0 -ngl 0 -c 2048 \
    --perf --no-display-prompt --simple-io
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
~/Desktop/agave/zig-out/bin/agave ~/.cache/gguf/Qwen3.5-0.8B-BF16.gguf --backend metal --ctx-size 2048 \
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

tinygrad bisect: last working ebe4623e9 (2026-10-09), first bad 741ec2853 hcq runtime dev; 4e45845 (2026-11-04) and f5ac546/45e6a26/bd683c4 still fail decode on M1 with the LLVM IR error, so table numbers are ce9c846. BEAM=0 on ebe4623e9 measures 15 tok/s, the same as ce9c846 in the same session.

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
~/Desktop/agave/zig-out/bin/agave ~/.cache/gguf/Qwen3.5-0.8B-BF16.gguf --backend metal \
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

llama.cpp e30ad80 is 869 tok/s prefill and 35.4 decode on bf16, against mlx_lm's 1008 and 36.7.

| build        | prefill tok/s | 10k time | decode tok/s | ms/token |
|--------------|--------------:|---------:|-------------:|---------:|
| e30ad80 BF16 |         869.4 |  11.50 s |         35.4 |    28.23 |
| mlx_lm BF16  |        1008.0 |   9.92 s |         36.7 |    27.24 |

| kernel            | share |   ms |
|-------------------|------:|-----:|
| `mul_mm`          | 57.2% | 6600 |
| `flash_attn`      | 24.2% | 2790 |
| `gated_delta_net` |  8.2% |  945 |
| `concat`          |  2.2% |  257 |
| `rms_norm`        |  1.9% |  220 |
| rest              |  6.3% |  727 |

1. Cooled paired runs, M1 Air 8 GB, `-ngl 99 -t 4 -fa auto -ub 512 -b 2048`; `-t 0` segfaults, so `-t 4` is used.
2. Profile shares are PC samples, not counter timings; next kernel targets are `mul_mm` and flash-attention tiles, then a chunked GDN scan.

Against mlx_lm (cross-session; llama from PC samples, mlx from intervals):

| family           | llama ms | mlx ms | delta |
|------------------|---------:|-------:|------:|
| GEMM + attention |     9390 |   7471 | +1919 |
| gdn              |      945 |   1021 |   -76 |
| rest             |     1205 |   1056 |  +149 |
| total            |    11540 |   9709 | +1830 |

1. The GEMM+attention bundle is the whole gap; mlx's attention is fused inside `steel_gemm` nt, so the first row cannot be split 1:1.
2. llama's `gdn` is faster than mlx's, and `rest` differs by 0.15 s.

Decompose/next:

| hypothesis                            | test                                 | result                                                          | status                  |
|---------------------------------------|--------------------------------------|-----------------------------------------------------------------|-------------------------|
| L1 bf16 `mul_mm` is the gap           | same-shape MUL_MAT vs MLX            | 1.035 vs 1.544 TF/s on the FLOP-weighted mix (0.67×)            | confirmed               |
| L2 dk256 flash attention is the gap   | `-fa auto` vs `off`, 10 pairs        | prefill 1.000, decode 0.992, auto 3/10                          | eliminated at 1k        |
| L3 the bf16 penalty is shape-specific | f16 control at the same shapes       | f16 1.428 TF/s; 1.72× faster than bf16 at 3584×1024×1024        | confirmed               |
| L4 GEMV time explains decode          | 5 s decode trace                     | 27.11 vs 25.69 ms/fwd (+5.5%), within trace error               | weak                    |
| L5 host/launch gap dominates decode   | untraced wall vs GPU span            | 6.26 vs 2.07 ms/token                                           | supported               |

1. The `ms` column is share × 11.54 s wall, not counter time.
2. The earlier decode shares (57.2/24.2/8.2) were a prefill-contaminated window and are retracted; steady decode is the table below.

Decode profile (steady state, 1k prompt, 5 s window, 33.97 tok/s):

| family        |   share |   ms/token |
|---------------|--------:|-----------:|
| `mul_mm`      |   92.1% |      27.11 |
| `flash_attn`  |    0.6% |       0.18 |
| `gdn`         |    0.0% |       0.00 |
| `rms`         |    0.0% |       0.00 |
| rest          |    7.4% |       2.18 |
| total         |    100% |      29.44 |

bf16 `MUL_MAT` vs MLX `sweep_gemm` (2048 tokens; ggml `m` = features, `n` = tokens, `k` = contraction):

| shape (m×n×k)     |   llama TF/s |   MLX TF/s |   llama/MLX |
|-------------------|-------------:|-----------:|------------:|
| 6144×2048×1024    |        1.427 |       1.58 |        0.90 |
| 3584×2048×1024    |        0.824 |       1.56 |        0.53 |
| 2048×2048×1024    |        0.908 |       1.51 |        0.60 |
| 1024×2048×3584    |        1.067 |       1.54 |        0.69 |
| FLOP-weighted mix |        1.035 |      1.544 |        0.67 |

1. SOL against the 2.62 TFLOP/s fp32 ALU bound: llama 39.5%, MLX 58.9%.
2. f16 control: 1.428 TF/s mix, 7.5% below MLX bf16; at 3584×2048×1024 f16 is 10.584 ms vs bf16 18.244 ms.
3. `-fa auto` vs `off`: prefill 1025.2 vs 1025.7 tok/s (1.000), decode 33.97 vs 33.96 (0.992); the 10k dk256 path is untested by this pair.
4. Decode vs mlx: GEMV 27.11 vs 25.69 ms/fwd (+5.5%); host gap 6.26 vs 2.07 ms/token.

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

tinygrad BEAM=0 is 16.8 tok/s and BEAM=4 is 25.2, against mlx_lm at 36.9.

| variant         | tok/s | ms/token | GB/s | %SOL |
|-----------------|------:|---------:|-----:|-----:|
| tinygrad BEAM=0 |  16.8 |    59.52 | 25.3 |   43 |
| tinygrad BEAM=4 |  25.2 |    39.68 | 37.9 |   65 |
| mlx_lm          |  36.9 |    27.09 | 55.6 |   95 |
| measured SOL    |  38.7 |    25.86 | 58.2 |  100 |

BEAM=4 is 65% of measured SOL, so it needs +46% to match mlx_lm and +54% to reach 38.7.

Decode decomposition (BEAM=1 trace; the fastest BEAM=4 run has no per-kernel profile yet):

| family (shape-inferred)               | share |  ms |
|---------------------------------------|------:|----:|
| QKV `6144x1024` (`r_192_32_1024`)     | 30.7% | 592 |
| LM head `4x62080x1024` (`r_62080...`) | 29.4% | 567 |
| MLP `3584x1024` (`r_3584...`)         | 17.9% | 345 |
| GDN/other reductions                  | 20.2% | 389 |
| embed + rest                          |  1.8% |  34 |

1. The three GEMV families are 78% of decode; at 25.2 tok/s BEAM=4 reads at 37.9 GB/s vs mlx_lm 55.6, so the gap is GEMV efficiency, not launch count (8 dispatches/token).

| hypothesis                                     | evidence                                                    | test                                                              | status |
|------------------------------------------------|-------------------------------------------------------------|-------------------------------------------------------------------|--------|
| T1 generic reduction GEMV loads weights poorly | 78% of decode in three GEMV families; 37.9 vs mlx 55.6 GB/s | same-shape GEMV microbench; inspect generated MSL                 | open   |
| T2 tinygrad moves more than 1.5 GB/token       | RSS 2.73 GB vs 1.5 GB weights                               | count bytes/token or check fp32 staging; recompute effective GB/s | open   |
| T3 BEAM=4 does not tune the GEMV shapes        | profile is BEAM=1; BEAM=4 is 1.5x faster                    | trace BEAM=4, per-kernel shares                                   | open   |

1. Shares and ms are from the BEAM=1 trace at 12.8 tok/s; BEAM=4 may shift them. The family names are inferred from the kernel shape suffixes.

BEAM=4 is 1.5x faster than BEAM=0, so tuning helps; whether it tunes the three GEMV families is open (T3), and every commit since 741ec2853 fails decode with an LLVM IR error, so there is no newer number.

# 1.13 analyze mlx_lm

README.md §1.2 shows mlx_lm at 36.9 tok/s vs oMLX at 35.4; 36.9 is 95% of the 38.7 SOL.

GEMV is 97.5% of kernel time; the table below is a nominal sensitivity (68.3 GB/s = 45.4 tok/s), while README.md §1.4 and §1.16 here use the measured 58.2 GB/s roofline.

| GEMV gain | GB/s | ms/token | tok/s |    e2e |
|----------:|-----:|---------:|------:|-------:|
|        0% | 55.6 |    27.09 |  36.9 |      — |
|        2% | 56.5 |    26.62 |  37.6 |  +1.7% |
|        4% | 57.6 |    26.15 |  38.2 |  +3.6% |
|        6% | 58.6 |    25.69 |  38.9 |  +5.4% |
|      6.9% | 59.1 |    25.48 |  39.2 |  +6.3% |

`custom64` ties MLX (0.997, 5/10) and the chain reaches 58.2 GB/s (85% of peak, §1.16); a 5% kernel win is +4.6% e2e, and the only native lever is forcing bm4 vs bm8, below 2-3% we stop.

The same shapes in fp16 reach 1.858 TFLOP/s weighted (experimental/benchmarks/sweep_gemm.py --dtype fp16), so 1.517 is a bf16-kernel ceiling, not the M1 limit.

MegaQwen's best e2e result is +1.2% and its MLP block is ~1%, with fusion capped at 2.4%, so megakernels realistically give +1-2% (37.3-37.6 tok/s), not 38.7 — reaching that needs all non-GEMV work and the host gap to vanish.

# 1.14 analyze omlx

oMLX 0.7.4.dev1 is slower: 35.4 vs 36.9 tok/s is a 0.958 ratio with 0/10 wins; 35.4 is 91% of the 38.7 SOL.

The trace difference is GEMV 98.0% vs 97.5%, mostly bm1 at 3.1% vs 0%; 0.7.4.dev1 dropped its fa256 decode kernel and now uses sdpa_vector 1.6%, same as MLX's 1.5%.

Adding kernels, porting MLX's GEMV, or tuning the current one is not justified: both use the same bm8/bm4 families at 90–94% of measured SOL, and bm1 is only 3% of time; quantize before touching these.

# 1.15 omlx qwen3.5 0.8b decode

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

| kernel                    | intervals % | ms/token | what it does                          |
|---------------------------|------------:|---------:|---------------------------------------|
| `gemv`                    |        98.0 |    27.70 | BF16 GEMV for decode linear layers    |
| ↳ `bm8`                   |        47.7 |    13.48 | 8 simdgroup rows per threadgroup      |
| ↳ `bm4`                   |        47.2 |    13.34 | 4 simdgroup rows                      |
| ↳ `bm1`                   |         3.1 |     0.88 | 1 simdgroup row (small matrices)      |
| `sdpa_vector`             |         1.6 |     0.45 | single-token attention over KV cache  |
| rest                      |         0.4 |     0.12 | small kernels: rms, copies, argmax    |
| **total**                 |       100.0 |    28.27 |                                       |

1. Kernel mix from a 0.7.4.dev1 decode trace; ms/token scaled to the paired 35.4 tok/s.
2. The A/B is 35.4 vs 36.9 tok/s (README.md §1.2).

| metric   | paired A/B | measured SOL |
|----------|-----------:|-------------:|
| tok/s    |       35.4 |         38.7 |
| ms/token |      28.27 |        25.86 |
| GB/s     |       53.2 |         58.2 |
| % SOL    |        91% |         100% |

1. Measured SOL = 1.505 GB/token ÷ 58.2 GB/s = 25.86 ms/token → 38.7 tok/s; 58.2 GB/s is the best measured streaming read (85% of the 68.3 GB/s nominal).

ABBA (1k prompt, 16 tokens, 10 pairs): mlx 36.71 vs oMLX 34.64 tok/s, ratio **1.0568**, mlx 8/10.

| family      | mlx ms/tok | mlx disp/tok | oMLX ms/tok | oMLX disp/tok | delta ms |
|-------------|------------|--------------|-------------|---------------|----------|
| bm8 GEMV    |      13.37 |         26.3 |       12.89 |          26.2 |    -0.48 |
| bm4 GEMV    |      12.87 |         46.1 |       12.66 |          45.9 |    -0.21 |
| bm1 GEMV    |       0.00 |         0.85 |        0.81 |          13.2 |    +0.81 |
| sdpa_vector |       0.40 |          6.0 |        0.44 |           6.0 |    +0.04 |
| fa256       |       0.00 |            0 |        0.59 |           3.4 |    +0.59 |
| steel_gemm  |       0.00 |         0.11 |        0.00 |          0.00 |     0.00 |
| gdn         |       0.03 |          3.0 |        0.01 |           6.6 |    -0.02 |
| rms         |       0.00 |          2.5 |        0.00 |           5.4 |     0.00 |
| rest        |       0.51 |         15.2 |        0.07 |           5.1 |    -0.44 |
| total       |      27.19 |              |       27.48 |               |    +0.29 |

1. 5 s decode windows: mlx_lm `traces/ot-text` (149 tokens by sdpa count), oMLX default `traces/fresh-vlm` (163); ms/token = family ms ÷ tokens.
2. Retraction: the earlier bm1 VLM-only and fa256 prefill-only rows are wrong. bm1 is 12.9–13.2/token in both oMLX engines and 0.8 in mlx_lm; fa256 runs 3.4/token across the decode window and is 0 when the patch is disabled (`traces/nofa-vlm`).
3. GPU totals differ by +0.29 ms/token here, not the 1.7 e2e, because bm8+bm4 are 0.69 lower in the oMLX capture; the stable within-capture extras are bm1 +0.81 and fa256 +0.59.
4. Cross-session drift is 5–15%; compare extras within one capture, not totals across sessions.
5. Route: `EngineCore._step_burst` → `Scheduler.step` (`omlx/scheduler.py:15736`) → `BatchGenerator.next_generated` → `GenerationBatch._step` (patched at `omlx/scheduler.py:910`) → model. The scheduler only decides batch rows; MLX picks the kernel from the shape, and oMLX swaps attention/GDN only through load-time patches.
6. `_step_burst`'s docstring measures the per-token async hand-off at ~1 ms/token (80 in-process vs 74 through the loop); burst defaults to 64 steps / 0.1 s for a single request, so a short burst pays the full hand-off for a few tokens.
7. A warmed A/B that discards the first completion after load (fresh server per sample) gives VLM 34.41 vs mlx 36.88 tok/s, ratio 0.9323, mlx 10/10 — the gap is steady-state, not a load artifact.
8. A direct mlx_lm vs mlx_vlm language-model check is 1.0006, so the gap is oMLX's serving layer, not the VLM model.
9. Warmed force_lm vs VLM is 34.78 vs 34.78 tok/s (ratio 1.0109, 5/10), so both oMLX engines are identical; the ~6% deficit is common to the serving path, not the engine or the lane. The earlier 1.865x force_lm figure was a first-request JIT artifact and is retracted.

Decode queue results (10 pairs each, default engine, warmed ABBA; ratios are within-experiment):

| test                  | faster lane     |       delta |     wins |
|-----------------------|-----------------|------------:|---------:|
| H0 mlx_lm vs oMLX     | mlx_lm          |        5.6% |    10/10 |
| H1 FA-256 off         | default         |       0.57% |     8/10 |
| G1 gdn-prework off    | default         |        2.2% |     9/10 |
| G2 gdn-off + 0.5 s    | default         |        2.6% |    10/10 |
| H3a burst off vs 64   | 64              |        2.3% |     7/10 |
| H3b 0.5 s vs 64       | 0.5 s           |   **1.34%** |    10/10 |
| H13 single-row bypass | flat            |        0.3% |     5/10 |
| H12 stream vs batch   | stream          |       14.3% |     9/10 |

1. Only H3b is shippable: 64/0.1 s to 256/0.5 s gains 1.34%, 10/10; 0.25 s is flat (1.0014, 6/10) and 1.0 s is 1% worse than 0.5 s (0.9902, 4/10), so 0.5 s is the knee.
2. H1 and G1 are retractions: the FA-256 and gdn-prework patches are net faster even though they add the fa256/bm1 kernels. Removing them loses 0.57% and 2.2%; G2 loses 2.6%. Their kernels are GPU work inside the patch, not exposed headroom.
3. The residual is the batched decode path, not HTTP: in-process `mlx_lm` `batch_generate` is 14.3% slower than `stream_generate` (30.333 vs 34.736 tok/s, stream 9/10). oMLX (35.1) beats stock batch (30.3) but trails stream (36.7/36.9); the ~4.3% left after the burst is that batching penalty.
4. Guards: H7 16k prefill 501 vs 510 tok/s and RSS ~1.9 GB, no OOM; H8 gdn-off outputs are bit-identical on 3 prompts; H9 TTFT +8 ms; H10 streaming gaps ~150 ms in both arms, so 0.5 s does not stall streaming.
5. H5 is inconclusive: the serving path bypassed the Python `_native_kernel` hook (4 calibration calls only); the source gate rules out q_len=1, but no serving q-shape was observed.
6. H13 is flat: 10 pairs give 1.0028 and 1.0040 (5–6/10), so the single-row stream bypass recovers ~0.4%, not 4.3%.
7. T4–T6 close the host and forward terms: burst 27.584 ms/token, output distribution 0.008 ms, idle gap 0.009 ms (p95 0.98); in-process forward B/A = 1.0027; SSE/collector = 1.0026. Per-step intervals for a 16-token request are 43.79, 26.27, 28.29, 27.44, ... (sum 456 ms); mlx_lm is 28.44 first, 28.37 steady. Steady oMLX (27.6 ms) is faster; the gap is the first decode step (+16 ms, 43.79 vs 27.8).
8. T7: at 128 tokens oMLX 34.850 vs mlx_lm 34.742 (ratio 1.0041, oMLX 7/10), first interval 42.07 vs 28.18 ms, steady 28.01 vs 28.62. T8: first step 44.6 ms, `mx.eval` 22.7 vs 0.02 ms; T9b: the first forward is the 1,012th prompt token, forwarded exactly once, so it is not a duplicate. T10/T11 lookahead and first-step prefetch both change the token stream (diverge at id 8 and 7) because the `ArraysCache` GDN state cannot be advanced ahead of the consuming step. The 16 ms is inherent to the current hybrid-cache step structure; the 5% at 16 tokens is a short-window artifact.

| id         | observation                    | evidence                                                                                                                         |
|------------|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| O1         | warmed gap                     | H0: mlx 36.691 vs oMLX 35.140, ratio 1.0564, mlx 10/10                                                                           |
| O2         | engine lane                    | force_lm = VLM, 34.78 vs 34.78, 1.0109                                                                                           |
| O3         | kernel extras                  | bm1 13.2/token, fa256 3.4/token (fresh-vlm)                                                                                      |
| O4         | extras are not exposed         | FA-off −0.57%, gdn-off −2.2%; both patches are net faster                                                                        |
| O5         | burst                          | 0.5 s +1.34%, 10/10; 0.25 s flat; 1.0 s −1%                                                                                      |
| O6         | batched path                   | batch 30.333 vs stream 34.736 tok/s, ratio 1.143, stream 9/10                                                                    |
| O7         | guards                         | 16k prefill 501 vs 510, RSS ~1.9 GB, no OOM; 3/3 identical; TTFT +8 ms; gaps ~150 ms                                             |
| O8         | first decode step              | 43.79 ms vs 27.8 steady; mlx_lm 28.44 first, 28.37 steady                                                                        |
| O9         | prefill drops the last token   | `PromptProcessingBatch.generate` prefills `t[:-1]`; the first `GenerationBatch._step` forwards `t[-1]` (mlx_lm/generate.py:1305) |
| O10        | prefill split differs          | oMLX prefills 1,011 + last token in decode; mlx_lm prefills 1,012, so the 16-token decode-only rate is not like-for-like         |

| id              | hypothesis                                  | test                        | result                               | status                    |
|-----------------|---------------------------------------------|-----------------------------|--------------------------------------|---------------------------|
| H1              | FA-256 costs decode                         | patch off                   | −0.57%                               | false                     |
| H2              | gdn bm1 costs decode                        | patch off                   | −2.2%                                | false                     |
| H3              | burst hand-off                              | 0.5 s vs 64                 | +1.34%                               | true, ships               |
| H4              | MTP verify adds kernels                     | patch off                   | no change                            | retracted                 |
| H5              | FA q_len bypass                             | q-shape hook                | 4 calibration calls only             | inconclusive              |
| H12             | batching machinery costs decode             | stream vs batch             | 14.3%                                | confirmed                 |
| H13             | single-row stream path                      | scheduler bypass            | flat 1.003                           | false                     |
| H15             | inter-step gap                              | T4                          | 0.009 ms/token                       | false                     |
| H16             | forward is slower                           | T5                          | 1.0027                               | false                     |
| H17             | SSE/collector                               | T6                          | 1.0026                               | false                     |
| H18             | first decode step after prefill             | T7/T8/T9b                   | 43.8 vs 27.8; 128 tok 0.4%           | confirmed, inherent       |
| H19             | general lookahead                           | T10                         | tokens differ at id 8                | false                     |
| H20             | first-step prefetch                         | T11                         | tokens differ at id 7                | false                     |
| H21             | full-prompt prefill removes the 16 ms       | H21 prototype               | implemented, id gate pending         | test                      |
| H22             | time-to-16-tokens is fair                   | T14                         | —                                    | open                      |

Verdict: the 5% is the first decode step amortized over 15 intervals, not a sustained-throughput deficit. T12 clean points: 0.782, 0.870, 0.949, 0.977 at 4/8/16/32; L64+ are invalid (host memory pressure). T7 measured 128 tokens at 1.004 (oMLX 7/10), so the crossover is between 32 and 128. H21 (full-prompt prefill) is implemented in `/tmp/omlx-h13`; the id gate is the next test. T13 forked prefetch, T14 fair timing and the combined arm remain. Only the 0.5 s burst is banked (+1.34%).

# 1.16 backup

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

Roofline: 1.505 GB/token; 68.3 GB/s nominal = 45.4 tok/s, 58.2 GB/s measured = 38.7 tok/s; 36.9 tok/s is 95% of measured. GEMV sensitivity: §1.13.

```bash
python experimental/benchmarks/gemv.py --k 1024 --n 248320 --paired --rounds 10 --warmup-seconds 2
```
Paired ABBA, 1024/248320, 2 s warmup — negative control:

| variant  | ratio vs MLX | wins |
|----------|-------------:|-----:|
| custom64 | 0.997        | 5/10 |

# TODO

- [baseRT](https://github.com/basecompute/baseRT) — Apple Silicon LLM runtime; inspect its prefill kernels.
- [metalBLAS](https://github.com/Isalia20/metalBLAS) — Metal GEMM; compare with `steel_gemm`.
- [forge-llm: beating llama.cpp](https://github.com/sauravpanda/forge-llm/blob/main/blog/beating-llama-cpp.md) — Rust AOT compiler for small LLMs.
- [fla-mlx](https://github.com/HallerPatrick/fla-mlx) — flash linear attention for Apple Silicon; relevant to the `gdn` kernel.
- [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) — MLX inference server claiming up to 4× mlx-lm on the same weights; verify its prefill path.
- [magnitude](https://github.com/magnitudedev/magnitude) — engine that tunes kernels per hardware, claiming 2× Ollama; check its Apple Silicon prefill kernels.
- [vllm.cpp](https://github.com/localai-org/vllm.cpp) — C++ vLLM-alike with paged KV and continuous batching; check its Metal backend.
- [ZML/LLMD](https://github.com/zml/zml) — Zig/XLA engine with a Metal backend; 24.5 tok/s decode and 10k prefill ~34 s, so it needs kernel work, not one GEMV.

Scan of the TODO repos (static claims plus follow-up runs in the notes):

| repo        | Metal     | claim                                                     | license                          | verdict                                                         |
|-------------|-----------|-----------------------------------------------------------|----------------------------------|-----------------------------------------------------------------|
| baseRT      | yes       | Qwen3-0.6B Q4 on M3 base: pp512 2,199–2,312 tok/s         | Apache-2.0 repo, engine closed   | ran; no kernel source; MLX `.base` OOMs at 10k                  |
| metalBLAS   | yes       | 4.49 TFLOP/s bf16 4096³ on M1 Max, 1.03× torch            | MIT                              | done; 0.60–0.63× `mx.matmul` bf16 on this M1                    |
| forge-llm   | yes       | Llama-3.2-1B 178 vs MLX 111 tok/s on M5 Pro               | MIT                              | skip; no Qwen3.5                                                |
| fla-mlx     | yes       | 47.4× chunkwise vs naive at 8k                            | none                             | not run; GDN kernels + Qwen3.5 text wrapper, no GEMM API        |
| Rapid-MLX   | yes       | up to 4× decode vs mlx-lm; 115.5 tok/s on Qwen3.5-9B/M5   | Apache-2.0                       | ran; 947 prefill, 33.3 decode; 4× not reproduced                |
| magnitude   | yes       | 2.19× decode / 1.32× prefill vs Ollama                    | Apache-2.0                       | skip; no 0.8B catalog, Bun missing                              |
| vllm.cpp    | partial   | Metal prefill 1.5% ahead MLX-LM                           | Apache-2.0                       | skip; Metal GDN stub (metal_ops.mm:29) → hybrid 0.8B cannot run |

1. Follow-up GEMM run: metalBLAS is 0.90–0.94× `mx.matmul` in fp16 and 0.60–0.63× in bf16 on this M1 Air (7-core GPU), so it is not a candidate; its 4.49 TFLOP/s bf16 claim is an M1 Max (32-core) number.
2. Follow-up e2e: Rapid-MLX measured 947 tok/s prefill and 33.3 decode on the same bf16 weights (vision lane), so its 4x decode claim does not reproduce on 0.8B; baseRT's MLX `.base` OOMs at 10k on 8 GB and its GGUF path warmed at 312 tok/s.
