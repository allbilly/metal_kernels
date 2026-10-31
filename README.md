# README

This is a repo learning metal kernels on m1 air 8gb macOS 27.2.
I want to write sth like the [yalm blog by andrewkchan](https://andrewkchan.dev/posts/yalm.html) 

The plan
1. profile qwen3.6 0.8B metal kernels from ollama / llama.cpp / tinygrad / oMLX / mlx.lm / ZML / vllm-metal / ZMLX / agave (sorted by GitHub stars)
2. PR the faster kernel to the slower engine
2. Bypass apple code with AGXForge(https://github.com/sbryngelson/agxforge) 

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
uv venv --python 3.12 .venv-mla
uv pip install --python .venv-mla/bin/python torch==2.15.0 transformers setuptools wheel
sed -i '' "s/'-std=c++17', '-O3'/'-std=c++20', '-O3', '-Wno-invalid-specialization'/" mps-linear-attention/setup.py
SDKROOT=$(xcrun --show-sdk-path) CC=/usr/bin/clang CXX=/usr/bin/clang++ \
  uv pip install --python .venv-mla/bin/python --no-build-isolation -e mps-linear-attention
```

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
    -d "$(jq -n --arg model "$MODEL" --arg prompt "$(cat benchmark_prompt.txt)" '{model:$model,prompt:$prompt,stream:false,options:{num_predict:16,temperature:0}}')" \
    | jq -r '.eval_count / (.eval_duration/1e9)'
done

# llama.cpp
for i in {1..5}; do
  /usr/bin/time -l llama-cli -m ~/.cache/gguf/Qwen3.5-0.8B-BF16.gguf -f benchmark_prompt.txt -n 16 --temp 0 -ngl all --perf --no-display-prompt --simple-io
done

# llama.cpp CPU
for i in {1..5}; do
  /usr/bin/time -l llama-cli -m ~/.cache/gguf/Qwen3.5-0.8B-BF16.gguf -f benchmark_prompt.txt -n 16 --temp 0 -ngl 0 --perf --no-display-prompt --simple-io
done

# tinygrad, restart with desired BEAM value and matching cache (beam-1.db, beam-2.db, ...)
BEAM=0 CACHEDB=~/.cache/tinygrad/beam-0.db python -m tinygrad.llm --model ~/.cache/gguf/Qwen3.5-0.8B-BF16.gguf --serve 8002 &
for i in {1..900}; do curl -s -o /dev/null http://127.0.0.1:8002/v1/models && break; sleep 1; done
for i in {1..5}; do
  curl http://127.0.0.1:8002/v1/chat/completions \
    -H 'Content-Type: application/json' \
    --data-binary "$(jq -n --arg prompt "$(cat benchmark_prompt.txt)" '{model:"Qwen3.5-0.8B",messages:[{role:"user",content:$prompt}],max_tokens:16,temperature:0}')"
done

# mlx-lm
for i in {1..5}; do
  mlx_lm.generate --model ~/.cache/models/Qwen3.5-0.8B-bf16 -p "$(cat benchmark_prompt.txt)" -m 16 --temp 0
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
    --data-binary "$(jq -n --arg prompt "$(cat benchmark_prompt.txt)" '{model:"Qwen3.5-0.8B",messages:[{role:"user",content:$prompt}],max_tokens:16,temperature:0}')"
done

# vllm-metal
MTL_CAPTURE_ENABLED=1 VLLM_METAL_KV_COMMIT_PROBE=0 vllm serve mlx-community/Qwen3.5-0.8B-bf16 --port 8003 --max-model-len 2048 --gpu-memory-utilization 0.95 &
for i in {1..300}; do curl -s -o /dev/null http://127.0.0.1:8003/v1/models && break; sleep 1; done
for i in {1..5}; do
  curl http://127.0.0.1:8003/v1/completions \
    -H 'Content-Type: application/json' \
    --data-binary "$(jq -n --arg prompt "$(cat benchmark_prompt.txt)" '{model:"mlx-community/Qwen3.5-0.8B-bf16",prompt:$prompt,max_tokens:16,temperature:0}')"
done
# decode tok/s = 1 / (sum / count)
curl -s http://127.0.0.1:8003/metrics | grep request_time_per_output_token_seconds_

# zmlx: runs stock mlx_lm first ("baseline", no patches), then the same model
# with zmlx.patch applied ("patched"); table averages the 5 gen tok/s of each
python -m zmlx.validate ~/.cache/models/Qwen3.5-0.8B-bf16 --max-tokens 16 --runs 5 --prompt "$(cat benchmark_prompt.txt)"

# agave
agave/zig-out/bin/agave ~/.cache/gguf/Qwen3.5-0.8B-BF16.gguf --backend metal --ctx-size 2048 \
  -n 16 -t 0 --seed 42 "$(cat benchmark_prompt.txt)"

# mps-linear-attn (drop the import for baseline; 23.2 -> 27.6 tok/s)
uv run --no-project --python .venv-mla/bin/python python -c 'import time,torch; import mps_linear_attn; from transformers import Qwen3_5ForConditionalGeneration,AutoTokenizer; p="Qwen/Qwen3.5-0.8B"; t=AutoTokenizer.from_pretrained(p,trust_remote_code=True); m=Qwen3_5ForConditionalGeneration.from_pretrained(p,trust_remote_code=True,dtype=torch.float16).to("mps"); i=t("hi",return_tensors="pt").to("mps"); m.generate(**i,max_new_tokens=4,do_sample=False); torch.mps.synchronize(); s=time.time(); m.generate(**i,max_new_tokens=64,do_sample=False); torch.mps.synchronize(); print(64/(time.time()-s))'
```

| Engine          | Model                  | Decode tok/s (5-run avg)     | Startup/search | Runtime memory | bandwidth used (68.3 GB/s) |
|-----------------|------------------------|------------------------------|----------------|----------------|----------------------------|
| mlx_lm          | Qwen3.5-0.8B BF16      | 33.3 tok/s                   | -              | 2.32 GB        | ~78.0%                     |
| oMLX            | Qwen3.5-0.8B BF16      | 33.0 tok/s                   | -              | 2.49 GB        | ~77.3%                     |
| Ollama          | Qwen3.5-0.8B BF16 GGUF | 30.6 tok/s                   | -              | 1.77 GB        | ~67.2%                     |
| vllm-metal      | Qwen3.5-0.8B BF16      | 28.0 tok/s                   | ~15s           | ~3.6 GB        | ~65.6%                     |
| ZMLX            | Qwen3.5-0.8B BF16      | 25.4 patched / 34.5 baseline | -              | 2.31 GB        | ~59.5% / ~80.4%            |
| tinygrad BEAM=4 | Qwen3.5-0.8B BF16 GGUF | 25.2 tok/s                   | ~10m 45s       | 2.73 GB        | ~59.0%                     |
| ZML llmd        | Qwen3.5-0.8B BF16      | 24.5 tok/s                   | ~34s           | ~5.2 GB        | ~57.4%                     |
| llama.cpp       | Qwen3.5-0.8B BF16 GGUF | 23.9 tok/s                   | -              | 2.70 GB        | ~55.9%                     |
| tinygrad BEAM=3 | Qwen3.5-0.8B BF16 GGUF | 23.2 tok/s                   | ~6m 45s        | 2.67 GB        | ~54.3%                     |
| tinygrad BEAM=0 | Qwen3.5-0.8B BF16 GGUF | 16.0 tok/s                   | -              | 1.94 GB        | ~37.5%                     |
| tinygrad BEAM=2 | Qwen3.5-0.8B BF16 GGUF | 15.0 tok/s                   | 4m 09s         | 2.60 GB        | ~35.1%                     |
| Agave           | Qwen3.5-0.8B BF16 GGUF | 13.7 tok/s                   | -              | 1.79 GB        | ~30.1%                     |
| tinygrad BEAM=5 | Qwen3.5-0.8B BF16 GGUF | 13.2 tok/s                   | ~12m 54s       | 2.79 GB        | ~30.9%                     |
| tinygrad BEAM=1 | Qwen3.5-0.8B BF16 GGUF | 12.8 tok/s                   | 2m 14s         | 2.57 GB        | ~30.0%                     |
| llama.cpp CPU   | Qwen3.5-0.8B BF16 GGUF | 2.2 tok/s                    | -              | ~2.0 GB        | ~4.8%                      |


LPM off; 5-run averages. All rows re-run except the tinygrad BEAM sweep (BEAM=1 re-run at 12.8).

TensorFold v1.2.0 rejects Qwen3.5-0.8B: `tensorfold info` reports `qwen3_5` full precision, 1.61 GiB, 3.61 GiB floor, but `serve` fails with `UnsupportedQwenGeometry`. Its native Qwen engine supports Qwen3.5-2B, not 0.8B.

# 1.3 Dump the metal kernels

```bash
go install github.com/tmc/gputrace/cmd/gputrace@latest

# ollama (MTL_CAPTURE_ENABLED=1, then gpucapture the llama-server runner; kernel names are UUIDs)
MTL_CAPTURE_ENABLED=1 ollama serve &
for i in {1..120}; do curl -s -o /dev/null http://127.0.0.1:11434/api/tags && break; sleep 1; done
curl -s http://127.0.0.1:11434/api/generate \
  -d "$(jq -n --arg prompt "$(cat benchmark_prompt.txt)" '{model:"qwen3.5:0.8b-bf16",prompt:$prompt,stream:false,options:{num_predict:4,temperature:0}}')" > /dev/null
gpucapture start -p $(gpucapture list | awk '/llama-server/ {print $1; exit}') -c 1 -o /tmp/ollama.gputrace &
GP=$!
sleep 1
curl -s http://127.0.0.1:11434/api/generate \
  -d "$(jq -n --arg prompt "$(cat benchmark_prompt.txt)" '{model:"qwen3.5:0.8b-bf16",prompt:$prompt,stream:false,options:{num_predict:16,temperature:0}}')" > /dev/null
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

# oMLX (gpucapture can't attach; xctrace does. Needs MST-shader.tracetemplate from §1.5.)
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
  --data-binary "$(jq -n --arg prompt "$(cat benchmark_prompt.txt)" '{model:"Qwen3.5-0.8B-bf16",prompt:$prompt,max_tokens:32,temperature:0}')" > /dev/null
wait $XP
xcrun xctrace export --input /tmp/omlx.trace \
  --xpath '//trace-toc/run[@number="1"]/data/table[@schema="metal-shader-profiler-intervals"]' > /tmp/omlx_shader.xml
rm -rf /tmp/omlx.trace

# mlx_lm (one decode step; 1.7 GB, delete after)
MTL_CAPTURE_ENABLED=1 python -c 'import mlx.core as mx; from mlx_lm import load, stream_generate; from mlx_lm.sample_utils import make_sampler; model, tokenizer = load("mlx-community/Qwen3.5-0.8B-bf16"); prompt = open("benchmark_prompt.txt").read(); gen = stream_generate(model, tokenizer, prompt, max_tokens=3, sampler=make_sampler(temp=0)); next(gen); mx.metal.start_capture("/tmp/mlx_decode.gputrace"); next(gen); mx.metal.stop_capture()'
gputrace kernels /tmp/mlx_decode.gputrace
rm -rf /tmp/mlx_decode.gputrace

# ZML llmd (MTL_CAPTURE_ENABLED=1 + gpucapture)
MTL_CAPTURE_ENABLED=1 llmd --model=$HOME/.cache/models/Qwen3.5-0.8B --model-name=Qwen3.5-0.8B \
  --max-context-len=2048 --token-batch-size=512 --prefill-chunk-size=128 --batch-size=1 \
  --gpu-memory-fraction=0.6 --cache-memory-fraction=0.5 --listen=127.0.0.1:8004 &
for i in {1..600}; do curl -s -o /dev/null http://127.0.0.1:8004/v1/models && break; sleep 1; done
curl -s http://127.0.0.1:8004/v1/chat/completions -H 'Content-Type: application/json' \
  --data-binary "$(jq -n --arg prompt "$(cat benchmark_prompt.txt)" '{model:"Qwen3.5-0.8B",messages:[{role:"user",content:$prompt}],max_tokens:4,temperature:0}')" > /dev/null
gpucapture start -p $(gpucapture list | awk '/llmd/ {print $1; exit}') -c 1 -o /tmp/llmd.gputrace &
GP=$!
sleep 1
curl -s http://127.0.0.1:8004/v1/chat/completions -H 'Content-Type: application/json' \
  --data-binary "$(jq -n --arg prompt "$(cat benchmark_prompt.txt)" '{model:"Qwen3.5-0.8B",messages:[{role:"user",content:$prompt}],max_tokens:16,temperature:0}')" > /dev/null
wait $GP
gputrace kernels /tmp/llmd.gputrace
rm -rf /tmp/llmd.gputrace "$TMPDIR"/llmd_*.gputrace

# vllm-metal: MTL_CAPTURE_ENABLED=1 makes it capturable; gpucapture triggers the trace.
# one command buffer ≈ 6 GB (4.2 GB weights); raise -c for more dispatches.
MTL_CAPTURE_ENABLED=1 VLLM_METAL_KV_COMMIT_PROBE=0 vllm serve mlx-community/Qwen3.5-0.8B-bf16 --port 8003 --max-model-len 2048 --gpu-memory-utilization 0.95 &
for i in {1..300}; do curl -s -o /dev/null http://127.0.0.1:8003/v1/models && break; sleep 1; done
# warm up so the worker shows GPU time in `gpucapture list`
curl -s http://127.0.0.1:8003/v1/completions -H 'Content-Type: application/json' \
  --data-binary "$(jq -n --arg prompt "$(cat benchmark_prompt.txt)" '{model:"mlx-community/Qwen3.5-0.8B-bf16",prompt:$prompt,max_tokens:4,temperature:0}')" > /dev/null
gpucapture start -p $(gpucapture list | awk '/Python/ && $4 > 0 {print $1; exit}') -c 1 -o /tmp/vllm.gputrace &
GP=$!
sleep 1
curl -s http://127.0.0.1:8003/v1/completions -H 'Content-Type: application/json' \
  --data-binary "$(jq -n --arg prompt "$(cat benchmark_prompt.txt)" '{model:"mlx-community/Qwen3.5-0.8B-bf16",prompt:$prompt,max_tokens:16,temperature:0}')" > /dev/null
wait $GP
gputrace kernels /tmp/vllm.gputrace
rm -rf /tmp/vllm.gputrace "$TMPDIR"/Python_*.gputrace

# sources
ls /opt/homebrew/Cellar/vllm-metal/*/libexec/lib/python3.12/site-packages/vllm_metal/metal/kernels_v2/

# zmlx (same, with the patch applied; 1.8 GB, delete after)
MTL_CAPTURE_ENABLED=1 python -c 'import mlx.core as mx; from mlx_lm import load, stream_generate; from mlx_lm.sample_utils import make_sampler; from zmlx.patch import patch; model, tokenizer = load("mlx-community/Qwen3.5-0.8B-bf16"); patch(model); prompt = open("benchmark_prompt.txt").read(); gen = stream_generate(model, tokenizer, prompt, max_tokens=3, sampler=make_sampler(temp=0)); next(gen); mx.metal.start_capture("/tmp/zmlx_decode.gputrace"); next(gen); mx.metal.stop_capture()'
gputrace kernels /tmp/zmlx_decode.gputrace
rm -rf /tmp/zmlx_decode.gputrace

# agave: no in-process capture hook; --profile is op timing. For a Metal trace use xctrace --launch (§1.5).
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

Parse and delete in the same run; sizes are per capture. Counts are capture-scoped (one graph, one `-c 1` buffer, or 32 tokens), not per-token and not comparable; rank by time % in §1.5.

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

Values are % of GPU time (oMLX/llama: Shader Timeline intervals; mlx_lm: PC samples). Ollama, vllm-metal, ZMLX, ZML, tinygrad: no per-kernel trace yet.

oMLX is +0.7 tok/s (~2%).
Its extra kernels are cheap at 32 tokens (`fa256` 0.3%, gdn 0.1%, copy 0.2%);
the gain is a smaller non-matmul tail, 0.8% vs mlx_lm's 7.5%. `fa256` is the piece to re-test at 4k+.

```bash
git clone https://github.com/Kr1sso/xtrace-skill ~/.cache/xtrace-skill
python3 ~/.cache/xtrace-skill/scripts/trace-template.py enable-shader-timeline \
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

| engine    | matmul % | measured by                                     |
|-----------|----------|-------------------------------------------------|
| oMLX      | 99.2     | Shader Timeline intervals (gemv bm4+bm8+bm1)    |
| llama.cpp | 96.0     | Shader Timeline intervals (mul_mv bf16+ext+mma) |
| mlx_lm    | 92.5     | PC samples (gemv bm8+bm4+bm1)                   |

PC samples = `gpu-shader-profiler-sample` PC hits: sample counts, not durations. oMLX/llama shares are Shader Timeline interval durations.

Stalls in ms ≈ limiter % × kernel duration (Shader Timeline `duration`). For overlap/timeline:

```bash
python3 ~/.cache/xtrace-skill/scripts/trace-gpu.py llama_decode.trace
```

`trace-gpu.py` reports GPU busy, command-buffer submissions and CPU→GPU start latency. For raw gaps, export `metal-application-command-buffer-submissions` and `metal-command-buffer-completed`.
