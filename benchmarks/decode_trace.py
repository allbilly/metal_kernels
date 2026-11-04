"""Run a 64-token decode for a Metal trace, then idle so profile.sh can finish."""

import time
from pathlib import Path

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parents[1]
model, tok = load("mlx-community/Qwen3.5-0.8B-bf16")
prompt = (ROOT / "benchmark_prompt_10k.txt").read_text()[:5465]
sampler = make_sampler(temp=0)

for _ in stream_generate(model, tok, "hello", max_tokens=4, sampler=sampler):
    pass
print("READY", flush=True)
time.sleep(5)

last = None
for r in stream_generate(model, tok, prompt, max_tokens=64, sampler=sampler):
    last = r
print(
    "tokens %d gen_tps %.2f prompt_tps %.1f"
    % (last.generation_tokens, last.generation_tps, last.prompt_tps),
    flush=True,
)
time.sleep(25)
