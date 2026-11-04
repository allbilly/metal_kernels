// tinygrad BEAM=1, first decode kernel (token id fetch)
// DEBUG=4 BEAM=1 CACHEDB=~/.cache/tinygrad/beam-1.db python -m tinygrad.llm --model ~/.cache/gguf/Qwen3.5-0.8B-BF16.gguf --benchmark 1 2>&1 | grep -m1 -A3 'kernel void E'
#include <metal_stdlib>
using namespace metal;
kernel void E(device int* data0_1, device int* data1_4096, constant int& start_pos_, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int val0 = (*(data1_4096+start_pos_));
  *(data0_1+0) = val0;
}
