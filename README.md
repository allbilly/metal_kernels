# README

This is a repo learning metal kernels on m1 air 8gb macOS 27.2.
I want to write sth like the [yalm blog by andrewkchan](https://andrewkchan.dev/posts/yalm.html) 

The plan 
1. profile qwen3.6 0.8B metal kernels from oMLX / mlx.lm / tinygrad
2. PR the faster kernel to the slower engine
2. Bypass apple code with AGXForge(https://github.com/sbryngelson/agxforge) 

# 1. profile kernels from oMLX / mlx.lm / tinygrad

```
brew install omlx
uv tool install --python 3.13 mlx.lm==0.32.2
```
