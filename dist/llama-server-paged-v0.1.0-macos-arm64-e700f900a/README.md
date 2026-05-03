# llama-server paged scheduler v0.1.0

Initial macOS arm64 package for the private `llama.cpp` paged scheduler fork.

## Contents

- `bin/llama-server`
- local `libllama`, `libggml`, `libmtmd`, and backend dynamic libraries
- `LICENSE`

Models are not included. Pass a local `.gguf` model with `-m`.

## Requirements

- macOS arm64
- Apple Silicon Metal runtime
- Homebrew OpenSSL 3, matching the current build:

```sh
brew install openssl@3
```

## Run

```sh
./bin/llama-server \
  -m /path/to/model.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  -ngl 99 \
  --scheduler paged \
  --max-model-len 128000 \
  --gpu-memory-utilization 0.90 \
  --kv-prefix-cache \
  --cache-ram 0 \
  --webui
```

## Baseline Smoke Benchmark

Single local run on MacBook Air Apple M2 with `LFM2-2.6B-Q4_0.gguf`:

| Metric | Value |
| --- | --- |
| Global KV pool | 389120 tokens |
| Per-request context | 128000 tokens |
| Full-context concurrency | 3 requests |
| KV buffer | 6080 MiB |
| Prompt eval | 36 tokens in 274.02 ms, 131.38 tok/s |
| Decode | 952 tokens in 25577.22 ms, 37.22 tok/s |
| Total | 988 tokens in 25851.24 ms |

Build commit: `e700f900a`
