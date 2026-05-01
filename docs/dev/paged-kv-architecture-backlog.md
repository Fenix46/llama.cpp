# Paged KV Architecture Backlog

This document tracks the remaining work for the paged KV scheduler and paged
attention kernels after the first Metal migration. Mark items as complete only
after the architecture has been loaded, generated with `--scheduler paged`, and
validated against a representative model.

## Current Status

- [x] Standard KV attention graphs pass paged tensors through `FLASH_ATTN_EXT`
      via `block_table`, `seq_ids_q`, and `page_limits_q`.
- [x] ISWA/SWA KV graphs pass separate base and SWA paged tensors.
- [x] Hybrid attention + recurrent memory graphs pass paged tensors for the
      attention side.
- [x] Metal paged attention generic kernel works for contiguous F16 KV.
- [x] Metal paged decode fast path for F16 `dk64/dv64`, validated on
      `LFM2-2.6B-Q4_0`.
- [ ] CUDA paged attention implementation and H200 validation.

## Architecture Matrix

| Architecture family | Load / paged wiring | Kernel status | Notes |
| --- | --- | --- | --- |
| LLaMA-like standard KV | Done | Generic Metal; CUDA TODO | Includes common decoder-only MHA/GQA models using `build_attn_inp_kv()`. |
| Gemma 3 | Done | Generic Metal unless covered by existing dims | Uses KV or ISWA path depending on SWA metadata. |
| Gemma 3n | Done | Generic Metal unless covered by existing dims | Uses special KV layer reuse in memory creation. |
| Gemma 4 | Done | Generic Metal unless covered by existing dims | Uses ISWA-capable path and special KV layer reuse. |
| Qwen 3 | Done | Generic Metal unless covered by existing dims | Standard KV path. |
| Qwen 3 MoE | Done | Generic Metal unless covered by existing dims | Standard KV path plus MoE layers. |
| Qwen 3 Next | Done | Generic Metal; hybrid recurrent kernels separate | Hybrid path via `build_inp_mem_hybrid()`. |
| Qwen 3.5 / Qwen 3.5 MoE | Done | Generic Metal; hybrid recurrent kernels separate | Arch names in this tree are `qwen35` and `qwen35moe`. |
| Qwen 3.6 | Unknown | Unknown | No distinct `qwen36` arch exists in this tree. Check GGUF `general.architecture`; if it maps to `qwen3`, `qwen3next`, or `qwen35`, use that row. |
| LFM2 / LFM2 MoE | Done | Metal F16 `dk64/dv64` fast path done; other dims TODO | LFM2-2.6B validated for long paged generation. |
| Jamba | Done | Generic Metal; recurrent kernels separate | Hybrid path. |
| Falcon-H1 | Done | Generic Metal; recurrent kernels separate | Hybrid path with custom layer filters. |
| Plamo2 | Done | Generic Metal; recurrent kernels separate | Hybrid path. |
| Granite Hybrid | Done | Generic Metal; recurrent kernels separate | Hybrid path. |
| Nemotron-H / Nemotron-H-MoE | Done | Generic Metal; recurrent kernels separate | Hybrid path with custom layer filters. |
| Kimi Linear non-MLA | Done | Generic Metal; recurrent kernels separate | Hybrid KV path. |
| Kimi Linear MLA | Gap | Not migrated to paged attention | Uses K-only attention input; see MLA section below. |
| DeepSeek2 / DeepSeek2-OCR MLA | Gap | Not migrated to paged attention | Uses K-only attention input; see MLA section below. |
| GLM-DSA | Gap | Not migrated to paged attention | Uses DeepSeek2 MLA builder. |
| Mistral4 MLA | Gap | Not migrated to paged attention | Uses DeepSeek2 MLA builder. |
| Mamba / Mamba2 | Not applicable | Recurrent kernels only | No KV attention cache to page. |
| RWKV6 / RWKV7 / ARWKV7 | Not applicable | Recurrent kernels only | No KV attention cache to page. |
| BERT / embedding / diffusion no-cache models | Not applicable | No paged generation target | These use no-cache or encoder-style attention paths. |

## Paged Wiring Work

- [ ] Migrate K-only / MLA attention input to paged tensors.
  - Affected graph inputs: `build_attn_inp_k()` and
    `build_inp_mem_hybrid_k()`.
  - Affected architectures: DeepSeek2 MLA, DeepSeek2-OCR MLA, GLM-DSA,
    Mistral4, Kimi Linear MLA.
  - Required graph change: pass `block_table`, `seq_ids_q`, and
    `page_limits_q` into `build_attn_mha()` for K-only attention, or add a
    dedicated paged MLA attention op if the current tensor layout cannot use
    the generic FA path safely.

- [ ] Add a runtime diagnostic for paged fallback coverage.
  - Log when `FLASH_ATTN_EXT` receives paged tensors but falls back to a generic
    kernel.
  - Include head dims, KV type, mask/sink/bias/softcap flags, and backend.

- [ ] Add a representative model checklist per architecture family.
  - Record model path or Hugging Face ID, quant type, expected arch name,
    context length, and validation prompt.

## Metal Kernel Work

- [x] F16 `dk64/dv64` paged decode fast path.
- [ ] F16/BF16 `dk128/dv128` paged decode fast path.
- [ ] F16/BF16 `dk96/dv96` paged decode fast path.
- [ ] F16/BF16 `dk80/dv80` paged decode fast path.
- [ ] F16/BF16 `dk32/dv32` paged decode fast path.
- [ ] Quantized KV paged decode fast paths matching existing non-paged vec
      coverage.
- [ ] Paged kernel support for sinks, ALiBi/bias, and softcap without falling
      back to the slow generic path.
- [ ] ISWA-specific validation on a real Gemma/Qwen-style SWA model.
- [ ] Long-generation slope benchmark for each new fast path.

## CUDA / H200 Work

- [x] Identify current CUDA `FLASH_ATTN_EXT` dispatch points for paged tensors.
- [ ] Add CUDA paged attention gather path using `block_table` and
      `page_limits_q`.
  - Initial conservative path exists in `ggml/src/ggml-cuda/fattn.cu` for
    masked F32 Q + F16 K/V, `dk/dv` 64 and 128. It must be CUDA-built and
    validated on H200 before this item is marked complete.
- [ ] Implement CUDA decode fast path for single-token generation.
- [ ] Implement CUDA tiled/prefill path for paged KV.
- [ ] Validate on H200 with at least:
  - [ ] LFM2-2.6B or LFM2 family hybrid model.
  - [ ] Gemma 3 or Gemma 4 SWA/ISWA model.
  - [ ] Qwen3/Qwen3.5 model.
  - [ ] MLA model: DeepSeek2, GLM-DSA, Mistral4, or Kimi Linear MLA.
- [ ] Compare against non-paged scheduler and record:
  - prompt tokens/s,
  - 128-token decode tokens/s,
  - 1700-token long-generation slope,
  - memory footprint,
  - max admitted concurrent full-context requests.

## Validation Protocol

For every checked item:

1. Build the relevant backend.
2. Start `llama-server` with `--scheduler paged`.
3. Run a short decode benchmark, at least 128 generated tokens.
4. Run a long decode benchmark, at least 1700 generated tokens when context
   allows it.
5. Compare generated token IDs against a known-good baseline when the sampler is
   deterministic.
6. Record the result in the architecture row or add a note below.

## Notes

- Loading successfully is not the same as having an optimized paged kernel.
  Most supported architectures currently run through the generic paged Metal
  kernel unless their head dims and flags match an implemented fast path.
- `Qwen 3.6` must be checked from the GGUF metadata. The current architecture
  registry has `qwen3`, `qwen3next`, `qwen35`, and `qwen35moe`, but no separate
  `qwen36` entry.
