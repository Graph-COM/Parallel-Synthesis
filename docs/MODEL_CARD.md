---
license: apache-2.0
base_model: Qwen/Qwen3-14B
tags:
- qwen3
- peft
- lora
- kv-cache
- multi-agent
- agentic-workflows
---

# Parallel Synthesis for Qwen3-14B

This repository contains the released trainable components for
**Parallel Synthesis**, introduced in [Towards Direct Latent-Space Synthesis
for Parallel Branches in LLM-Agent Workflows](https://arxiv.org/abs/2606.14672).
Parallel Synthesis lets a synthesizer consume the KV caches produced by
independent worker agents directly, avoiding the usual step of concatenating
and prefilling all worker outputs again.

The checkpoint is designed for the open-source
[Parallel Synthesis codebase](https://github.com/Graph-COM/Parallel-Synthesis)
and the frozen [`Qwen/Qwen3-14B`](https://huggingface.co/Qwen/Qwen3-14B)
backbone.

## Important: this is not a standalone Transformers model

Do not pass this repository to `AutoModelForCausalLM.from_pretrained`. It is a
custom component bundle containing a cache mapper and a PEFT LoRA adapter; it
does not contain the Qwen3-14B backbone. The project runner loads the backbone
and both trained components in their correct roles.

## Files

```text
cache_mapper.pt
judger_lora/
  adapter_config.json
  adapter_model.safetensors
```

- `cache_mapper.pt` is a lightweight length- and worker-count-aware affine
  mapper for re-encoded worker KV caches.
- `judger_lora/` is the synthesizer LoRA applied to the frozen Qwen3-14B
  backbone.

Component sizes and SHA256 checksums are recorded in the
[release manifest](https://github.com/Graph-COM/Parallel-Synthesis/blob/main/artifacts/release_checkpoint.json).

## Usage

```bash
git clone https://github.com/Graph-COM/Parallel-Synthesis.git
cd Parallel-Synthesis

conda create -n parallel-synthesis python=3.10 -y
conda activate parallel-synthesis
pip install -e .

parallel-synthesis-single \
  --checkpoint_dir Graph-COM/Parallel-Synthesis-qwen3-14B \
  --method parallel_kv \
  --model_name Qwen/Qwen3-14B \
  --tasks aime2024,gpqa,humanevalplus \
  --split test \
  --eval_samples_per_task 10 \
  --temperature 0 \
  --top_p 1 \
  --output_dir results/single_turn
```

The runner downloads this repository and the Qwen3-14B backbone to the local
Hugging Face cache on first use. To download the component checkpoint manually:

```bash
hf download Graph-COM/Parallel-Synthesis-qwen3-14B \
  --local-dir checkpoints/parallel-synthesis-qwen3-14b
```

Then replace the Hub repository ID passed to `--checkpoint_dir` with that local
directory. GAIA and MARBLE DB use `--parallel_kv_load_dir` for the same value;
see the project [evaluation guide](https://github.com/Graph-COM/Parallel-Synthesis/blob/main/docs/EVALUATION.md).

## Model details

- **Backbone:** Qwen/Qwen3-14B, loaded separately and kept frozen during
  Parallel Synthesis training
- **Backbone revision:** `40c069824f4251a91eefaf281ebe4c544efd3e18`
- **Checkpoint type:** custom Parallel Synthesis components
- **Cache mapper:** hidden dimension 32; conditioned on worker-output length
  and number of parallel workers
- **Synthesizer adapter:** LoRA rank 16, alpha 32, dropout 0; applied to
  `q_proj`, `k_proj`, `v_proj`, and `o_proj`
- **Adapter vocabulary size:** 151,669 (the loader aligns the backbone
  embeddings before attaching the adapter)
- **License:** Apache License 2.0
- **Languages:** inherits the multilingual coverage of Qwen3-14B; training data
  is primarily English and multilingual chat/tool-use data

## Training

The Qwen3-14B backbone remained frozen while the cache mapper and synthesizer
LoRA were optimized with teacher-forced next-token loss over parallel cache
contexts.

Training used two stages followed by a checkpoint merge:

1. General adaptation on multi-turn dialogue, parallel tool-use, in-context
   learning, and multi-document QA sources. The source pool includes WildChat,
   UltraChat, LMSYS-Chat, Toucan, DTA-Tool, FLAN, and 2WikiMultiHopQA.
2. Reasoning distillation on 1,266 saved BrowseComp text-synthesis trajectories.
3. A 50/50 parameter average of the BrowseComp adapter and an earlier general
   adapter. The released cache mapper comes from the earlier general checkpoint.

The release recipes use AdamW with learning rate `1e-4`, weight decay `0`, a
per-rank batch size of one, four distributed processes, and one epoch per
training stage. Exact data revisions, row counts, checksums, preprocessing,
checkpoint lineage, and memory guards are documented in the project
[data](https://github.com/Graph-COM/Parallel-Synthesis/blob/main/docs/DATA.md)
and [training](https://github.com/Graph-COM/Parallel-Synthesis/blob/main/docs/TRAINING.md)
guides.

The public code repository includes a filtered 1,211-row BrowseComp trajectory
artifact for exercising the training workflow. That artifact is not the
unfiltered 1,266-row file used to produce this checkpoint and therefore cannot
reproduce these weights exactly.

## Evaluation

The paper evaluates Parallel Synthesis on nine downstream datasets spanning
math, science and medical QA, code generation, multi-turn tool use, and
multi-agent database diagnosis:

- GSM8K, AIME 2024, AIME 2025, GPQA Diamond, and MedQA
- HumanEval+ and MBPP+
- GAIA Levels 1–3
- MARBLE DB

Across those datasets, Parallel Synthesis matches or outperforms conventional
text-concatenation synthesis on seven of nine datasets and remains close on the
other two. It reduces synthesis time-to-first-token by 2.5x–11x in the reported
experiments. See the [paper](https://arxiv.org/abs/2606.14672) for the complete
protocol, baselines, and per-dataset results.

## Intended use

This release is intended for research on cache-based context interfaces,
parallel and multi-agent workflows, synthesis efficiency, and reproduction or
extension of the paper's experiments.

It is not intended as a drop-in chat model, a standalone PEFT adapter, or a
general-purpose replacement for Qwen3-14B. It has not been validated for
high-stakes medical, legal, financial, safety-critical, or autonomous decision
making.

## Limitations and risks

- The checkpoint requires the matching Qwen3-14B architecture and the custom
  Parallel Synthesis runtime. Other backbones and ordinary text-generation
  pipelines are unsupported.
- Outputs may be incorrect, unsupported, biased, or inconsistent, including
  when worker branches or tool results contain errors. The checkpoint inherits
  limitations and risks from Qwen3-14B and its training data.
- Reported quality and latency depend on prompts, worker count, branch lengths,
  hardware, attention kernels, dependency versions, and tool availability.
- GAIA web results can change over time. GAIA's optional Python interpreter
  executes model-produced code and should only be enabled in an isolated
  environment.
- MARBLE DB should be run only against the disposable Docker environment
  supplied by the project, never a database containing valuable data.
- This release adds no independent safety alignment or guarantee of factuality.

Users should review generated outputs and apply safeguards appropriate to their
deployment context.

## Software environment

The release was validated with Python 3.10.19, PyTorch 2.9.1+cu128,
Transformers 4.57.6, PEFT 0.18.1, Hugging Face Hub 0.36.2, Datasets 4.5.0,
and SDPA attention. A CUDA-capable GPU is recommended. The `parallel_kv` route
uses the Transformers backend; vLLM is not currently supported for this route.

## Citation

```bibtex
@article{liu2026parallel_synthesis,
  title={Towards Direct Latent-Space Synthesis for Parallel Branches in LLM-Agent Workflows},
  author={Liu, Shikun and Li, Mufei and Fu, Dongqi and Wang, Haoyu and Xia, Yinglong and Li, Hong and Yan, Hong and Li, Pan},
  journal={arXiv preprint arXiv:2606.14672},
  year={2026}
}
```

## Acknowledgments

This checkpoint builds on Qwen3-14B and PEFT. Upstream models, datasets, and
benchmarks remain subject to their own licenses and terms.
