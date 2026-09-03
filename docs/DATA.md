# Data preparation

Run all commands from the repository root. Generated datasets are excluded
from Git and written under `data/<dataset>/processed/`. Upstream access
requirements and licenses still apply.

## Evaluation datasets

The single-turn runner downloads the seven paper benchmarks through Hugging
Face `datasets`: GSM8K, AIME 2024, AIME 2025, GPQA Diamond, MedQA,
HumanEval+, and MBPP+. Set `HF_TOKEN` or authenticate with
`huggingface-cli login` when required.

GAIA loads the gated `gaia-benchmark/GAIA` dataset from Hugging Face. Accept
the dataset access conditions and authenticate before running. Reported
evaluation uses the validation split; test answers require official
submission, and no train split is used. The runner reads each row's
`file_path` or `file_name`, downloads its attachment automatically, and
reuses the Hugging Face cache on later runs. By default Hugging Face stores its
dataset cache under `~/.cache/huggingface/datasets/` and downloaded attachment
files under `~/.cache/huggingface/hub/`; set `HF_HOME` to move the cache root.
No dataset-path or source-code configuration is required.

MARBLE DB downloads its benchmark JSONL directly from the public MARBLE
repository when the runner starts.

## General training data

The general phase uses these task names:

```text
wildchat,ultrachat,lmsys_chat,toucan_single_parallel,
toucan_multi_parallel,dta_tool,flan,2wiki_multihopqa
```

Install `requirements-tools.txt`, then run:

```bash
python data/wildchat/process_wildchat.py \
  --split train \
  --min_turns 3 \
  --include_all_languages \
  --output_file train_all_lang_multi_turn_min3.jsonl

python data/ultrachat/process_ultrachat.py \
  --split train_sft \
  --min_turns 3 \
  --output_file train_sft_multi_turn_min3.jsonl

python data/lmsys-chat/process_lmsys_chat.py \
  --split train \
  --min_turns 3 \
  --output_file train_all_lang_multi_turn_min3.jsonl

python data/toucan/process_toucan.py \
  --split train \
  --config_names SFT,Kimi-K2,Qwen3 \
  --validation_rows 200

python data/dta-tool/process_dta_tool.py \
  --split train \
  --output_file train_parallel_function_calls.jsonl

python data/flan/process_flan.py \
  --split train \
  --output_file train_in_context_examples.jsonl

python data/2wiki-multihopqa/process_2wiki_multihopqa.py \
  --splits train
```

The processors pin their upstream dataset revisions. The expected output
paths, row counts, byte sizes, revisions, and SHA256 checksums are listed in
[artifacts/training_data.json](../artifacts/training_data.json). General-phase
processed files do not need to be distributed separately because the scripts
rebuild them from their pinned sources.

### Toucan unified files

The Toucan command creates all four files consumed by the training and
validation loaders:

```text
data/toucan/processed/train_single_turn_parallel_tool_call_unified.jsonl
data/toucan/processed/validation_single_turn_parallel_tool_call_unified.jsonl
data/toucan/processed/train_multi_turn_parallel_tool_call.jsonl
data/toucan/processed/validation_multi_turn_parallel_tool_call.jsonl
```

It concatenates the SFT, Kimi-K2, and Qwen3 single-turn configurations in that
order. The final 200 Qwen3 single-turn rows and final 200 SFT multi-turn rows
form the validation files. The resulting training files contain 141,717
single-turn and 1,171 multi-turn examples.

## BrowseComp trajectory distillation data

The second training stage uses saved text-synthesis tool trajectories, not
BrowseComp as an evaluation benchmark. This repository includes a 1,211-row
format-filtered artifact at:

```text
data/browsecomp-textmas/processed/filtered_train.jsonl.gz
```

The training loader streams this gzip file directly, so it does not need to
be unpacked. Its integrity values are:

```text
Compressed bytes: 33,827,874
Compressed SHA256: 1b170a6e4354c7b5999e66671af4dc5e671a2ec984f0e7fda4ed4243fb7723f8
Uncompressed bytes: 194,414,797
Uncompressed SHA256: 4abd4dd4eb314784c83eec46b0475cda77dcd4a88e4af9a133977ad3e1d6f222
```

The filter validates the synthesizer output structure and rejects malformed
or degenerate responses. It does not guarantee that answers are correct or
that every conclusion is supported by the retrieved evidence. Two expired S3
presigned-URL credential identifiers were redacted from the distributable
copy; questions, trajectories, tool-result content, and answers are unchanged.

The published checkpoint used a different, unfiltered 1,266-row trajectory
file:

```text
data/browsecomp-textmas/processed/train.jsonl
SHA256: be052a666b9cd148f6f8c88b1df7ec9107139460f7d7042f1cc7102e2553c832
```

That unfiltered file is not bundled. The included 1,211-row artifact supports
the complete training workflow, but it cannot reproduce the published
checkpoint weights exactly.

Verify the bundled compressed file with:

```bash
sha256sum data/browsecomp-textmas/processed/filtered_train.jsonl.gz
gzip -dc data/browsecomp-textmas/processed/filtered_train.jsonl.gz | sha256sum
```
