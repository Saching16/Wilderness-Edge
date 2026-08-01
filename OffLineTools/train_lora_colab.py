#!/usr/bin/env python3
"""Fine-tune the Wilderness Edge decision LoRA on Gemma 4 E2B and export a `.litertlm` bundle.

Intended to run on a Colab GPU (a free T4 is sufficient — E2B LoRA needs roughly 8-10 GB of
VRAM). It does not run on the Mac used for the rest of OffLineTools.

    !pip install -q unsloth trl datasets
    !python train_lora_colab.py --hf-token $HF_TOKEN

Stages: load Gemma 4 E2B -> attach LoRA to language layers only -> supervised fine-tune on
the dataset from build_training_data.py -> merge to 16-bit safetensors -> convert to
`.litertlm` via litert-torch, including the vision encoder.

Model size tradeoff, worth understanding before you run this
------------------------------------------------------------
Google's smallest published Gemma 4 E2B mobile build is ~1.1 GB, but it is produced with a
QAT int2/int4 mobile recipe from quantized safetensors that `litert-torch export_hf` cannot
currently consume. Exporting your own fine-tune therefore means post-training quantization,
where int4 (`dynamic_wi4c_hr_afp32`) lands nearer ~2.9 GB and the int8 default lands nearer
~5.7 GB. This script defaults to int4 for that reason.

That is a bundle-size cost, not necessarily a runtime memory cost: LiteRT-LM memory-maps
weights with XNNPACK weight caching, so resident memory is far below file size. Validate
actual resident usage on the target device against PLAN.md's 2.8 GB Phase 3 ceiling before
committing to this path — if the fine-tune is not earning its keep, shipping the stock
prebuilt `litert-community/gemma-4-E2B-it-litert-lm` bundle is a legitimate fallback.

Safety note: this adapter is trained to enforce cited, non-diagnostic checklist output. It is
a defense-in-depth layer, never a replacement for the on-device SafetyFilter.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BASE_MODEL = "google/gemma-4-E2b-it"
CHAT_TEMPLATE_SOURCE = "litert-community/gemma-4-E2B-it-litert-lm"

# int4 post-training recipe; see the module docstring for why the int8 default is unsuitable.
LLM_QUANTIZATION_RECIPE = "dynamic_wi4c_hr_afp32"
VISION_QUANTIZATION_RECIPE = "dynamic_wi8_afp32"


def log(message: str) -> None:
    print(f"\n=== {message}", file=sys.stderr, flush=True)


def require_gpu() -> None:
    try:
        import torch
    except ImportError:
        raise SystemExit("torch is not installed. Run this on a Colab GPU runtime.")
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU detected. This script must run on a GPU runtime "
            "(Colab: Runtime > Change runtime type > T4 GPU)."
        )
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    log(f"GPU: {name} ({total:.1f} GB)")


def load_dataset_file(path: Path):
    from datasets import Dataset

    if not path.exists():
        raise SystemExit(
            f"Dataset not found: {path}\nGenerate it first with:  python build_training_data.py"
        )

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise SystemExit(f"{path} is empty.")

    malformed = [i for i, r in enumerate(records) if len(r.get("messages", [])) != 3]
    if malformed:
        raise SystemExit(f"{len(malformed)} records lack the expected 3 messages (first: index {malformed[0]}).")

    log(f"Loaded {len(records)} training examples from {path}")
    return Dataset.from_list(records)


def load_model(args):
    from unsloth import FastLanguageModel

    log(f"Loading {args.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        token=args.hf_token,
    )

    log("Attaching LoRA adapters (language layers only; vision tower frozen)")
    peft_kwargs = dict(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            **peft_kwargs,
        )
    except TypeError:
        # Older Unsloth builds lack the multimodal freeze flags; naming the language
        # projections explicitly achieves the same thing.
        model = FastLanguageModel.get_peft_model(
            model,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            **peft_kwargs,
        )
    return model, tokenizer


def train(model, tokenizer, dataset, args):
    from trl import SFTConfig, SFTTrainer

    def formatting(batch) -> list[str]:
        return [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            for messages in batch["messages"]
        ]

    config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=args.seed,
        max_length=args.max_seq_length,
        report_to="none",
        save_strategy="epoch",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=config,
        formatting_func=formatting,
    )

    log("Training")
    stats = trainer.train()
    log(f"Training complete: {stats.metrics}")
    return trainer


def merge_and_save(model, tokenizer, merged_dir: Path) -> Path:
    log(f"Merging LoRA into base weights -> {merged_dir}")
    merged_dir.mkdir(parents=True, exist_ok=True)
    # litert-torch export_hf reads standard 16-bit safetensors; it cannot consume adapters
    # or pre-quantized checkpoints.
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    return merged_dir


def export_litertlm(merged_dir: Path, output_dir: Path, args) -> Path:
    log("Converting to .litertlm")
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "litert-torch", "export_hf",
        f"--model={merged_dir}",
        f"--output_dir={output_dir}",
        "--externalize_embedder",
        f"--jinja_chat_template_override={CHAT_TEMPLATE_SOURCE}",
        f"--quantization_recipe={args.quantization_recipe}",
    ]
    if args.export_vision:
        # Required for the camera-snapshot path; without it the bundle is text-only.
        command += [
            "--task=image_text_to_text",
            "--export_vision_encoder",
            f"--vision_encoder_quantization_recipe={args.vision_quantization_recipe}",
            "--experimental_lightweight_conversion",
        ]

    print(" ".join(command), file=sys.stderr)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(
            "litert-torch export failed. Install it with:  uv tool install litert-torch-nightly\n"
            "Note that export_hf cannot consume quantized/QAT checkpoints — pass 16-bit "
            "merged safetensors only."
        )

    produced = sorted(output_dir.glob("*.litertlm"))
    if not produced:
        raise SystemExit(f"Export reported success but no .litertlm file appeared in {output_dir}")

    bundle = produced[0]
    size_gb = bundle.stat().st_size / 1024**3
    log(f"Wrote {bundle} ({size_gb:.2f} GB)")
    if size_gb > 3.5:
        print(
            f"WARNING: {size_gb:.2f} GB is large for a 6 GB device. Consider a more aggressive "
            "quantization recipe, or the prebuilt litert-community bundle.",
            file=sys.stderr,
        )
    return bundle


def smoke_test(bundle: Path) -> None:
    log("Smoke-testing the exported bundle")
    prompt = (
        "Retrieved protocol excerpts:\n[Source: US Army TCCC Handbook v5, p. 12]\n"
        "Apply a tourniquet 2-3 inches above the wound. Tighten until bleeding stops.\n\n"
        "Responder asked: severe bleeding from the thigh, what do I do"
    )
    result = subprocess.run(
        ["litert-lm", "run", str(bundle), f"--prompt={prompt}"], check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(
            "Could not run litert-lm locally (install with: uv tool install litert-lm). "
            "Validate on-device instead.",
            file=sys.stderr,
        )
        return
    print(result.stdout)
    print(
        "Check that the response cites the source, gives numbered steps, and offers no "
        "diagnosis before shipping this bundle.",
        file=sys.stderr,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the Wilderness Edge decision LoRA and export it for LiteRT-LM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--hf-token", default=None, help="Needed because Gemma weights are gated")
    parser.add_argument("--dataset", default="build/wfr_lora_dataset.jsonl")
    parser.add_argument("--output-dir", default="outputs/gemma4_e2b_wfr")
    parser.add_argument("--merged-dir", default="outputs/gemma4_e2b_wfr_merged")
    parser.add_argument("--litertlm-dir", default="outputs/litertlm")

    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1, help="Keep at 1 on a T4")
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3407)

    parser.add_argument("--quantization-recipe", default=LLM_QUANTIZATION_RECIPE)
    parser.add_argument("--vision-quantization-recipe", default=VISION_QUANTIZATION_RECIPE)
    parser.add_argument(
        "--no-vision",
        dest="export_vision",
        action="store_false",
        help="Export a text-only bundle (breaks the camera-snapshot feature)",
    )
    parser.add_argument("--skip-export", action="store_true", help="Train and merge only")
    parser.add_argument("--skip-smoke-test", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    require_gpu()

    dataset = load_dataset_file(Path(args.dataset).expanduser().resolve())
    model, tokenizer = load_model(args)
    train(model, tokenizer, dataset, args)
    merged = merge_and_save(model, tokenizer, Path(args.merged_dir).expanduser().resolve())

    if args.skip_export:
        log(f"Skipping export. Merged weights are at {merged}")
        return 0

    bundle = export_litertlm(merged, Path(args.litertlm_dir).expanduser().resolve(), args)
    if not args.skip_smoke_test:
        smoke_test(bundle)

    log(
        f"Done. Rename {bundle.name} to gemma-4-e2b-wfr.litertlm and place it in "
        "WildernessEdge/Resources/."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
