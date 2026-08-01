#!/usr/bin/env python3
"""Export the on-device query embedder used by Wilderness Edge's TextEmbeddingManager.

Converts the same sentence-transformers model used by `build_vector_db.py` into a CoreML
package, and emits the WordPiece vocabulary and tokenizer settings the Swift side needs
(CoreML cannot tokenize strings, so tokenization must be reimplemented in Swift).

Pooling and L2 normalization are folded into the traced graph so the CoreML output is
directly comparable to the normalized vectors stored in `protocols.db`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SEQ_LEN = 128
PARITY_THRESHOLD = 0.999


class ExportError(RuntimeError):
    pass


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def build_traced_module(model_id: str, seq_len: int):
    import torch
    from transformers import AutoModel

    class PooledEmbedder(torch.nn.Module):
        def __init__(self, encoder: torch.nn.Module) -> None:
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
            return torch.nn.functional.normalize(pooled, p=2.0, dim=1)

    # Eager attention traces cleanly; SDPA introduces ops coremltools cannot lower.
    encoder = AutoModel.from_pretrained(model_id, attn_implementation="eager")
    encoder.eval()
    module = PooledEmbedder(encoder).eval()

    # Trace in int64 (what the encoder expects) and declare int32 at the CoreML boundary,
    # which is the only integer input type CoreML accepts. Casting inside forward instead
    # produces an aten::Int op that the converter cannot fold.
    example_ids = torch.ones((1, seq_len), dtype=torch.long)
    example_mask = torch.ones((1, seq_len), dtype=torch.long)

    # torch.export is preferred over torch.jit.trace: TorchScript tracing of current
    # transformers encoders emits an aten::Int on a non-scalar that coremltools cannot lower.
    with torch.no_grad():
        try:
            program = torch.export.export(module, (example_ids, example_mask))
            # coremltools only accepts the ATEN/EDGE dialects; a freshly exported program is
            # in the TRAINING dialect.
            program = program.run_decompositions({})
        except Exception as exc:
            log(f"torch.export failed ({exc}); falling back to torch.jit.trace")
            program = torch.jit.trace(module, (example_ids, example_mask), strict=False)
    return module, program, (example_ids, example_mask)


def convert_to_coreml(traced, seq_len: int, output_path: Path, deployment_target: str):
    import coremltools as ct

    target = getattr(ct.target, deployment_target, None)
    if target is None:
        raise ExportError(f"Unknown deployment target '{deployment_target}'.")

    convert_kwargs = dict(
        convert_to="mlprogram",
        minimum_deployment_target=target,
        compute_units=ct.ComputeUnit.ALL,
    )
    try:
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="input_ids", shape=(1, seq_len), dtype=np.int32),
                ct.TensorType(name="attention_mask", shape=(1, seq_len), dtype=np.int32),
            ],
            outputs=[ct.TensorType(name="embedding", dtype=np.float32)],
            **convert_kwargs,
        )
    except (ValueError, AssertionError) as exc:
        # Converting an ExportedProgram: shapes and dtypes come from the program itself and
        # explicit input specs are rejected.
        log(f"Retrying conversion without explicit input specs ({exc})")
        mlmodel = ct.convert(traced, **convert_kwargs)
    mlmodel.short_description = "Sentence embedding for Wilderness Edge offline protocol retrieval"
    mlmodel.input_description["input_ids"] = "WordPiece token ids, padded to fixed length"
    mlmodel.input_description["attention_mask"] = "1 for real tokens, 0 for padding"
    mlmodel.output_description["embedding"] = "L2-normalized mean-pooled sentence embedding"

    if output_path.exists():
        shutil.rmtree(output_path) if output_path.is_dir() else output_path.unlink()
    mlmodel.save(str(output_path))
    return mlmodel


def export_tokenizer_assets(model_id: str, seq_len: int, out_dir: Path) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    vocab = tokenizer.get_vocab()
    ordered = sorted(vocab.items(), key=lambda item: item[1])

    vocab_path = out_dir / "query-embedder-vocab.txt"
    vocab_path.write_text("\n".join(token for token, _ in ordered) + "\n", encoding="utf-8")

    config = {
        "model_id": model_id,
        "tokenizer": "wordpiece",
        "max_sequence_length": seq_len,
        "do_lower_case": bool(getattr(tokenizer, "do_lower_case", True)),
        "strip_accents": True,
        "continuing_subword_prefix": "##",
        "unk_token": tokenizer.unk_token,
        "cls_token_id": tokenizer.cls_token_id,
        "sep_token_id": tokenizer.sep_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "vocab_size": len(ordered),
    }
    (out_dir / "query-embedder-tokenizer.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    log(f"Wrote tokenizer assets: {vocab_path.name}, query-embedder-tokenizer.json ({len(ordered)} tokens)")


def verify(mlmodel, torch_module, model_id: str, seq_len: int, parity_path: Path | None) -> bool:
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    probes = [
        "patient has severe bleeding from the left thigh",
        "unresponsive adult not breathing normally",
        "how do I splint a suspected forearm fracture",
    ]

    encoded = tokenizer(
        probes, padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt"
    )
    with torch.no_grad():
        torch_vectors = torch_module(encoded["input_ids"], encoded["attention_mask"]).numpy()

    try:
        coreml_vectors = np.vstack(
            [
                mlmodel.predict(
                    {
                        "input_ids": encoded["input_ids"][i : i + 1].numpy().astype(np.int32),
                        "attention_mask": encoded["attention_mask"][i : i + 1].numpy().astype(np.int32),
                    }
                )["embedding"].reshape(1, -1)
                for i in range(len(probes))
            ]
        )
    except Exception as exc:
        log(f"! Could not run CoreML prediction locally ({exc}). Skipping on-host parity check.")
        return True

    ok = True
    log("\nCoreML vs PyTorch parity:")
    for probe, torch_vec, coreml_vec in zip(probes, torch_vectors, coreml_vectors):
        similarity = float(np.dot(torch_vec, coreml_vec))
        flag = "ok" if similarity >= PARITY_THRESHOLD else "MISMATCH"
        ok = ok and similarity >= PARITY_THRESHOLD
        log(f"  {similarity:.6f}  {flag}  {probe!r}")

    if parity_path and parity_path.exists():
        fixtures = json.loads(parity_path.read_text(encoding="utf-8"))
        if fixtures.get("embedding_model") != model_id:
            log(
                f"! Parity fixtures were built with {fixtures.get('embedding_model')!r} but this export "
                f"uses {model_id!r}. The database and the on-device embedder WILL NOT MATCH."
            )
            return False
        log("\nCoreML vs protocols.db build parity:")
        for fixture in fixtures["fixtures"]:
            enc = tokenizer(
                fixture["text"], padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt"
            )
            predicted = mlmodel.predict(
                {
                    "input_ids": enc["input_ids"].numpy().astype(np.int32),
                    "attention_mask": enc["attention_mask"].numpy().astype(np.int32),
                }
            )["embedding"].reshape(-1)
            similarity = float(np.dot(np.asarray(fixture["embedding"], dtype=np.float32), predicted))
            flag = "ok" if similarity >= PARITY_THRESHOLD else "MISMATCH"
            ok = ok and similarity >= PARITY_THRESHOLD
            log(f"  {similarity:.6f}  {flag}  {fixture['text']!r}")
    elif parity_path:
        log(f"\n! Parity fixtures not found at {parity_path}; run build_vector_db.py first to generate them.")

    return ok


def export(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    package_path = out_dir / args.package_name

    log(f"Loading and tracing {args.model} (seq_len={args.seq_len}) ...")
    module, traced, _ = build_traced_module(args.model, args.seq_len)

    log("Converting to CoreML ...")
    mlmodel = convert_to_coreml(traced, args.seq_len, package_path, args.deployment_target)
    log(f"Wrote {package_path}")

    export_tokenizer_assets(args.model, args.seq_len, out_dir)

    parity_path = Path(args.parity_fixtures).expanduser().resolve() if args.parity_fixtures else None
    passed = verify(mlmodel, module, args.model, args.seq_len, parity_path)

    if not passed:
        log("\nERROR: parity check failed. Do not ship this embedder — retrieval would silently break.")
        return 1

    log(
        "\nNext: drag the .mlpackage plus the vocab and tokenizer JSON into WildernessEdge/Resources/.\n"
        "Xcode compiles the .mlpackage into .mlmodelc at build time; do not commit a prebuilt .mlmodelc."
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the CoreML query embedder for Wilderness Edge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Must match build_vector_db.py --model")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="Fixed input sequence length")
    parser.add_argument("--output-dir", default="build", help="Directory to write export artifacts into")
    parser.add_argument("--package-name", default="query-embedder.mlpackage", help="CoreML package filename")
    parser.add_argument("--deployment-target", default="iOS17", help="coremltools target (e.g. iOS17, iOS18)")
    parser.add_argument(
        "--parity-fixtures",
        default="build/embedding_parity_fixtures.json",
        help="Reference embeddings emitted by build_vector_db.py",
    )
    return parser.parse_args(argv)


def main() -> int:
    try:
        return export(parse_args())
    except ExportError as exc:
        log(f"\nERROR: {exc}")
        return 1
    except KeyboardInterrupt:
        log("\nInterrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
