"""
Benchmark and validation tool for evaluating sub-bit quantized foundation models.
Supports evaluating numerical stability and activation energy gain across model layers,
as well as relative reconstruction fidelity (Frobenius error, cosine similarity) against
unquantized reference weights when provided.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

import torch
from weft.serve_any import UniversalSubBitModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("weft.eval_fidelity")


def evaluate_layer_reconstruction(
    model: UniversalSubBitModel,
    layer_name: str,
    batch_size: int = 4,
    seq_len: int = 64,
    reference_weight: torch.Tensor = None,
) -> Dict[str, float]:
    """
    Evaluates forward layer numerical stability, activation energy gain, and optionally
    relative Frobenius error and cosine similarity against reference unquantized weights.
    """
    if layer_name not in model.layer_metadata:
        raise KeyError(f"Layer {layer_name} not found in model.")

    meta = model.layer_metadata[layer_name]
    in_features = meta["in_features"]
    out_features = meta["out_features"]

    # Generate synthetic activation tensor
    x = torch.randn(batch_size, seq_len, in_features, dtype=torch.float32)

    with torch.no_grad():
        out = model.forward_layer(layer_name, x)

    # Compute statistics
    out_norm = torch.linalg.norm(out).item()
    in_norm = torch.linalg.norm(x).item()
    energy_gain = out_norm / max(in_norm, 1e-6)

    has_nans = bool(torch.isnan(out).any().item())
    has_infs = bool(torch.isinf(out).any().item())

    res = {
        "layer": layer_name,
        "in_features": in_features,
        "out_features": out_features,
        "rank": meta["rank"],
        "energy_gain": float(energy_gain),
        "is_numerically_stable": not (has_nans or has_infs),
    }

    if reference_weight is not None:
        with torch.no_grad():
            ref_out = torch.nn.functional.linear(x, reference_weight.float())
            rel_frob = (torch.linalg.norm(out - ref_out) / torch.linalg.norm(ref_out).clamp_min(1e-6)).item()
            cos_sim = torch.nn.functional.cosine_similarity(
                out.reshape(-1), ref_out.reshape(-1), dim=0
            ).item()
            res["relative_frobenius_error"] = float(rel_frob)
            res["cosine_similarity"] = float(cos_sim)

    return res


def evaluate_model_health(model_dir: str) -> Dict[str, Any]:
    """
    Loads all subbit shards in model_dir and checks numerical fidelity across representative layers.
    """
    logger.info(f"Loading model from: {model_dir}")
    shard_paths = sorted([str(p) for p in Path(model_dir).glob("*.weft")])
    if not shard_paths:
        raise FileNotFoundError(f"No .weft shards found in {model_dir}")

    model = UniversalSubBitModel(shard_paths)

    summary = model.summary()
    layer_names = list(model.layer_metadata.keys())

    # Sample representative layers: first, middle, last
    sample_indices = [0, len(layer_names) // 2, len(layer_names) - 1] if layer_names else []
    sample_layers = [layer_names[i] for i in sample_indices if i < len(layer_names)]

    eval_results = []
    for l_name in sample_layers:
        res = evaluate_layer_reconstruction(model, l_name)
        eval_results.append(res)
        logger.info(f"Layer {l_name}: energy_gain={res['energy_gain']:.4f}, stable={res['is_numerically_stable']}")

    all_stable = all(r["is_numerically_stable"] for r in eval_results)

    report = {
        "model_summary": summary,
        "evaluated_sample_count": len(eval_results),
        "all_layers_numerically_stable": all_stable,
        "layer_evaluations": eval_results,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate fidelity and numerical stability of subbit model.")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing .weft shards")
    parser.add_argument("--out-json", type=str, default=None, help="Optional output JSON report path")
    args = parser.parse_args()

    report = evaluate_model_health(args.model_dir)
    print(json.dumps(report, indent=2))

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {args.out_json}")


if __name__ == "__main__":
    main()
