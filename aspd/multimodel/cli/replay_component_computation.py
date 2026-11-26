"""Targeted native replay on verified, archived controlled-evaluation anchors."""

import argparse
import json
from pathlib import Path

import torch
from torch.distributed.checkpoint import FileSystemReader

from aspd.multimodel.cli.evaluate_aspd_v3_negation import read_contract
from aspd.multimodel.cli.export_component_computation import MODELS, factor, read, sha
from aspd.multimodel.cli.measure_induction_query import _supports
from aspd.multimodel.cli.measure_token_dependency_shard import load_dataset
from aspd.multimodel.posthoc.backend import MatrixHookSpec, ModelInputs
from aspd.multimodel.posthoc.computation import component_read_write, local_mlp, replay_native_attention
from aspd.multimodel.posthoc.ids import ComponentId, MatrixId
from aspd.multimodel.posthoc.source_hooks import capture_matrix_io
from aspd.multimodel.posthoc.v31_backend import _pad_id, load_frozen_config
from aspd.multimodel.v3.induction import subset_pool, task_pool_from_verified
from aspd.multimodel.v3.models import load_frozen_v3_model, load_v3_tokenizer
from aspd.multimodel.v3.registry import build_model_registry_2b, resolve_submodule
from aspd.multimodel.v3.training_capture import padded_model_inputs


def replay(args):
    _, reference = read_contract(args.config)
    checkpoint = reference["checkpoint_manifest_sha256"]
    rows, records, archived, source_models = load_dataset(args, checkpoint)
    component = ComponentId.from_key(args.component)
    down = ComponentId.from_key(args.downstream) if args.downstream else None
    if component.model not in MODELS or not 1 <= len(set(args.anchor)) <= 24:
        raise ValueError("Replay accepts one of the three pinned models and up to 24 unique anchors")
    if down and (component.matrix.matrix not in {"mlp_gate", "mlp_up"}
                 or down.matrix != MatrixId(component.model, component.matrix.layer, "mlp_down")):
        raise ValueError("Local replay requires Gate/Up and Down in the same model and layer")
    if args.attention and (component.matrix.matrix not in {"q", "k", "v"} or len(set(args.anchor)) > 4):
        raise ValueError("Native attention replay accepts Q/K/V and up to four exact anchors")
    archived_rows = {row["anchor_id"]: row for row in rows}
    selected = [archived_rows[key] for key in dict.fromkeys(args.anchor)]
    task = task_pool_from_verified(list({records[key]["example"]["example_id"]: records[key] for key in args.anchor}.values()),
                                   mechanism=args.dataset)
    positions = [task.anchor_ids.index(row["anchor_id"]) for row in selected]
    query = subset_pool(task, positions)
    checkpoint_path = Path(reference["checkpoint"])
    if sha(checkpoint_path / "manifest.json") != checkpoint or read(checkpoint_path / "COMPLETE")["manifest_sha256"] != checkpoint:
        raise ValueError("Replay checkpoint does not match the aligned archive")
    config = load_frozen_config(reference["training_config"], checkpoint_path,
                                skip_training_launch_evidence=reference["skip_training_launch_evidence"])
    layout = build_model_registry_2b(config)[component.model]
    native = (layout.model_id, layout.revision, layout.tokenizer_revision)
    if archived:
        recorded = archived["models"][component.model]
        expected = (recorded["model_id"], recorded["revision"], recorded["tokenizer_revision"])
    else:
        recorded = source_models[component.model]
        expected = (recorded["id"], recorded["revision"], recorded["tokenizer_revision"])
    if native != expected:
        raise ValueError("Replay model and tokenizer are not the frozen archive revisions")
    importance = Path(reference["components"]["reference_posthoc_root"]) / "importance"
    output_scales = read(importance / "importance_meta.json")
    if not args.execute:
        print(json.dumps({"anchors": len(positions), "model": component.model, "checkpoint": checkpoint,
                          "dataset": args.dataset, "requires_forward": True}))
        return

    model = load_frozen_v3_model(layout, args.device, local_files_only=True).forward_model.eval()
    if args.attention:
        model.set_attn_implementation("eager")
    tokenizer = load_v3_tokenizer(layout, local_files_only=True)
    padded, docs, tokens = padded_model_inputs(query.models[component.model], pad_token_id=_pad_id(tokenizer),
                                                device=torch.device(args.device))
    inputs = ModelInputs(padded["input_ids"], padded["attention_mask"], docs, tokens)
    names = {component.matrix}
    if down:
        names.update(MatrixId(component.model, component.matrix.layer, name) for name in ("mlp_gate", "mlp_up", "mlp_down"))
    specs = {}
    for target in layout.targets():
        matrix = MatrixId(component.model, target.layer_index, target.logical_name)
        if matrix not in names:
            continue
        path, _ = resolve_submodule(model, target.physical_module_candidates)
        sliced = target.output_slice
        specs[matrix] = MatrixHookSpec(matrix, path, (sliced.start, sliced.stop) if sliced else None, target.d_in, target.d_out)
    if specs.keys() != names:
        raise ValueError("Logical matrices do not match the pinned model")
    captured = capture_matrix_io(model, inputs, specs.values())
    rank = MODELS.index(component.model)
    location = checkpoint_path / f"rank-{rank:05d}"
    if read(location / "sidecar.json")["model_key"] != component.model:
        raise ValueError("Wrong checkpoint owner")
    dcp = location / "dcp"
    meta = FileSystemReader(dcp).read_metadata()

    def load_factor(identity, name):
        key = "model.matrix_bank.components." + identity.matrix.key.replace(".", "__dot__")
        value, _ = factor(dcp, meta, key + "." + name, identity.feature)
        return value.to(args.device)

    u, v = load_factor(component, "U"), load_factor(component, "V")
    if down:
        down_v = load_factor(down, "V")
        mlp_path = specs[down.matrix].module_path.rsplit(".", 1)[0]
        activate = model.get_submodule(mlp_path).act_fn
    result = []
    for i, row in enumerate(selected):
        support = _supports(row)
        activation = support[component.gate.site].get(component.feature, 0.0)
        x = captured[component.matrix][0][i]
        summary = component_read_write(x, u, v, bool(activation), output_scales[component.matrix.key]["sigma"])
        out = {"anchor": row["anchor_id"], "pool": row["pool_id"], "modelPosition": int(tokens[i]),
               "gateActivation": activation, "support": bool(activation), **summary}
        if down:
            gate_matrix = MatrixId(component.model, component.matrix.layer, "mlp_gate")
            up_matrix = MatrixId(component.model, component.matrix.layer, "mlp_up")
            out["local"] = local_mlp(captured[gate_matrix][1][i], captured[up_matrix][1][i], activate,
                                     summary["coefficient"] * u, component.matrix.matrix, down_v,
                                     down.feature in support[down.gate.site],
                                     checked_native_down_input=captured[down.matrix][0][i])
        if args.attention:
            document = int(docs[i])
            single = ModelInputs(inputs.input_ids[document:document + 1], inputs.attention_mask[document:document + 1],
                                 torch.zeros_like(docs[i:i + 1]), tokens[i:i + 1])
            out["local"] = replay_native_attention(model, single, specs[component.matrix], summary["coefficient"] * u)
        result.append(out)
    with args.output.open("x") as handle:
        json.dump({"schemaVersion": 1, "checkpoint": checkpoint,
                   "modelRevision": native[1], "tokenizerRevision": native[2], "component": component.key,
                   "downstream": down.key if down else None, "dataset": args.dataset, "split": args.split,
                   "attentionBackend": "eager" if args.attention else None,
                   "configSha256": sha(args.config), "activationManifestSha256": sha(args.activations / "manifest.json"),
                   "preparationManifestSha256": sha(args.preparation / "manifest.json"), "rows": result}, handle, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("negation", "induction", "selection-frame"), required=True)
    parser.add_argument("--split", choices=("smoke", "discovery"), required=True)
    for name in ("config", "preparation", "activations", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--component", required=True)
    parser.add_argument("--anchor", action="append", required=True)
    parser.add_argument("--downstream")
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--execute", action="store_true", help="Load one source model and replay the specified anchors")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new artifact")
    replay(args)


if __name__ == "__main__":
    main()
