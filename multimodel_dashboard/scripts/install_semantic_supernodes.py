"""Install a sealed Natural Validation supernode pilot in a static workbench directory."""

import argparse
import gzip
import hashlib
import json
import re
import tempfile
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path, data):
    contents = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if path.suffix == ".gz":
        contents = gzip.compress(contents, mtime=0)
    path.write_bytes(contents)
    return {"file": path.name, "sha256": hashlib.sha256(contents).hexdigest()}


def install(pilot, workbench, archived=None, validation=None):
    complete = load(pilot / "COMPLETE.json")
    for name, digest in complete["sha256"].items():
        if Path(name).name != name or sha(pilot / name) != digest:
            raise ValueError(f"Invalid sealed pilot artifact {name}")
    catalog = load(pilot / "catalog.json")
    run = catalog["runId"]
    shared_reference = catalog.get("mode") == "shared-reference-joint"
    if (
        not re.fullmatch("[a-f0-9]{64}", run)
        or run != complete["runId"]
        or catalog["dataset"] != "natural-validation"
    ):
        raise ValueError("Unexpected pilot provenance")
    if archived is not None:
        supplement = load(archived)
        if (supplement["runId"], supplement["checkpoint"], supplement["catalogSha256"]) != (
            run,
            catalog["checkpoint"],
            complete["sha256"]["catalog.json"],
        ):
            raise ValueError("Archived witnesses use a different semantic partition")
        catalog["taylor"]["witnesses"].extend(supplement["witnesses"])
        catalog["taylor"]["archive"] = {k: v for k, v in supplement.items() if k != "witnesses"}
        catalog["taylor"]["archiveSha256"] = sha(archived)
        # The browser release identity binds the partition and exact set of graph files.
        run = hashlib.sha256(f"{run}/{catalog['taylor']['archiveSha256']}".encode()).hexdigest()
        catalog["runId"] = run
    if validation is not None:
        replay = load(validation)
        if (
            replay["runId"],
            replay["checkpoint"],
            replay["gates"],
            replay["measurementContract"],
        ) != (
            complete["runId"],
            catalog["checkpoint"],
            "frozen-binary",
            catalog["measurementContract"],
        ):
            raise ValueError("Native ablation belongs to a different partition or gate protocol")
        if not replay["measurements"]:
            raise ValueError("Native ablation has no measured rows")
        run = hashlib.sha256(f"{run}/{sha(validation)}".encode()).hexdigest()
        catalog["runId"] = run
        catalog["exactAblation"] = {"status": "measured", "result": replay}
    evidence = load(pilot / "evidence.json")
    components = [item["id"] for item in catalog["components"]]
    counts = [len(evidence["examples"][key]["heldout"]) for key in components]
    if not shared_reference and (min(counts) < 10 or sum(counts) / len(counts) < 20):
        raise ValueError("Insufficient independent held-out coverage")
    directory = workbench / "semantic-supernodes"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / run
    with tempfile.TemporaryDirectory(dir=directory, prefix=".stage-") as staging:
        stage = Path(staging) / "run"
        stage.mkdir()
        catalog["examples"] = {}
        for i, key in enumerate(components):
            examples = evidence["examples"][key]
            anchors = {item["anchor"] for items in examples.values() for item in items}
            catalog["examples"][key] = save(
                stage / f"component-{i}.json.gz",
                {
                    "runId": run,
                    "examples": examples,
                    "witnesses": {a: evidence["witnesses"][a] for a in anchors},
                },
            )
        ids = {
            pair[side]
            for match in catalog["matches"]
            for pair in match.get("transported", [])
            for side in ("left", "right")
        }
        catalog["transport"] = save(
            stage / "transport.json.gz",
            {
                "runId": run,
                "witnesses": {a: evidence["witnesses"][a] for a in ids},
            },
        )
        catalog_ref = save(stage / "catalog.json.gz", catalog)
        marker = {
            "schemaVersion": 1,
            "runId": run,
            "checkpoint": catalog["checkpoint"],
            "catalog": catalog_ref,
            "pilotCompleteSha256": sha(pilot / "COMPLETE.json"),
        }
        if destination.exists():
            if sorted(p.name for p in stage.iterdir()) != sorted(
                p.name for p in destination.iterdir()
            ):
                raise ValueError("Installed semantic supernode release is incomplete")
            for path in stage.iterdir():
                if sha(destination / path.name) != sha(path):
                    raise ValueError("Installed semantic supernode release is immutable")
        else:
            stage.rename(destination)
        pointer_name = "shared-reference.json" if shared_reference else "index.json"
        pointer = Path(staging) / pointer_name
        save(pointer, marker)
        pointer.replace(directory / pointer_name)
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--workbench", required=True, type=Path)
    parser.add_argument("--archived", type=Path)
    parser.add_argument("--validation", type=Path)
    args = parser.parse_args()
    print(json.dumps(install(args.pilot, args.workbench, args.archived, args.validation), indent=2))


if __name__ == "__main__":
    main()
