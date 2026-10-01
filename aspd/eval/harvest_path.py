"""Resolve `--harvest-dir` to the `harvest.db` inside it."""

from pathlib import Path

_STEP_GLOB = "h-step*"


def resolve_harvest_db(harvest_dir: Path | str) -> Path:
    """`<harvest_dir>/harvest.db`, asserting it exists and is non-empty."""
    harvest_dir = Path(harvest_dir)
    assert harvest_dir.is_dir(), f"--harvest-dir {harvest_dir} is not a directory"
    db = harvest_dir / "harvest.db"
    if db.is_file() and db.stat().st_size > 0:
        return db

    steps = sorted(d for d in harvest_dir.glob(_STEP_GLOB) if (d / "harvest.db").is_file())
    if steps:
        raise AssertionError(
            f"{harvest_dir} holds no harvest.db of its own -- it is a CHECKPOINTED harvest, one "
            f"level up from the databases. Pass one of:\n"
            + "\n".join(f"    --harvest-dir {d}" for d in steps)
            + "\n(a decomposition or transcoder harvest is per checkpoint; a dictionary harvest is "
            "flat, which is the layout these tools are documented with.)"
        )
    if db.is_file():
        raise AssertionError(
            f"{db} is 0 bytes -- an empty database left behind by a readonly open of a path that "
            f"had no harvest (`?immutable=1` creates the file rather than failing). Delete it and "
            f"pass the directory that really holds the harvest."
        )
    raise AssertionError(
        f"{db} does not exist, and {harvest_dir} contains no {_STEP_GLOB}/harvest.db either."
    )
