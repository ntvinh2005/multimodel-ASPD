"""Figures for the evaluation results, written beside the results they describe."""

import traceback
from collections.abc import Callable
from pathlib import Path

__all__ = ["safe_plot", "plot_ce_kl_dir", "plot_scr_tpp_dir", "plot_matching_dir",
           "plot_saebench_dir", "plot_attr_edit_dir", "plot_attr_edit_multi_dir", "plot_aggregate"]


def safe_plot(fn: Callable[..., list[Path]], *args, **kwargs) -> list[Path]:
    try:
        return fn(*args, **kwargs)
    except Exception:  # noqa: BLE001 -- see the module docstring
        print(f"[plots] FAILED in {getattr(fn, '__name__', fn)}; the results are unaffected",
              flush=True)
        traceback.print_exc()
        return []


def plot_matching_dir(matching_dir: Path) -> list[Path]:
    from aspd.eval.plots.matching import plot_dir

    return plot_dir(matching_dir)


def plot_attr_edit_dir(attr_edit_dir: Path) -> list[Path]:
    from aspd.eval.plots.editing import plot_dir

    return plot_dir(attr_edit_dir)


def plot_attr_edit_multi_dir(multi_dir: Path) -> list[Path]:
    """Every `(m, setup)` arm's own figure, then the cross-arm one that joins them."""
    from aspd.eval.plots.editing_multi import _ARM_RE, plot_arm_dir, plot_multi_root

    multi_dir = Path(multi_dir)
    arms = sorted(
        ((int(match["m"]), match["setup"], path)
         for path in multi_dir.iterdir()
         if path.is_dir() and (match := _ARM_RE.match(path.name))),
        key=lambda arm: arm[:2],
    )
    written: list[Path] = []
    for _m, _setup, path in arms:
        written += plot_arm_dir(path)
    return written + plot_multi_root(multi_dir)


def plot_aggregate(run_dir: Path | None, sae_dir: Path | None, out_dir: Path) -> list[Path]:
    from aspd.eval.plots.aggregate import plot_aggregate as _plot

    return _plot(run_dir, sae_dir, out_dir)
