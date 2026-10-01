"""Patches to the lab's LM data loader: BOS handling for tokenizers that add one, and resumable stream state."""

from typing import Any


def install_streamed_features_resolution() -> None:
    """Idempotent. Call before anything builds an LM loader over a streamed raw-text corpus."""
    from param_decomp_lab.experiments.lm import data

    if getattr(data._keep_single_column, "_aspd_resolve_features", False):
        return
    stock = data._keep_single_column

    def _keep_single_column(dataset: Any, col_name: str) -> Any:
        if getattr(dataset, "features", None) is None and hasattr(dataset, "_resolve_features"):
            dataset = dataset._resolve_features()
        return stock(dataset, col_name)

    _keep_single_column._aspd_resolve_features = True  # pyright: ignore[reportFunctionMemberAccess]
    data._keep_single_column = _keep_single_column


def install_bos_for_tokenizers_that_add_it() -> None:
    """Idempotent. Give the on-the-fly packer the BOS its tokenizer prepends by default."""
    from param_decomp_lab.experiments.lm import data

    if getattr(data._tokenize_and_concatenate, "_aspd_bos", False):
        return
    stock = data._tokenize_and_concatenate

    def _tokenize_and_concatenate(
        dataset: Any,
        tokenizer: Any,
        column_name: str,
        max_length: int = 1024,
        add_bos_token: bool = False,
        num_proc: int = 10,
        to_lower: bool = False,
    ) -> Any:
        wants_bos = bool(getattr(tokenizer, "add_bos_token", False))
        if wants_bos:
            assert tokenizer.bos_token_id is not None, (
                f"{tokenizer} sets add_bos_token=True but has no bos_token_id, so the packer has "
                "nothing to prefix"
            )
        return stock(
            dataset,
            tokenizer,
            column_name=column_name,
            max_length=max_length,
            add_bos_token=add_bos_token or wants_bos,
            num_proc=num_proc,
            to_lower=to_lower,
        )

    _tokenize_and_concatenate._aspd_bos = True  # pyright: ignore[reportFunctionMemberAccess]
    data._tokenize_and_concatenate = _tokenize_and_concatenate
    # `_prepare_lm_dataset` calls it through the module global, so the single rebind is enough.
