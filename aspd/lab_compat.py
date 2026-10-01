"""Makes the lab's own tools (harvest, app, autointerp) parse this package's configs and build its CI functions."""

from pathlib import Path


def widen_lab_config_parsing() -> None:
    from param_decomp_lab.experiments.lm.run import LMExperimentConfig, SavedLMRun

    from param_decomp_lab.infra.hf_http import configure_hf_http_retries

    from aspd.checkpoints import install_decomposition_only_checkpoints
    from aspd.ci.setup import attach_ci_fn, install_ci_fns
    from aspd.component_setup import install_component_parameterization
    from aspd.loader_patch import (
        install_bos_for_tokenizers_that_add_it,
        install_streamed_features_resolution,
    )
    from aspd.topology import install_qwen3_path_schema, install_split_qkv_path_schema
    from aspd.config import LMInterpExperimentConfig

    def _from_file(cls: type, path: Path | str) -> LMInterpExperimentConfig:
        del cls
        cfg = LMInterpExperimentConfig.from_file(path)
        install_component_parameterization(cfg)
        return cfg

    LMExperimentConfig.from_file = classmethod(_from_file)  # pyright: ignore[reportAttributeAccessIssue]

    if not getattr(SavedLMRun.load_model, "_aspd_attach_patched", False):
        stock_load_model = SavedLMRun.load_model

        def _load_model(self: SavedLMRun):
            model = stock_load_model(self)
            attach_ci_fn(model)
            return model

        _load_model._aspd_attach_patched = True  # pyright: ignore[reportFunctionMemberAccess]
        SavedLMRun.load_model = _load_model  # pyright: ignore[reportAttributeAccessIssue]
    install_ci_fns()
    install_decomposition_only_checkpoints()
    install_split_qkv_path_schema()
    install_qwen3_path_schema()
    install_streamed_features_resolution()
    install_bos_for_tokenizers_that_add_it()
    configure_hf_http_retries()
