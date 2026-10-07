"""Multi-model ASPD: one sparse coordinate system across models and weight matrices.

The package follows the notation in ``docs/multimodel/method.md``.  In particular, model index
``n`` is never folded into the latent index ``c``: one shared ``g^s_{t,c}`` gates model-specific
rank-1 components ``P^{(n)}_{j,c} = u^{(n)}_{j,c} v^{(n)T}_{j,c}``.
"""

from aspd.multimodel.config import MultiModelExperimentConfig, load_experiment_config
from aspd.multimodel.model import MultiModelASPD

__all__ = ["MultiModelASPD", "MultiModelExperimentConfig", "load_experiment_config"]
