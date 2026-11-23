"""Multimodel ASPD post-hoc analysis: edge scores, parent sets and cross-model comparison.

Mathematics: ``docs/multimodel/multimodel_aspd_formulation.tex`` ("Post-hoc analysis").
Runbook, equation-to-code map and open decisions: ``docs/multimodel/posthoc_analysis.md``.

Module map (each module is independently testable; only ``source_hooks``/``edges`` and
the backends touch a ``torch.nn.Module``)::

    ids              MatrixId, ComponentId, GateId; p(j), compute order, gamma(b)
    contributions    e_{j,alpha,i}, ||Delta y||^2
    importance       streaming sigma_j, beta, q, rho  -> ImportanceTable
    representatives  R_{p,i}^{(n)}
    source_hooks     capture x/y, zero probes on y, exact component ablation
    edges            E_{b->a}(alpha) via one backward per downstream component
    parents          Pa_{K_par,tau}(a; alpha), G_{p,i}^{(n)}(alpha)
    comparison       Jaccard, w, S, pi, D_TV, M
    backend          PosthocBackend protocol + AnchorPool/ModelInputs/MatrixHookSpec
    toy_backend      tiny random Llama models for tests and smoke runs
    v31_backend      the ASPD v3.1 three-model checkpoint (not validated on GPU yet)
    config           YAML schema; null decisions block only the stages that need them
    artifacts        immutable stage directories, COMPLETE hashes, provenance
    pipeline         importance -> representatives -> edges -> compare

CLI: ``python -m aspd.multimodel.cli.posthoc_analysis --help``.
"""
