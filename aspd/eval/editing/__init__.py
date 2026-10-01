"""Causal weight editing: remove selected components from W and measure the effect on output-SAE features.

Procedure: rank components for each target feature j by
effect_{j,c} = M_{j,c} E_{t in A_j}[zeta_c(t)]; delete the top-k from the frozen weight,
W' = W - sum_c u_c v_c^T; rerun the model and measure localization, the change in target features
divided by the total change in all features; divide by the same measure for random components.
"""
