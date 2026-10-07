# Deviations from the parent methods

Every intentional deviation is listed here so a config change cannot silently redefine the method.

1. **Range-ASPD is new.** Original ASPD can share an encoder across matrices reading a residual
   site, but the supplied experiment uses one range-entry or range-exit `R^(n)` to gate every
   matrix in a layer.
2. **The model axis is new.** Original ASPD has one target model. Here `g^s_{t,c}` is shared across
   models while activation decoders and parameter factors are model-specific.
3. **The main run is smaller.** `C=8192` and about 0.5M unique training tokens replace the much
   larger parent-paper settings. Training may revisit cached tokens across steps.
4. **Targets are reconstructed from snapshots.** The cache stores `X_j^(n)` and `W_j^(n)`, then
   forms `Y_j^(n)` during training. This is algebraically exact for the weight matrix and excludes
   module bias, matching `y=Wx`.
5. **No decoder row normalization.** Cross-model decoder norms are needed for P1 and S2. The shared
   activation-code scaling gauge changes every model decoder together, so relative norms remain
   invariant.
6. **AuxK is activation-side.** Dead features fit the residual of `L_act` using every model-specific
   activation decoder. Internal reconstruction has no separate AuxK term.
7. **D1 tying is optional and limited to equal shapes.** Cross-architecture experiments must leave
   tying off and compare normalized post-hoc quantities instead.
