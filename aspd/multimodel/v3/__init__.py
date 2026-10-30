"""ASPD v3 three-family training infrastructure.

This package is intentionally independent of the historical Mix3 and original-ASPD
configuration paths.  The main v3 run remains blocked until the unresolved scientific
choices represented by :mod:`aspd.multimodel.v3.config` are approved.
"""

from aspd.multimodel.v3.config import load_v3_config_bundle

__all__ = ["load_v3_config_bundle"]
