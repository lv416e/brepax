"""Persistent JAX compilation cache for BRepAX.

BRepAX compiles one XLA artifact per surface type and per unique BSpline
signature the first time a shape is triangulated.  On a busy part such as
NIST CTC-02 this is ~10 seconds of one-shot work on a fresh Python
process.  Enabling the persistent compilation cache lets subsequent
process starts reuse the compiled artifacts from disk, bringing cold
triangulation below 6 seconds on CTC-02.

The cache is opt-in: nothing on BRepAX import changes ``jax.config``.
Users call :func:`enable_compilation_cache` early in their program.

Cached artifacts may produce numerically equivalent but not
bit-identical results compared to fresh compilation because XLA's
optimization pipeline can choose slightly different reduction orders
per run.  The difference is within standard floating-point tolerance
(~1e-5 relative on accumulated volumes).
"""

from __future__ import annotations

import os
from pathlib import Path

import jax

_ENV_VAR = "BREPAX_COMPILATION_CACHE_DIR"


def _resolve_cache_dir(path: str | os.PathLike[str] | None) -> Path:
    """Resolve the target cache directory without side effects.

    Precedence: explicit ``path`` argument > ``$BREPAX_COMPILATION_CACHE_DIR``
    > ``$XDG_CACHE_HOME/brepax/jax-compile`` > ``~/.cache/brepax/jax-compile``.
    """
    if path is not None:
        return Path(path)
    env = os.environ.get(_ENV_VAR)
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "brepax" / "jax-compile"


def enable_compilation_cache(path: str | os.PathLike[str] | None = None) -> Path:
    """Enable JAX's persistent compilation cache for BRepAX.

    First call in a fresh Python process triggers XLA compilation of
    BRepAX's per-surface-type and per-BSpline-signature JITs; subsequent
    process starts read those artifacts from disk.  The function is
    idempotent and safe to call before any BRepAX API use.

    Args:
        path: Directory for the cache.  If omitted, falls back to the
            ``BREPAX_COMPILATION_CACHE_DIR`` environment variable, then
            ``XDG_CACHE_HOME/brepax/jax-compile``, finally
            ``~/.cache/brepax/jax-compile``.  The directory is created
            if it does not exist.

    Returns:
        The resolved absolute cache directory.

    Examples:
        >>> import brepax
        >>> cache_dir = brepax.enable_compilation_cache()  # doctest: +SKIP
        >>> # subsequent BRepAX calls populate the cache on first run
        >>> # and reuse it on every subsequent process start
    """
    resolved = _resolve_cache_dir(path).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    # jax.config.update's public return is untyped in jax's stubs.
    jax.config.update("jax_compilation_cache_dir", str(resolved))  # type: ignore[no-untyped-call]
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)  # type: ignore[no-untyped-call]
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)  # type: ignore[no-untyped-call]
    return resolved


__all__ = ["enable_compilation_cache"]
