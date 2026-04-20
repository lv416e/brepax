"""Tests for brepax.compilation_cache."""

from __future__ import annotations

from pathlib import Path

import jax
import pytest

from brepax.compilation_cache import _resolve_cache_dir, enable_compilation_cache


class TestResolveCacheDir:
    """Path resolution precedence: arg > env var > XDG > ~/.cache."""

    def test_explicit_path_wins_over_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("BREPAX_COMPILATION_CACHE_DIR", str(tmp_path / "env"))
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        explicit = tmp_path / "explicit"
        assert _resolve_cache_dir(explicit) == explicit

    def test_env_var_wins_over_xdg(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        env_path = tmp_path / "env"
        monkeypatch.setenv("BREPAX_COMPILATION_CACHE_DIR", str(env_path))
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        assert _resolve_cache_dir(None) == env_path

    def test_xdg_wins_over_home(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("BREPAX_COMPILATION_CACHE_DIR", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        assert _resolve_cache_dir(None) == tmp_path / "brepax" / "jax-compile"

    def test_home_fallback(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("BREPAX_COMPILATION_CACHE_DIR", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        assert (
            _resolve_cache_dir(None) == tmp_path / ".cache" / "brepax" / "jax-compile"
        )


class TestEnable:
    """Side-effecting test; one invocation to limit JAX config pollution."""

    def test_creates_dir_and_sets_jax_config(self, tmp_path: Path) -> None:
        cache = tmp_path / "brepax_cache"
        resolved = enable_compilation_cache(cache)
        assert resolved == cache.resolve()
        assert cache.is_dir()
        assert jax.config.jax_compilation_cache_dir == str(resolved)
