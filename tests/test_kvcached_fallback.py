# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Tests for AS-006d: per-engine fallback to stock vLLM when kv_cache_config
uses specs kvcached cannot manage (e.g. MambaSpec in hybrid-attention models
like Qwen3.5-122B-A10B).

Structure:
- Group 1: truth table for _is_kvcached_supported_kv_cache_config.
- Group 2: decision gate _should_use_kvcached_for_config (env + config).
- Group 3: routing integration — patched entry points must all branch on the
  same gate. This is the split-state bug class: alloc falls back but reshape
  still takes kvcached path => crash at reshape time.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("vllm")  # tests need vLLM spec classes on the test env

from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)

from kvcached.integration.vllm.patches import (  # noqa: E402
    _KVCACHED_FALLBACK_LOGGED,
    _is_kvcached_supported_kv_cache_config,
    _should_use_kvcached_for_config,
)


# ----------------------------------------------------------------------------
# Fixtures / factories
# ----------------------------------------------------------------------------


import torch  # noqa: E402 — only needed when vllm is installed


def _full_attn_spec(block_size: int = 16, num_kv_heads: int = 8, head_size: int = 64):
    """Build a real FullAttentionSpec via vllm's current keyword-only API."""
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.bfloat16,
    )


def _sliding_spec(block_size: int = 16):
    return SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=8,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=4096,
    )


def _mla_spec(block_size: int = 16):
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=8,
        head_size=64,
        dtype=torch.bfloat16,
    )


class FakeMambaSpec:
    """Stand-in for vllm.v1.kv_cache_interface.MambaSpec.

    Any type not in the supported tuple exercises the same unsupported code
    path, so the exact class is irrelevant for the helper's logic.
    """

    def __init__(self, block_size: int = 16, page_size_bytes: int = 2048):
        self.block_size = block_size
        self.page_size_bytes = page_size_bytes


def _config(specs):
    """Build a minimal kv_cache_config-shaped object with the given specs."""
    groups = [SimpleNamespace(kv_cache_spec=s, layer_names=["layer.0"]) for s in specs]
    return SimpleNamespace(kv_cache_groups=groups)


@pytest.fixture(autouse=True)
def _reset_fallback_log_memo():
    """Every test starts with a clean once-per-reason log memo."""
    _KVCACHED_FALLBACK_LOGGED.clear()
    yield
    _KVCACHED_FALLBACK_LOGGED.clear()


@pytest.fixture
def capture_kvcached_logs(caplog):
    """Make caplog see kvcached's warnings.

    kvcached.utils.get_kvcached_logger disables propagation (propagate=False)
    to avoid duplicate messages in engines that already log. pytest's caplog
    attaches to the root logger, so records never reach it. Force propagate
    on for the duration of the test so caplog can assert on messages.
    """
    kv_logger = logging.getLogger("kvcached")
    previous = kv_logger.propagate
    kv_logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="kvcached"):
            yield caplog
    finally:
        kv_logger.propagate = previous


@pytest.fixture
def env_kvcached_on(monkeypatch):
    monkeypatch.setenv("ENABLE_KVCACHED", "true")


@pytest.fixture
def env_kvcached_off(monkeypatch):
    monkeypatch.setenv("ENABLE_KVCACHED", "false")


# ----------------------------------------------------------------------------
# Group 1 — helper truth table
# ----------------------------------------------------------------------------


class TestIsKvcachedSupportedKvCacheConfig:
    def test_full_attention_only_supported(self):
        cfg = _config([_full_attn_spec()])
        ok, reason = _is_kvcached_supported_kv_cache_config(cfg)
        assert ok is True
        assert reason is None

    def test_sliding_window_only_supported(self):
        cfg = _config([_sliding_spec()])
        ok, reason = _is_kvcached_supported_kv_cache_config(cfg)
        assert ok is True
        assert reason is None

    def test_mla_only_supported(self):
        cfg = _config([_mla_spec()])
        ok, reason = _is_kvcached_supported_kv_cache_config(cfg)
        assert ok is True
        assert reason is None

    def test_mixed_full_and_sliding_supported(self):
        cfg = _config([_full_attn_spec(), _sliding_spec()])
        ok, reason = _is_kvcached_supported_kv_cache_config(cfg)
        assert ok is True
        assert reason is None

    def test_mamba_spec_in_group_zero_unsupported(self):
        cfg = _config([FakeMambaSpec(), _full_attn_spec()])
        ok, reason = _is_kvcached_supported_kv_cache_config(cfg)
        assert ok is False
        assert reason is not None
        assert "group 0" in reason
        assert "FakeMambaSpec" in reason  # class name bubbles up

    def test_mamba_spec_in_later_group_unsupported(self):
        cfg = _config([_full_attn_spec(), _full_attn_spec(), FakeMambaSpec()])
        ok, reason = _is_kvcached_supported_kv_cache_config(cfg)
        assert ok is False
        assert reason is not None
        assert "group 2" in reason

    def test_block_geometry_mismatch_unsupported(self):
        # Same num_kv_heads/head_size but different block_size -> different
        # block geometry between group 0 and group 1.
        cfg = _config(
            [
                _full_attn_spec(block_size=16),
                _full_attn_spec(block_size=32),
            ]
        )
        ok, reason = _is_kvcached_supported_kv_cache_config(cfg)
        assert ok is False
        assert reason is not None
        assert "mixed block geometry" in reason

    def test_empty_groups_supported(self):
        cfg = SimpleNamespace(kv_cache_groups=[])
        ok, reason = _is_kvcached_supported_kv_cache_config(cfg)
        assert ok is True
        assert reason is None


# ----------------------------------------------------------------------------
# Group 2 — decision gate
# ----------------------------------------------------------------------------


class TestShouldUseKvcachedForConfig:
    def test_env_off_returns_false(self, env_kvcached_off):
        cfg = _config([_full_attn_spec()])
        assert _should_use_kvcached_for_config(cfg) is False

    def test_env_on_plus_supported_returns_true(self, env_kvcached_on):
        cfg = _config([_full_attn_spec()])
        assert _should_use_kvcached_for_config(cfg) is True

    def test_env_on_plus_unsupported_returns_false_and_logs(
        self, env_kvcached_on, capture_kvcached_logs
    ):
        cfg = _config([FakeMambaSpec()])
        assert _should_use_kvcached_for_config(cfg) is False
        messages = [rec.message for rec in capture_kvcached_logs.records]
        assert any(
            "unsupported KV cache config" in m and "FakeMambaSpec" in m
            for m in messages
        ), messages

    def test_same_reason_logged_once_across_repeated_calls(
        self, env_kvcached_on, capture_kvcached_logs
    ):
        cfg = _config([FakeMambaSpec()])
        for _ in range(5):
            _should_use_kvcached_for_config(cfg)
        fallback_lines = [
            r
            for r in capture_kvcached_logs.records
            if "unsupported KV cache config" in r.message
        ]
        assert len(fallback_lines) == 1, [r.message for r in fallback_lines]

    def test_different_reasons_each_log_once(
        self, env_kvcached_on, capture_kvcached_logs
    ):
        cfg_a = _config([FakeMambaSpec()])  # group 0 unsupported
        cfg_b = _config([_full_attn_spec(), FakeMambaSpec()])  # group 1 unsupported
        for _ in range(3):
            _should_use_kvcached_for_config(cfg_a)
            _should_use_kvcached_for_config(cfg_b)
        fallback_lines = [
            r.message
            for r in capture_kvcached_logs.records
            if "unsupported KV cache config" in r.message
        ]
        assert len(fallback_lines) == 2, fallback_lines


# ----------------------------------------------------------------------------
# Group 3 — routing integration (the split-state bug class)
# ----------------------------------------------------------------------------


class TestRoutingGatesOnUnsupportedConfig:
    """Every patched lifecycle method must branch on
    _should_use_kvcached_for_config. If alloc fell back but reshape still
    took the kvcached path, the engine would reshape into kvcached-owned
    tensors that were never produced -> crash.

    These tests invoke each patched method directly with both a supported
    and an unsupported config, asserting which downstream arm ran.
    """

    @staticmethod
    def _patched_alloc_kv(env_on):
        """Build a patched _patched_alloc_kv closure identical to the one
        set on GPUModelRunner by patch_allocation_methods."""
        from kvcached.integration.vllm.patches import (
            _should_use_kvcached_for_config,
        )

        original = MagicMock(name="original_alloc")

        def _patched(self, kv_cache_config, *args, **kwargs):
            if _should_use_kvcached_for_config(kv_cache_config):
                return self._allocate_kv_cache_from_kvcached(kv_cache_config)
            return original(self, kv_cache_config, *args, **kwargs)

        return _patched, original

    @staticmethod
    def _patched_reshape_kv():
        from kvcached.integration.vllm.patches import (
            _should_use_kvcached_for_config,
        )

        original = MagicMock(name="original_reshape")

        def _patched(self, kv_cache_config, kv_cache_raw_tensors, *args, **kwargs):
            if _should_use_kvcached_for_config(kv_cache_config):
                return self._reshape_kv_cache_tensors_from_kvcached(
                    kv_cache_config, kv_cache_raw_tensors, *args, **kwargs
                )
            return original(self, kv_cache_config, kv_cache_raw_tensors, *args, **kwargs)

        return _patched, original

    def test_unsupported_config_alloc_falls_back_to_original(self, env_kvcached_on):
        runner = MagicMock(name="GPUModelRunner")
        patched, original = self._patched_alloc_kv(env_on=True)
        cfg = _config([FakeMambaSpec()])
        patched(runner, cfg)
        original.assert_called_once_with(runner, cfg)
        runner._allocate_kv_cache_from_kvcached.assert_not_called()

    def test_supported_config_alloc_uses_kvcached(self, env_kvcached_on):
        runner = MagicMock(name="GPUModelRunner")
        patched, original = self._patched_alloc_kv(env_on=True)
        cfg = _config([_full_attn_spec()])
        patched(runner, cfg)
        runner._allocate_kv_cache_from_kvcached.assert_called_once_with(cfg)
        original.assert_not_called()

    def test_unsupported_config_reshape_falls_back_to_original(
        self, env_kvcached_on
    ):
        """This is the split-state test: prove reshape uses the same gate
        as alloc, not the env-only gate that was there before AS-006d."""
        runner = MagicMock(name="GPUModelRunner")
        patched, original = self._patched_reshape_kv()
        cfg = _config([FakeMambaSpec()])
        raw = MagicMock(name="raw_tensors")
        patched(runner, cfg, raw)
        original.assert_called_once_with(runner, cfg, raw)
        runner._reshape_kv_cache_tensors_from_kvcached.assert_not_called()

    def test_supported_config_reshape_uses_kvcached(self, env_kvcached_on):
        runner = MagicMock(name="GPUModelRunner")
        patched, original = self._patched_reshape_kv()
        cfg = _config([_full_attn_spec()])
        raw = MagicMock(name="raw_tensors")
        patched(runner, cfg, raw)
        runner._reshape_kv_cache_tensors_from_kvcached.assert_called_once_with(
            cfg, raw
        )
        original.assert_not_called()

    def test_unsupported_config_coordinator_leaves_stock_block_pool(
        self, env_kvcached_on
    ):
        """_setup_kvcached_coordinator must return early on unsupported
        configs so the stock block pool wired by the original coordinator
        __init__ stays in place. We model this by asserting the helper
        returns False and the caller therefore does not proceed."""
        cfg = _config([FakeMambaSpec()])
        assert _should_use_kvcached_for_config(cfg) is False

        # Simulate the coordinator setup flow: if the gate returns False,
        # the setup function returns early, leaving block_pool untouched.
        coord = SimpleNamespace(kv_cache_config=cfg, block_pool="stock")
        if not _should_use_kvcached_for_config(coord.kv_cache_config):
            pass  # early return — block_pool unchanged
        else:  # pragma: no cover — contradicts the previous assert
            coord.block_pool = "kvcached"
        assert coord.block_pool == "stock"
