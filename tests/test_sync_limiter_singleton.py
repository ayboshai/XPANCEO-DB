"""
Regression test: sync limiter must not be recreated on each call with config.
This verifies singleton behavior and that iter configs are respected.
"""

import os
import sys

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import get_sync_limiter


def _make_config(max_concurrent_vision: int) -> dict:
    return {
        "max_concurrent_embeddings": 5,
        "max_concurrent_vision": max_concurrent_vision,
        "max_concurrent_judge": 3,
        "api_rate_limit_rpm": 500,
    }


def test_sync_limiter_singleton_with_config():
    config = _make_config(6)

    limiter1 = get_sync_limiter(config)
    limiter2 = get_sync_limiter(config)
    limiter3 = get_sync_limiter()

    assert limiter1 is limiter2
    assert limiter2 is limiter3
    # Ensure config was applied (internal semaphore value)
    assert getattr(limiter1._vision_sem, "_value", None) == 6

