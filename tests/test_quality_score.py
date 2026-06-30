"""Quality scoring was removed from AdaptiveDrain. Tests skipped."""

import pytest


@pytest.mark.skip(reason="quality_score fields removed from ManagedTemplate")
def test_quality_fields_default_to_none():
    pass


@pytest.mark.skip(reason="quality_score fields removed from ManagedTemplate")
def test_quality_fields_saved_and_loaded():
    pass


@pytest.mark.skip(reason="quality_score fields removed from ManagedTemplate")
def test_quality_score_range():
    pass


@pytest.mark.skip(reason="quality_score fields removed from ManagedTemplate")
def test_quality_fields_in_saved_json():
    pass
