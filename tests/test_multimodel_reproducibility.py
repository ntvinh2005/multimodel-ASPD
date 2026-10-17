from pathlib import Path

from aspd.multimodel.feasibility import compare_tokenizer_vocabularies
from aspd.multimodel.provenance import source_state


def test_tokenizer_comparison_reports_id_conflicts_and_unique_tokens() -> None:
    result = compare_tokenizer_vocabularies(
        {"same": 1, "conflict": 2, "left": 3},
        {"same": 1, "conflict": 4, "right": 5},
    )
    assert not result["identical"]
    assert result["same_id_common_tokens"] == 1
    assert result["conflicting_common_tokens"] == ["conflict"]
    assert result["only_left"] == ["left"]
    assert result["only_right"] == ["right"]


def test_source_state_fingerprints_dirty_and_untracked_content(tmp_path: Path) -> None:
    # A non-repository path returns an explicit unavailable state rather than false provenance.
    result = source_state(tmp_path)
    assert result == {
        "git_commit": None,
        "git_dirty": None,
        "git_status": None,
        "source_state_sha256": None,
    }
