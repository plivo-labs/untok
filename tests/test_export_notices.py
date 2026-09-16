"""Static source and asset licenses remain part of the installed distribution."""
from importlib import resources
from pathlib import Path


def test_packaged_notices_match_repository_sources():
    root = Path(__file__).resolve().parents[1]
    notices = resources.files("untok").joinpath("notices")
    assert notices.joinpath("THIRD_PARTY.md").read_bytes() == (root / "THIRD_PARTY.md").read_bytes()
    for path in (root / "licenses").glob("*.txt"):
        assert notices.joinpath("licenses", path.name).read_bytes() == path.read_bytes()
