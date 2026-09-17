"""TP3 launcher defaults ABLIT=0 and does not inherit it from .env."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_start_tp3_clears_ablit_after_env() -> None:
    launcher = (ROOT / "start-tp3.sh").read_text()
    assert 'source "$SCRIPT_DIR/.env"\n# TP=3 does not inherit ABLIT=1' in launcher
    assert "\nABLIT=0\n# TP=3 overlay wins over the 2× knobs in .env." in launcher
    assert "\nABLIT=0\n" in (ROOT / ".env.tp3.example").read_text()


if __name__ == "__main__":
    test_start_tp3_clears_ablit_after_env()
    print("tp3 ABLIT default OK")
