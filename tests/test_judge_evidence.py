import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_continuous_demo_artifacts_match_reviewed_capture():
    video = ROOT / "docs/evidence/p23-live-showcase.mp4"
    preview = ROOT / "docs/evidence/p23-live-showcase-preview.jpg"
    assert digest(video) == "ddf4239bdd50985f1c5ddb7ae2dfd75001033f56d0895261d720b062b95922a4"
    assert digest(preview) == "52aa3e7194d542d864bf7bd9346d4e655fa189412dd9f77876ee76ba4bf659bd"
    assert b"ftyp" in video.read_bytes()[:32]
    assert preview.read_bytes().startswith(b"\xff\xd8\xff")


def test_judge_package_links_the_required_evidence_boundaries():
    judge = (ROOT / "docs/JUDGE_GUIDE.md").read_text(encoding="utf-8")
    architecture = (ROOT / "docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    model_card = (ROOT / "docs/MODEL_CARD.md").read_text(encoding="utf-8")
    assert "p23-live-showcase.mp4" in judge
    assert "SCRIPTED ROUTING (NOT MODEL ACCURACY)" in judge
    assert "documentation lookup only" in architecture
    assert "Fail closed" in architecture
    assert "Not intended" in model_card
    assert "4/11" in model_card
