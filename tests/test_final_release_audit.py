from src.final_release_audit import EXPECTED_EVIDENCE, EXPECTED_PARENTS, FORBIDDEN_TRACKED


def test_source_branch_tips_are_frozen_to_the_integrated_baseline():
    assert EXPECTED_PARENTS == {
        "work/perception": "fe70eab58185881190fc3ddc257f0933159384ed",
        "work/isaac-factory": "0f144ae65c23b13e3b92e0cf29d61ee71680122b",
    }


def test_generated_data_and_weights_are_forbidden_but_evidence_is_allowed():
    assert FORBIDDEN_TRACKED.fullmatch("data/raw/source/image.jpg")
    assert FORBIDDEN_TRACKED.fullmatch("data/processed/manifest.csv")
    assert FORBIDDEN_TRACKED.fullmatch("runs/model/best.pt")
    assert FORBIDDEN_TRACKED.fullmatch("runs/model/model.onnx.data")
    assert not FORBIDDEN_TRACKED.fullmatch("runs/model/results.json")
    assert not FORBIDDEN_TRACKED.fullmatch("docs/evidence/demo.mp4")
    assert set(EXPECTED_EVIDENCE) == {
        "docs/evidence/p23-live-showcase.mp4",
        "docs/evidence/p23-live-showcase-preview.jpg",
    }
