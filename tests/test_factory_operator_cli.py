import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "infra" / "isaac-sim" / "factory-demo.sh"


def test_operator_help_documents_modes_and_evidence_boundary():
    result = subprocess.run(["bash", str(SCRIPT), "help"], text=True, capture_output=True, check=True)
    assert "showcase" in result.stdout
    assert "model" in result.stdout
    assert "recover [MODE]" in result.stdout
    assert "not classifier accuracy" in result.stdout


def test_operator_cli_rejects_unknown_action_before_running_tools():
    result = subprocess.run(["bash", str(SCRIPT), "launch-shell"], text=True, capture_output=True)
    assert result.returncode == 2
    assert "unknown action" in result.stderr


def test_operator_cli_has_no_dynamic_shell_execution():
    source = SCRIPT.read_text()
    assert "eval " not in source
    assert "bash -c" not in source
    assert "sh -c" not in source
