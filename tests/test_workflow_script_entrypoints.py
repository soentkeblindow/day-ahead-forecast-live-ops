"""spec 6.9 section 2.14 / step 16: an "Aufruf-Form-Test" (call-shape test)
for every script a GitHub Actions workflow invokes by path.

The existing test suite imports scripts as a package (``from scripts.foo
import ...``), which only works because pyproject.toml's
``[tool.pytest.ini_options] pythonpath = ["."]`` puts the repo root on
sys.path for the whole test session. A real ``python scripts/foo.py``
invocation (what every workflow actually runs) gets a different sys.path
shape: the script's own directory at the front, no repo root -- so a
script that needs ``scripts`` importable as a package (only
run_daily_submission.py, via its ``from scripts.measurement_a_...``
import) needs its own explicit bootstrap. Without it, this whole class
of bug is structurally invisible to the test suite (spec 6.9's own real
incident, 2026-09-28: this exact mismatch cost a full delivery day).

Each script is loaded in a fresh subprocess with the real call's sys.path
shape, under a module name other than ``__main__`` (so the
``if __name__ == "__main__":`` body never runs -- only import-time code
is exercised, no network/API/file-system side effects).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from energy_price_forecast.config import PROJECT_ROOT

WORKFLOWS_DIR = PROJECT_ROOT / ".github" / "workflows"

# Only a real `run:` step counts -- a few workflow comments mention a
# script by path without invoking it (e.g. submit.yml's own docstring-style
# comments about run_daily_submission.py), and must not be picked up here.
_RUN_STEP_SCRIPT = re.compile(r"^\s*run:\s+.*python scripts/(\w+\.py)", re.MULTILINE)

_LOADER_CODE = """
import importlib.util
import sys
from pathlib import Path

script_path = Path(sys.argv[1]).resolve()
cwd = str(Path.cwd())
# Reproduce `python scripts/<name>.py`'s own sys.path shape: the script's
# own directory at the front. Strip both `-c`'s default sys.path[0] ('',
# meaning "resolve against cwd at import time") and the literal cwd --
# neither is present for a real `python scripts/<name>.py` invocation, and
# leaving either in would let a `scripts.*` import silently succeed via the
# repo root instead of failing the way it would for real (the exact
# blind spot this test exists to close).
sys.path = [p for p in sys.path if p not in ("", cwd)]
sys.path.insert(0, str(script_path.parent))

spec = importlib.util.spec_from_file_location("_workflow_entrypoint_smoke", script_path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
# Must be registered before exec_module: @dataclass (used by several of
# these scripts) resolves its owning module via sys.modules[cls.__module__]
# while the class body executes, and fails on a bare AttributeError if the
# module isn't there yet.
sys.modules[spec.name] = module
spec.loader.exec_module(module)
"""


def _scripts_invoked_by_workflows() -> list[str]:
    names: set[str] = set()
    for yml in WORKFLOWS_DIR.glob("*.yml"):
        names.update(_RUN_STEP_SCRIPT.findall(yml.read_text(encoding="utf-8")))
    return sorted(names)


def _run_loader(script_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _LOADER_CODE, str(script_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


_INVOKED_SCRIPTS = _scripts_invoked_by_workflows()


def test_at_least_one_script_is_actually_discovered() -> None:
    # A regex that silently matches nothing would make every test below
    # vacuously pass -- guard against that.
    assert _INVOKED_SCRIPTS
    assert "run_daily_submission.py" in _INVOKED_SCRIPTS


@pytest.mark.parametrize("script_name", _INVOKED_SCRIPTS)
def test_workflow_script_imports_cleanly_with_the_real_call_shape(script_name: str) -> None:
    script_path = PROJECT_ROOT / "scripts" / script_name
    assert script_path.is_file(), f"{script_path} referenced by a workflow but missing"

    result = _run_loader(script_path)

    assert result.returncode == 0, (
        f"{script_name} failed to import under the real python scripts/{script_name} "
        f"sys.path shape:\n{result.stderr}"
    )


def test_negative_control_catches_a_removed_sys_path_bootstrap(tmp_path: Path) -> None:
    """Proves the harness above would actually have caught the 2026-09-28
    incident: with run_daily_submission.py's own sys.path bootstrap
    stripped out, the same loader must fail with a missing `scripts`
    package, not silently succeed."""
    original = (PROJECT_ROOT / "scripts" / "run_daily_submission.py").read_text(encoding="utf-8")
    bootstrap = "if str(PROJECT_ROOT) not in sys.path:\n    sys.path.insert(0, str(PROJECT_ROOT))\n"
    assert bootstrap in original, "bootstrap snippet drifted, update this test's literal copy"
    broken = original.replace(bootstrap, "")

    broken_script = tmp_path / "run_daily_submission.py"
    broken_script.write_text(broken, encoding="utf-8")

    result = _run_loader(broken_script)

    assert result.returncode != 0
    assert "scripts" in result.stderr
