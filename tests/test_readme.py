import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).parents[1]


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readme_graph_matches_code() -> None:
    """Fails when the graph changed but the README diagram was not regenerated."""
    draw_graph = load_script("draw_graph")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert draw_graph.update_readme(readme) == readme, (
        "README diagram is outdated: run `uv run python scripts/draw_graph.py`"
    )
