from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_api_reference_lists_all_public_modules():
    api_index = (ROOT / "docs" / "api" / "index.rst").read_text()
    modules = []

    for module_path in sorted((ROOT / "src" / "sptnet").rglob("*.py")):
        if module_path.name == "__init__.py":
            continue
        module_name = (
            module_path.relative_to(ROOT / "src")
            .with_suffix("")
            .as_posix()
            .replace("/", ".")
        )
        modules.append(module_name)

    missing = [module_name for module_name in modules if module_name not in api_index]
    assert missing == []
