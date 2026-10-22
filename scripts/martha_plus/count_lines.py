"""Print reproducible Python LOC metrics for implementation and tests."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def count(files: list[Path]) -> tuple[int, int]:
    lines = 0
    files_with_content = 0
    for path in files:
        if path.name == "__init__.py" and path.stat().st_size == 0:
            continue
        lines += len(path.read_text(encoding="utf-8").splitlines())
        files_with_content += 1
    return lines, files_with_content


def main() -> None:
    source_files = sorted((ROOT / "martha_plus").rglob("*.py"))
    test_files = sorted((ROOT / "tests").rglob("*.py"))
    source_lines, source_count = count(source_files)
    test_lines, test_count = count(test_files)
    total = source_lines + test_lines
    print(f"implementation_files={source_count}")
    print(f"implementation_lines={source_lines}")
    print(f"test_files={test_count}")
    print(f"test_lines={test_lines}")
    print(f"total_files={source_count + test_count}")
    print(f"total_python_lines={total}")
    print(f"equivalent_50_line_pages={(total + 49) // 50}")

if __name__ == "__main__":
    main()
