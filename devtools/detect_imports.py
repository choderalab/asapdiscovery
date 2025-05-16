"""
Usage:
# Full report with files
python devtools/detect_imports.py /path/to/directory -o output.txt

# Package names only
python devtools/detect_imports.py /path/to/directory -o output.txt -p
"""

import ast
import os
from typing import Dict, Set, List, Tuple
import click
from pathlib import Path


def get_imports_from_file(file_path: str) -> Set[Tuple[str, str]]:
    """
    Extract import names and their submodules from a Python file.
    Returns set of tuples (package, submodule).
    """
    imports = set()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    parts = name.name.split('.')
                    base_pkg = parts[0]
                    submodule = '.'.join(parts) if len(parts) > 1 else ''
                    imports.add((base_pkg, submodule))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    parts = node.module.split('.')
                    base_pkg = parts[0]
                    for name in node.names:
                        if node.module == base_pkg:
                            submodule = name.name
                        else:
                            submodule = f"{node.module}.{name.name}"
                        imports.add((base_pkg, submodule))

    except (SyntaxError, UnicodeDecodeError, FileNotFoundError) as e:
        print(f"Could not parse {file_path}: {str(e)}")

    return imports


def find_all_imports_with_files(directory: str) -> Dict[str, Dict[str, Set[str]]]:
    """
    Find all unique imports and their submodules in Python files.

    Returns
    -------
    Dict[str, Dict[str, Set[str]]]
        Dictionary mapping package names to dict of {submodule: set(files)}
    """
    import_to_files = {}

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, directory)
                file_imports = get_imports_from_file(file_path)

                for base_pkg, submodule in file_imports:
                    if base_pkg not in import_to_files:
                        import_to_files[base_pkg] = {'': set()}
                    if submodule not in import_to_files[base_pkg]:
                        import_to_files[base_pkg][submodule] = set()
                    import_to_files[base_pkg][submodule].add(rel_path)

    # Remove standard library modules
    std_libs = set()
    for pkg in import_to_files:
        try:
            if not hasattr(__import__(pkg), '__file__'):
                std_libs.add(pkg)
        except ImportError:
            continue

    for pkg in std_libs:
        import_to_files.pop(pkg)

    return import_to_files


@click.command()
@click.argument('directory',
                type=click.Path(exists=True, file_okay=False, dir_okay=True),
                default='.')
@click.option('--output', '-o',
              type=click.Path(dir_okay=False),
              default='import_report.txt',
              help='Output file path')
@click.option('--packages-only', '-p',
              is_flag=True,
              help='Show only package names without their importing files')
def main(directory: str, output: str, packages_only: bool):
    """Find all unique package imports and their importing files within the specified directory."""
    directory = str(Path(directory).resolve())
    imports_with_files = find_all_imports_with_files(directory)

    with open(output, 'w', encoding='utf-8') as f:
        if packages_only:
            f.write("Unique third-party packages and their submodules:\n")
            for pkg in sorted(imports_with_files):
                f.write(f"\n{pkg}:\n")
                submodules = sorted(m for m in imports_with_files[pkg].keys() if m)
                for submodule in submodules:
                    f.write(f"  - {submodule}\n")
        else:
            f.write("Unique third-party packages, their submodules and importing files:\n")
            for pkg in sorted(imports_with_files):
                f.write(f"\n{pkg}:\n")
                for submodule in sorted(imports_with_files[pkg]):
                    if submodule:
                        f.write(f"  {submodule}:\n")
                        for file in sorted(imports_with_files[pkg][submodule]):
                            f.write(f"    - {file}\n")
                    else:
                        for file in sorted(imports_with_files[pkg][submodule]):
                            f.write(f"  - {file}\n")

    print(f"Report written to: {output}")


if __name__ == "__main__":
    main()