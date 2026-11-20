"""Helpers for .github/workflows/jax-compat.yml.

latest_tested.txt pins the newest JAX validated against the native backend
built from JAX_GIT_TAG, and pyproject.toml caps jax just above it.

  untested  print, as a JSON list, the final jax releases that pyproject.toml
            excludes but the widened range would admit: the versions to test
  admitted  print, as a JSON list, every final jax release that pyproject.toml
            admits: the versions to re-test after a backend change
  widen     raise every jax upper bound in pyproject.toml to just above the
            version in latest_tested.txt
"""

import argparse
import json
import re
import sys
import tomllib
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# A jax requirement with an upper bound, e.g. "jax[cuda12]>=0.11.1,<0.11.3".
# Group 1 is everything up to the upper bound, group 2 the bound itself.
BOUNDED_JAX = re.compile(r"(\bjax(?:\[[\w-]+\])?>=[\w.]+,<)([\w.]+)")


def read_tested(path):
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"jax==([\w.]+)", line)
        if m is None:
            sys.exit(f"{path}: expected 'jax==X', got {line!r}")
        return m.group(1)
    sys.exit(f"{path}: no 'jax==X' line")


def release(version):
    # Final releases only; a pre-release raises, which is what we want here.
    return tuple(int(part) for part in version.split("."))


def next_release(version):
    """Smallest version above `version` that keeps its precision: 0.12.0 -> 0.12.1."""
    parts = version.split(".")
    parts[-1] = str(int(parts[-1]) + 1)
    return ".".join(parts)


def jax_spec(pyproject):
    """The specifier of the base `jax` dependency in pyproject.toml."""
    from packaging.requirements import Requirement

    deps = tomllib.loads(Path(pyproject).read_text())["project"]["dependencies"]
    (spec,) = [r.specifier for r in map(Requirement, deps) if r.name == "jax"]
    return spec


def final_releases():
    """Every final, non-yanked jax release on PyPI, oldest first."""
    from packaging.version import InvalidVersion, Version

    with urllib.request.urlopen("https://pypi.org/pypi/jax/json", timeout=60) as f:
        releases = json.load(f)["releases"]

    versions = []
    for name, files in releases.items():
        try:
            v = Version(name)
        except InvalidVersion:
            continue
        if v.is_prerelease or v.is_devrelease:
            continue
        if not files or all(f.get("yanked") for f in files):
            continue
        versions.append(v)
    return sorted(versions)


def untested(args):
    from packaging.version import Version

    tested = read_tested(args.tested)
    # Test everything `widen` would let in, not just up to the pin: Dependabot
    # can pin 0.12.0 while 0.12.0.1, which <0.12.1 also admits, is already out.
    upper = Version(next_release(tested))
    spec = jax_spec(args.pyproject)
    lower = max(Version(s.version) for s in spec if s.operator in (">=", ">"))

    versions = [v for v in final_releases() if lower < v < upper and v not in spec]
    if not versions:
        print(f"{spec} already admits every release below {upper}; nothing to test", file=sys.stderr)
    print(json.dumps([str(v) for v in versions]))


def admitted(args):
    spec = jax_spec(args.pyproject)
    versions = [v for v in final_releases() if v in spec]
    if not versions:
        sys.exit(f"no jax release on PyPI satisfies {spec}")
    print(json.dumps([str(v) for v in versions]))


def widen(args):
    tested = read_tested(args.tested)
    upper = next_release(tested)
    path = Path(args.pyproject)
    text = path.read_text()

    bounds = {m.group(2) for m in BOUNDED_JAX.finditer(text)}
    if not bounds:
        sys.exit(f"{path}: no bounded jax requirement found")
    if len(bounds) != 1:
        sys.exit(f"{path}: jax upper bounds disagree: {sorted(bounds)}")
    (old,) = bounds

    if release(upper) <= release(old):
        print(f"<{old} already covers jax=={tested}; nothing to widen")
        return
    text, n = BOUNDED_JAX.subn(rf"\g<1>{upper}", text)
    path.write_text(text)
    print(f"widened {n} jax requirements: <{old} -> <{upper}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["untested", "admitted", "widen"])
    parser.add_argument("--pyproject", default=REPO / "pyproject.toml")
    parser.add_argument("--tested", default=Path(__file__).with_name("latest_tested.txt"))
    args = parser.parse_args()
    {"untested": untested, "admitted": admitted, "widen": widen}[args.command](args)


if __name__ == "__main__":
    main()
