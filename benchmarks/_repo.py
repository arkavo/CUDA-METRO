"""Path bootstrap for the benchmark scripts.

Why this exists: the repo's own scripts must be run FROM src/cudametro/,
because `import construct` is a bare top-level import that only resolves when
that directory is the cwd. Copying benchmark scripts in there works, but then
two copies exist and one of them goes stale - which is exactly the failure an
automated runner will hit and not notice.

Importing this module instead makes the benchmarks runnable from anywhere:
it puts <repo>/src and <repo>/src/cudametro on sys.path and exposes the
repo-relative config directory. There is one copy of every script, living in
benchmarks/, and it is the one that runs.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)                      # benchmarks/ -> repo root
SRC = os.path.join(REPO, "src")
PKG = os.path.join(SRC, "cudametro")
CONFIGS = os.path.join(REPO, "configs")

for p in (PKG, SRC):                              # PKG first: `import construct`
    if p not in sys.path:
        sys.path.insert(0, p)

if not os.path.isdir(PKG):
    sys.exit(f"cannot find {PKG}\n"
             f"benchmarks/ must sit at the repo root, beside src/ and configs/.")


def config(name="bench.json"):
    """Absolute path to a file in <repo>/configs, whatever the cwd is."""
    return os.path.join(CONFIGS, name)


def inputs():
    """Absolute path to <repo>/inputs.

    construct.MonteCarlo defaults input_folder="../../inputs/", which only
    resolves when the cwd happens to be src/cudametro. Pass this instead."""
    return os.path.join(REPO, "inputs") + os.sep


def scratch():
    """A throwaway cwd for the benchmark.

    MonteCarlo.__init__ does os.mkdir("Output_<prefix>_<material>_<timestamp>")
    in the CURRENT directory - so every construction litters, and two
    constructions in the same second collide with FileExistsError. Run from
    here and the mess is contained and disposable."""
    d = os.path.join(HERE, ".run", "scratch")
    os.makedirs(d, exist_ok=True)
    return d
