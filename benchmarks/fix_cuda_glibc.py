#!/usr/bin/env python3
"""
Reconcile CUDA's math headers with glibc >= 2.41.

THE CLASH
    glibc 2.41 added the C23 functions cospi/sinpi/tanpi (and float/long-double
    variants). In C++ glibc declares them `__THROW`, i.e. `noexcept (true)`.
    CUDA's crt/math_functions.h declares the same names WITHOUT an exception
    specification. Two declarations of one function that disagree on noexcept
    are ill-formed, so nvcc stops with

        /usr/include/bits/mathcalls.h(83): error: exception specification is
        incompatible with that of previous function "cospi"

    It is a header conflict, not a compiler-version problem. -ccbin cannot help
    (an older g++ reads the same /usr/include) and -D__GLIBC_USE_... cannot
    help either: glibc's features.h #undefs and redefines that macro itself, so
    a command-line define is overwritten before math.h is parsed.

THE FIX
    Add the missing `noexcept (true)` to CUDA's declarations so they agree with
    glibc. Minimal, local, and reversible.

HOW IT FINDS WHAT TO PATCH
    It does not carry a hardcoded list of function names - CUDA and glibc both
    move. It compiles a stub, reads the name out of nvcc's own error message,
    patches that one function, and repeats until the stub compiles. Whatever
    the pair happens to disagree about this month is what gets fixed.

USAGE
    python3 fix_cuda_glibc.py --check            # diagnose, change nothing
    python3 fix_cuda_glibc.py --dry-run          # show the exact diff
    python3 fix_cuda_glibc.py                    # patch (backs up first)
    python3 fix_cuda_glibc.py --revert           # restore the backup

    Add --cuda /usr/local/cuda-12.9 to pick a toolkit, and --arch sm_70 to
    match your GPU. Writing into /usr/local needs sudo; run --dry-run first,
    read the diff, then re-run under sudo.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

STUB = '#include <math.h>\n#include <cmath>\n__global__ void k(){}\n'
MARK = "/* patched for glibc C23 math (fix_cuda_glibc.py) */"

ERR = re.compile(
    r'exception specification is incompatible with that of previous function\s+"([A-Za-z_][A-Za-z0-9_]*)"')


def find_cuda(explicit):
    if explicit:
        return explicit
    if os.environ.get("CUDA_HOME"):
        return os.environ["CUDA_HOME"]
    nvcc = shutil.which("nvcc")
    if nvcc:
        return os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
    cands = sorted((d for d in os.listdir("/usr/local")
                    if d.startswith("cuda-") and "." in d), reverse=True)
    for c in cands:
        if os.path.exists(f"/usr/local/{c}/bin/nvcc"):
            return f"/usr/local/{c}"
    sys.exit("no CUDA toolkit found; pass --cuda")


def headers(cuda):
    """Every crt/math_functions.h under this toolkit. There are usually two:
    include/crt/... and targets/<arch>/include/crt/..., and on most installs
    the first is a symlink to the second - resolve so we patch the file once."""
    out, seen = [], set()
    for base in (os.path.join(cuda, "include"),
                 *[os.path.join(cuda, "targets", t, "include")
                   for t in (os.listdir(os.path.join(cuda, "targets"))
                             if os.path.isdir(os.path.join(cuda, "targets")) else [])]):
        p = os.path.join(base, "crt", "math_functions.h")
        if os.path.exists(p):
            r = os.path.realpath(p)
            if r not in seen:
                seen.add(r)
                out.append(r)
    return out


def query_arch():
    """pycuda usually lives in the benchmark venv, not in the system python
    this script may be running under (it needs no third-party imports of its
    own, and is often run with sudo). Ask each candidate interpreter."""
    snippet = ("import pycuda.autoinit, pycuda.driver as d;"
               "print('%d%d' % d.Device(0).compute_capability())")
    cands = [sys.executable]
    for v in (os.environ.get("VENV"), os.path.expanduser("~/venv-cudametro")):
        if v:
            cands.append(os.path.join(v, "bin", "python"))
    # under sudo, the user's venv is the interesting one
    sudo_user = os.environ.get("SUDO_USER")
    if sudo_user:
        cands.append(f"/home/{sudo_user}/venv-cudametro/bin/python")
    for py in cands:
        if not py or not os.path.exists(py):
            continue
        try:
            p = subprocess.run([py, "-c", snippet], capture_output=True,
                               text=True, timeout=60)
            out = (p.stdout or "").strip()
            if p.returncode == 0 and out.isdigit():
                return out
        except Exception:
            continue
    return None


def host_compilers():
    """Older g++ first: nvcc rejects host compilers newer than it knows."""
    import glob
    out = []
    for p in sorted(glob.glob("/usr/bin/g++-1?") + glob.glob("/usr/bin/g++1?"),
                    reverse=True):
        if os.access(p, os.X_OK):
            out.append(p)
    return out


def working_hostcc(cuda, arch):
    """Flags needed to get PAST nvcc's host-compiler version check.

    This is a separate gate from the glibc clash and it comes first: with the
    default gcc you never even reach the math headers. Returns (extra_flags,
    note) or (None, reason)."""
    ok, err = compile_stub(cuda, arch)
    if ok or "unsupported GNU version" not in err:
        return [], ""                    # default compiler is fine (or fails later)
    for cc in host_compilers():
        ok, err = compile_stub(cuda, arch, [f"-ccbin={cc}"])
        if ok or "unsupported GNU version" not in err:
            return [f"-ccbin={cc}"], cc
    return None, ("no installed g++ satisfies this toolkit's version check.\n"
                  "  Fedora: sudo dnf install gcc14-c++")


def supports_arch(cuda, arch):
    ok, err = compile_stub(cuda, arch)
    return ok or "Unsupported gpu architecture" not in err


def all_toolkits():
    out = []
    for d in sorted(os.listdir("/usr/local"), reverse=True):
        p = f"/usr/local/{d}"
        if d.startswith("cuda") and os.path.exists(f"{p}/bin/nvcc"):
            out.append(os.path.realpath(p))
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def compile_stub(cuda, arch, extra=()):
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "t.cu")
        open(src, "w").write(STUB)
        cmd = [os.path.join(cuda, "bin", "nvcc"), f"-arch=sm_{arch}", "-cubin",
               "-Wno-deprecated-gpu-targets", "-o", os.path.join(td, "t.cubin"),
               *extra, src]
        p = subprocess.run(cmd, capture_output=True, text=True)
        return p.returncode == 0, (p.stderr or "") + (p.stdout or "")


def patch_function(text, name):
    """Give every declaration of `name` in this header the same exception
    specification glibc uses. Returns (new_text, count).

    Only declarations are touched: a line that ends in `;` after the closing
    paren. Definitions (`{`) are left alone, as is anything already carrying a
    noexcept or throw specification."""
    # extern ... name ( args ) ;      possibly spanning lines
    pat = re.compile(
        r'((?:extern|static)[^;{}]*?\b' + re.escape(name) + r'\s*\([^;{}]*?\))\s*;',
        re.S)

    n = 0

    def sub(m):
        nonlocal n
        decl = m.group(1)
        if "noexcept" in decl or "throw" in decl or "__THROW" in decl:
            return m.group(0)
        n += 1
        return decl + " __CUDA_GLIBC_NOEXCEPT;"

    return pat.sub(sub, text), n


def ensure_macro(text):
    if "__CUDA_GLIBC_NOEXCEPT" in text and "#define __CUDA_GLIBC_NOEXCEPT" in text:
        return text
    shim = f"""{MARK}
#if !defined(__CUDA_GLIBC_NOEXCEPT)
#if defined(__cplusplus)
#define __CUDA_GLIBC_NOEXCEPT noexcept (true)
#else
#define __CUDA_GLIBC_NOEXCEPT
#endif
#endif

"""
    # Line 1, unconditionally. Placing it "after the include guard" is the bug
    # that produced `expected a "{"`: the first #define matching a guard-like
    # name can sit further down the file than the declarations being patched,
    # leaving the macro undefined exactly where it is used. The shim is an
    # idempotent #if !defined block, so it is harmless above the guard.
    return shim + text


def backup(path):
    b = path + ".pre-glibc-fix"
    if not os.path.exists(b):
        shutil.copy2(path, b)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", default=None)
    ap.add_argument("--arch", default=None, help="e.g. 70; default: ask the GPU")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--max-iter", type=int, default=40)
    args = ap.parse_args()

    cuda = find_cuda(args.cuda)
    hs = headers(cuda)
    if not hs:
        sys.exit(f"no crt/math_functions.h under {cuda}")
    print(f"toolkit : {cuda}")
    print(f"header  : {hs[0]}" + (f"  (+{len(hs)-1} more)" if len(hs) > 1 else ""))

    if args.revert:
        n = 0
        for h in hs:
            b = h + ".pre-glibc-fix"
            if os.path.exists(b):
                shutil.copy2(b, h)
                os.remove(b)
                print(f"reverted {h}")
                n += 1
        sys.exit(0 if n else "nothing to revert (no .pre-glibc-fix backup found)")

    arch = args.arch
    if not arch:
        arch = query_arch()
        if arch:
            print(f"(compute capability read from the device)")
        else:
            arch = "70"
            print("note: could not query the GPU; assuming sm_70 (--arch to override)")
    print(f"arch    : sm_{arch}")

    ok, err = compile_stub(cuda, arch)

    if not ok and "Unsupported gpu architecture" in err:
        # This toolkit is simply wrong for this card (CUDA 13 dropped Volta).
        # Patching its headers would be pointless, so find one that can build
        # for sm_<arch> before going any further.
        print(f"\n{os.path.basename(cuda)} cannot target sm_{arch}; looking for one that can")
        alt = next((t for t in all_toolkits()
                    if t != cuda and supports_arch(t, arch)), None)
        if not alt:
            sys.exit(f"no installed toolkit can target sm_{arch}.\n"
                     f"Installed: {' '.join(all_toolkits())}")
        print(f"  using {alt}")
        cuda = alt
        hs = headers(cuda)
        if not hs:
            sys.exit(f"no crt/math_functions.h under {cuda}")
        print(f"  header  : {hs[0]}")
        ok, err = compile_stub(cuda, arch)

    # Host-compiler gate BEFORE the header question: with a too-new gcc, nvcc
    # stops at host_config.h and never reaches crt/math_functions.h, so the
    # clash we are here to fix is invisible.
    HOSTCC, note = working_hostcc(cuda, arch)
    if HOSTCC is None:
        print(f"\nthis toolkit rejects every installed host compiler.\n  {note}")
        sys.exit(1)
    if HOSTCC:
        print(f"host cc : {note}   (gcc {os.popen('gcc -dumpversion').read().strip()} "
              f"is too new for this toolkit)")

    ok, err = compile_stub(cuda, arch, HOSTCC)
    if ok:
        print("\nnothing to do: the stub already compiles"
              + (f" (with {note})" if HOSTCC else "") + ".")
        if HOSTCC:
            print(f"  run.sh discovers this too; nothing further needed.")
        return
    if not ERR.search(err):
        print("\nthis is NOT the glibc C23 clash. nvcc says:\n")
        print("\n".join(err.strip().splitlines()[:8]))
        sys.exit(1)

    print("\nglibc C23 math clash confirmed. Functions nvcc objects to:")

    originals = {h: open(h).read() for h in hs}
    working = dict(originals)
    fixed = []

    for _ in range(args.max_iter):
        ok, err = compile_stub(cuda, arch, HOSTCC)
        if ok:
            break
        m = ERR.search(err)
        if not m:
            break
        name = m.group(1)
        if name in fixed:
            print(f"  {name}: patched but still reported - stopping to avoid a loop")
            break

        total = 0
        for h in hs:
            new, n = patch_function(working[h], name)
            if n:
                working[h] = ensure_macro(new)
                total += n
        if not total:
            print(f"  {name}: no declaration found to patch - stopping")
            break
        fixed.append(name)
        print(f"  {name}: {total} declaration(s)")

        if args.check or args.dry_run:
            # cannot iterate without writing, so stop after the first find
            break
        for h in hs:
            backup(h)
            open(h, "w").write(working[h])

    if args.check:
        print("\n--check: nothing written. Re-run without --check to patch.")
        return

    if args.dry_run:
        import difflib
        for h in hs:
            if working[h] != originals[h]:
                print(f"\n--- {h}\n+++ {h} (patched)")
                for line in difflib.unified_diff(
                        originals[h].splitlines(), working[h].splitlines(),
                        lineterm="", n=1):
                    if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                        print(line)
        print("\n--dry-run: nothing written. Only the FIRST function is shown;"
              "\nthe real run repeats until nvcc is satisfied.")
        return

    ok, err = compile_stub(cuda, arch, HOSTCC)
    print()
    if not ok and ERR.search(err) is None:
        # A new KIND of failure means our edit made things worse. Put the
        # headers back before reporting: a half-patched system header is a
        # worse outcome than the original clash.
        for h in hs:
            b = h + ".pre-glibc-fix"
            if os.path.exists(b):
                shutil.copy2(b, h)
                os.remove(b)
        print("the patch did not help and introduced a different error, so the")
        print("headers have been REVERTED automatically. nvcc said:\n")
        print("\n".join(err.strip().splitlines()[:8]))
        sys.exit(1)
    if ok:
        print(f"PATCHED: {', '.join(fixed)}")
        if HOSTCC:
            print(f"NOTE: this toolkit also needs {note} as the host compiler."
                  f"\n      run.sh finds that on its own and pins it for pycuda.")
        print("stub compiles. Backups at <header>.pre-glibc-fix "
              "(python3 fix_cuda_glibc.py --revert)")
    else:
        print("still failing. nvcc says:\n")
        print("\n".join(err.strip().splitlines()[:8]))
        print("\nRevert with --revert; this toolkit may simply be older than "
              "your glibc, in which case install CUDA 12.9u1+ or use a "
              "nvidia/cuda:12.x-devel container.")
        sys.exit(1)


if __name__ == "__main__":
    main()
