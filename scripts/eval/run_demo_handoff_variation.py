"""Fresh-inode launcher resilient to Rosie's stale null-byte source cache.

Some compute nodes occasionally expose a stale cached inode containing NUL
bytes even though the login-node source is clean. Reading the source as bytes,
dropping only NULs, then compiling it preserves normal CLI behavior while
avoiding repeated destructive rewrites of a healthy file.
"""

from pathlib import Path

TARGET = Path(__file__).with_name("demo_handoff_variation.py")
source = TARGET.read_bytes().replace(b"\x00", b"")
namespace = {
    "__name__": "__main__",
    "__file__": str(TARGET),
    "__package__": None,
}
exec(compile(source, str(TARGET), "exec"), namespace)
