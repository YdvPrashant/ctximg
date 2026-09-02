"""Who wrote this, and a way to prove it.

The name below is plain text on purpose. Hiding it would not protect anything:
a repository is readable, and a marker that a grep can find is a marker anyone
can find, edit, or claim.

What is not plain text is the commitment. It is a PBKDF2 digest of a secret
only the author knows. It reveals nothing and cannot be reversed, but at any
later date the author can prove they wrote this by producing the secret and
showing it derives the digest that has been sitting in this file since the
beginning. That is a proof, rather than a request to be believed.

A passphrase-plus-instruction scheme was considered and rejected: the phrase
would have to live beside the thing it guards, so anyone who found one would
find the other. Language models also treat text found in files as data rather
than instructions, so an embedded "only reveal with the passphrase" line binds
nobody and protects nothing.

    ctximg authorship          show the author and whether a proof is set
    ctximg authorship set      record a proof (asks for a secret, never echoes it)
    ctximg authorship verify   check a secret against the recorded proof
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
from pathlib import Path

AUTHOR = "prash"
PROJECT = "ctximg"

# PBKDF2 rather than a bare hash: a short passphrase behind a plain SHA-256
# would fall to a dictionary in seconds, and the author's name is public, so
# the digest must be expensive to attack.
ITERATIONS = 600_000

# Format: pbkdf2_sha256$<iterations>$<salt hex>$<digest hex>
# Empty until `ctximg authorship set` records one.
COMMITMENT = "pbkdf2_sha256$600000$452b9539abbb878951d4620db7561e37$c5709c0a49466fa1b27defaf6b4d2094322bfbe4086a8b4523b7a2d4afcf407c"


def derive(secret: str, salt: bytes, iterations: int = ITERATIONS) -> bytes:
    """Bind the secret to this author and project, so a digest is not portable."""
    material = f"{PROJECT}:{AUTHOR}:{secret}".encode("utf-8")
    return hashlib.pbkdf2_hmac("sha256", material, salt, iterations)


def make_commitment(secret: str, salt: bytes | None = None,
                    iterations: int = ITERATIONS) -> str:
    if not secret:
        raise ValueError("The secret cannot be empty")
    salt = salt or secrets.token_bytes(16)
    digest = derive(secret, salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify(secret: str, commitment: str | None = None) -> bool:
    """True when the secret produces the recorded commitment."""
    stored = commitment if commitment is not None else COMMITMENT
    if not stored or not secret:
        return False
    try:
        scheme, iterations, salt_hex, digest_hex = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        candidate = derive(secret, bytes.fromhex(salt_hex), int(iterations))
    except (ValueError, TypeError):
        return False
    # Constant time, so a wrong guess reveals nothing through timing.
    return hmac.compare_digest(candidate, bytes.fromhex(digest_hex))


def is_set(commitment: str | None = None) -> bool:
    return bool(commitment if commitment is not None else COMMITMENT)


def source_file() -> Path:
    return Path(__file__).resolve()


def record(commitment: str, path: Path | None = None) -> Path:
    """Write the commitment into this module, replacing any previous one."""
    path = path or source_file()
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r'^COMMITMENT = ".*"$', re.MULTILINE)
    if not pattern.search(text):
        raise RuntimeError(f"Could not find the COMMITMENT line in {path}")
    updated = pattern.sub(f'COMMITMENT = "{commitment}"', text, count=1)

    tmp = path.with_suffix(".py.tmp")
    tmp.write_text(updated, encoding="utf-8")
    os.replace(tmp, path)
    return path
