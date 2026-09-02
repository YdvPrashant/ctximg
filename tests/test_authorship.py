"""Attribution, and the proof that backs it.

The proof has to be worth something: reversible digests, a fixed salt, or a
digest that travels between projects would all make it decorative.
"""

from __future__ import annotations

import pytest

import ctximg
from ctximg import authorship

FAST = 1_000  # PBKDF2 rounds; the shipped default is deliberately expensive


def commit(secret, **kw):
    return authorship.make_commitment(secret, iterations=FAST, **kw)


# --- plain attribution ----------------------------------------------------


def test_the_package_says_who_wrote_it():
    assert ctximg.__author__ == "prash"
    assert authorship.AUTHOR == "prash"


def test_pyproject_carries_the_same_name():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8")
    assert 'authors = [{ name = "prash" }]' in text


# --- the commitment -------------------------------------------------------


def test_a_secret_verifies_against_its_own_commitment():
    stored = commit("correct horse battery staple")
    assert authorship.verify("correct horse battery staple", stored) is True


def test_a_wrong_secret_does_not_verify():
    stored = commit("the real one")
    assert authorship.verify("the wrong one", stored) is False
    assert authorship.verify("", stored) is False


def test_the_commitment_does_not_contain_the_secret():
    """The whole point: the file can be public and still give nothing away."""
    secret = "moonlight-sonata-42"
    stored = commit(secret)
    assert secret not in stored
    assert secret.encode().hex() not in stored


def test_each_commitment_gets_its_own_salt():
    """A fixed salt would let one rainbow table break every copy."""
    first, second = commit("same secret"), commit("same secret")
    assert first != second
    assert authorship.verify("same secret", first)
    assert authorship.verify("same secret", second)


def test_the_stored_form_records_how_it_was_made():
    stored = commit("x")
    scheme, iterations, salt, digest = stored.split("$")
    assert scheme == "pbkdf2_sha256"
    assert int(iterations) == FAST
    assert len(bytes.fromhex(salt)) == 16
    assert len(bytes.fromhex(digest)) == 32


def test_the_shipped_cost_is_high_enough_to_matter():
    """The author's name is public, so a weak secret must still be expensive."""
    assert authorship.ITERATIONS >= 100_000


def test_an_empty_secret_is_refused():
    with pytest.raises(ValueError):
        commit("")


# --- tampering ------------------------------------------------------------


def test_a_edited_digest_stops_verifying():
    stored = commit("secret")
    scheme, iterations, salt, digest = stored.split("$")
    flipped = digest[:-1] + ("0" if digest[-1] != "0" else "1")
    assert authorship.verify("secret", f"{scheme}${iterations}${salt}${flipped}") is False


def test_a_swapped_salt_stops_verifying():
    stored = commit("secret")
    scheme, iterations, _, digest = stored.split("$")
    other = commit("secret").split("$")[2]
    assert authorship.verify("secret", f"{scheme}${iterations}${other}${digest}") is False


def test_rubbish_in_the_field_is_refused_not_crashed():
    for junk in ("", "nonsense", "a$b$c$d", "pbkdf2_sha256$x$y$z", "md5$1$aa$bb"):
        assert authorship.verify("secret", junk) is False


def test_a_commitment_does_not_travel_to_another_project(monkeypatch):
    """Binding in the project and author stops a digest being lifted wholesale."""
    stored = commit("secret")
    monkeypatch.setattr(authorship, "PROJECT", "someone-elses-app")
    assert authorship.verify("secret", stored) is False


def test_a_commitment_does_not_survive_renaming_the_author(monkeypatch):
    stored = commit("secret")
    monkeypatch.setattr(authorship, "AUTHOR", "someone-else")
    assert authorship.verify("secret", stored) is False


# --- recording it ---------------------------------------------------------


def test_recording_writes_the_line_back_into_the_module(tmp_path):
    module = tmp_path / "authorship.py"
    module.write_text('AUTHOR = "prash"\nCOMMITMENT = ""\n', encoding="utf-8")

    stored = commit("secret")
    authorship.record(stored, module)

    text = module.read_text(encoding="utf-8")
    assert f'COMMITMENT = "{stored}"' in text
    assert list(tmp_path.glob("*.tmp")) == [], "the write must be atomic"


def test_recording_replaces_an_earlier_proof(tmp_path):
    module = tmp_path / "authorship.py"
    module.write_text('COMMITMENT = "old"\n', encoding="utf-8")
    authorship.record("new", module)
    text = module.read_text(encoding="utf-8")
    assert 'COMMITMENT = "new"' in text
    assert "old" not in text


def test_recording_refuses_a_file_with_nowhere_to_put_it(tmp_path):
    module = tmp_path / "authorship.py"
    module.write_text("nothing here\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Could not find"):
        authorship.record("x", module)


def test_is_set_reflects_whether_a_proof_exists():
    assert authorship.is_set("") is False
    assert authorship.is_set(commit("x")) is True
