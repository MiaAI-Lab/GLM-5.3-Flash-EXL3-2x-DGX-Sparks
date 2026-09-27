import os
import pwd
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Every strict-mode entry point that derives the worker identity from $USER (#197).
# Each is sliced from the top through the LAST worker-identity line, matched by TEXT so the
# slice survives edits above it. Those regions are assignments only, so they are safe to run.
ENTRY_POINTS = [
    # rel path, anchor line, extra rank suffixes whose identity is derived from WORKER_USER
    ("start.sh", 'WORKER_SSH="${WORKER_SSH:-${WORKER_USER}@${WORKER_IP}}"', []),
    ("start-tp3.sh", 'WORKER2_SSH="${WORKER2_SSH:-${WORKER2_USER}@${WORKER2_IP}}"', ["2"]),
    ("start-tp4.sh", 'WORKER3_SSH="${WORKER3_SSH:-${WORKER3_USER}@${WORKER3_IP}}"', ["2", "3"]),
    ("scripts/spark_doctor.sh", 'WORKER_USER="${WORKER_USER:-$USER}"', None),
]
EFFECTIVE_ACCOUNT = pwd.getpwuid(os.geteuid()).pw_name


def _run_identity_prefix(rel, anchor, caller, env_file="", home=None):
    source = (ROOT / rel).read_text()
    assert anchor in source, f"{rel}: identity anchor is missing"
    head = source[: source.index(anchor) + len(anchor)]
    probe = "\n" + "".join(
        f'printf "{k}=%s\\n" "${{{k}-}}"\n'
        for k in ("WORKER_USER", "WORKER_HOME", "WORKER_SSH",
                  "WORKER2_USER", "WORKER2_HOME", "WORKER2_SSH",
                  "WORKER3_USER", "WORKER3_HOME", "WORKER3_SSH")
    )
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        script = tmp / rel                      # keep scripts/ layout for spark_doctor's SCRIPT_DIR
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(head + probe)
        script.chmod(0o755)
        (tmp / ".env").write_text(env_file)
        for overlay in (".env.tp3", ".env.tp4"):   # tp3/tp4 require these to exist; empty is fine
            (tmp / overlay).write_text("")
        # The TP4 launcher sources the ring helper before deriving worker identities.
        # Include it in the sliced fixture so this test reaches its intended assertions.
        if rel == "start-tp4.sh":
            helper = tmp / "files" / "switchless-ring.sh"
            helper.parent.mkdir(parents=True, exist_ok=True)
            helper.write_text((ROOT / "files" / "switchless-ring.sh").read_text())
        env = {"PATH": "/usr/bin:/bin", "HOME": home or str(tmp)}
        env.update(caller)
        return subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, timeout=30)


def _fields(out):
    return {l.split("=", 1)[0]: l.split("=", 1)[1] for l in out.splitlines() if "=" in l}


@pytest.mark.parametrize("rel,anchor,ranks", ENTRY_POINTS, ids=[e[0] for e in ENTRY_POINTS])
def test_worker_identity_resolves_without_USER_in_environment(rel, anchor, ranks):
    """#197: non-login invocations (cron, some service managers) omit USER. Under `set -u`
    an unguarded $USER aborts the script at the worker-identity block, before any preflight,
    container or transfer. The fallback must be the EFFECTIVE account, not merely non-empty."""
    home = "/srv/nonstandard-home"     # catches a fix that compares against an empty USER
    has_block = ranks is not None

    def check_same_account(f, user):
        assert f["WORKER_USER"] == user
        if has_block:
            assert f["WORKER_HOME"] == home, "same-account path must keep the caller's HOME"
            assert f["WORKER_SSH"] == f"{user}@10.0.0.2"
            for n, ip in zip(ranks, ("10.0.0.3", "10.0.0.4")):
                assert f[f"WORKER{n}_USER"] == user
                assert f[f"WORKER{n}_HOME"] == home
                assert f[f"WORKER{n}_SSH"] == f"{user}@{ip}"

    # the bug: no USER at all. Must not abort; must resolve to the effective account.
    r = _run_identity_prefix(rel, anchor, {}, home=home)
    assert "unbound variable" not in r.stderr, r.stderr
    assert r.returncode == 0, r.stderr
    check_same_account(_fields(r.stdout), EFFECTIVE_ACCOUNT)

    # USER set but empty: same outcome, never an empty account
    r = _run_identity_prefix(rel, anchor, {"USER": ""}, home=home)
    assert r.returncode == 0, r.stderr
    check_same_account(_fields(r.stdout), EFFECTIVE_ACCOUNT)

    # a stale LOGNAME must NOT be trusted over the effective account
    r = _run_identity_prefix(rel, anchor, {"LOGNAME": "stale-account"}, home=home)
    assert r.returncode == 0, r.stderr
    check_same_account(_fields(r.stdout), EFFECTIVE_ACCOUNT)

    # USER present: used verbatim
    r = _run_identity_prefix(rel, anchor, {"USER": "spark-op"}, home=home)
    assert r.returncode == 0, r.stderr
    check_same_account(_fields(r.stdout), "spark-op")

    # .env supplies the worker account (mixed-account kit), no USER present: .env wins,
    # and the mixed-account branch derives /home/<user> rather than the caller's HOME
    env_file = "WORKER_USER=otheruser\n" + "".join(f"WORKER{n}_USER=other{n}\n" for n in (ranks or []))
    r = _run_identity_prefix(rel, anchor, {}, env_file=env_file, home=home)
    assert r.returncode == 0, r.stderr
    f = _fields(r.stdout)
    assert f["WORKER_USER"] == "otheruser"
    if has_block:
        assert f["WORKER_HOME"] == "/home/otheruser"
        assert f["WORKER_SSH"] == "otheruser@10.0.0.2"
        for n, ip in zip(ranks, ("10.0.0.3", "10.0.0.4")):
            assert f[f"WORKER{n}_USER"] == f"other{n}"
            assert f[f"WORKER{n}_HOME"] == f"/home/other{n}"
            assert f[f"WORKER{n}_SSH"] == f"other{n}@{ip}"


if __name__ == "__main__":
    for rel, anchor, ranks in ENTRY_POINTS:
        test_worker_identity_resolves_without_USER_in_environment(rel, anchor, ranks)
    print("worker identity without USER OK")
