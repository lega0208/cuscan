import hashlib
import sys
from pathlib import Path
from urllib.error import HTTPError

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_artifacts as release


@pytest.fixture
def files(tmp_path):
    paths = [tmp_path / name for name in ("a.whl", "b.whl", "c.tar.gz")]
    for path in paths:
        path.write_bytes(path.name.encode())
    return paths


def error(code):
    return HTTPError("https://example.org", code, "error", {}, None)


def metadata(**changes):
    value = {"id": 5, "tag_name": "v1.2.3", "draft": False, "prerelease": False, "immutable": False}
    value.update(changes)
    return value


def test_release_visibility_retry_and_sha(monkeypatch):
    responses = iter(
        [
            error(404),
            metadata(),
            {"ref": "refs/tags/v1.2.3", "object": {"type": "commit", "sha": "abc"}},
        ]
    )
    calls = []

    def request(url, token):
        calls.append(url)
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(release, "request_json", request)
    monkeypatch.setattr(release.time, "sleep", lambda seconds: None)
    assert release.release_metadata("owner/repo", "v1.2.3", "abc", "token")["id"] == 5
    assert len(calls) == 3


@pytest.mark.parametrize("status,attempts,expected", [(404, 3, 3), (403, 3, 1), (500, 3, 1)])
def test_release_only_retries_404(monkeypatch, status, attempts, expected):
    calls = []

    def request(*args):
        calls.append(args)
        raise error(status)

    monkeypatch.setattr(release, "request_json", request)
    monkeypatch.setattr(release.time, "sleep", lambda seconds: None)
    with pytest.raises(HTTPError):
        release.release_metadata("owner/repo", "v1.2.3", "abc", "token", attempts)
    assert len(calls) == expected


@pytest.mark.parametrize(
    "changes",
    [
        {"immutable": True},
        {"immutable": None},
        {"draft": True},
        {"prerelease": True},
        {"tag_name": "v1.2.4"},
        {"id": None},
    ],
)
def test_release_invalid_metadata_fails(monkeypatch, changes):
    monkeypatch.setattr(release, "request_json", lambda *args: metadata(**changes))
    with pytest.raises(ValueError):
        release.release_metadata("owner/repo", "v1.2.3", "abc", "token")


def test_release_sha_mismatch(monkeypatch):
    responses = iter(
        [metadata(), {"ref": "refs/tags/v1.2.3", "object": {"type": "commit", "sha": "other"}}]
    )
    monkeypatch.setattr(release, "request_json", lambda *args: next(responses))
    with pytest.raises(ValueError, match="commit"):
        release.release_metadata("owner/repo", "v1.2.3", "abc", "token")


def test_annotated_release_tag_is_peeled(monkeypatch):
    responses = iter(
        [
            metadata(),
            {"ref": "refs/tags/v1.2.3", "object": {"type": "tag", "sha": "tagsha"}},
            {"object": {"type": "commit", "sha": "abc"}},
        ]
    )
    monkeypatch.setattr(release, "request_json", lambda *args: next(responses))
    assert release.release_metadata("owner/repo", "v1.2.3", "abc", "token")["id"] == 5


def test_pypi_partial_and_identical(monkeypatch, files):
    entry = {"filename": files[0].name, "digests": {"sha256": release.digest(files[0])}}
    monkeypatch.setattr(release, "request_json", lambda *args: {"urls": [entry]})
    assert release.missing_pypi(files, "1.2.3") == files[1:]
    entry["digests"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="conflict"):
        release.missing_pypi(files, "1.2.3")


def test_pypi_missing_and_ambiguous(monkeypatch, files):
    def missing(*args):
        raise error(404)

    monkeypatch.setattr(release, "request_json", missing)
    assert release.missing_pypi(files, "1.2.3") == files
    monkeypatch.setattr(release, "request_json", lambda *args: {"urls": [{"filename": "unknown"}]})
    with pytest.raises(ValueError, match="ambiguous"):
        release.missing_pypi(files, "1.2.3")


def test_asset_identical_conflict_and_missing(monkeypatch, files):
    asset = {
        "name": files[0].name,
        "state": "uploaded",
        "digest": f"sha256:{release.digest(files[0])}",
    }
    monkeypatch.setattr(release, "request_json", lambda *args: [asset])
    assert release.missing_assets("owner/repo", 5, "token", files) == files[1:]
    asset["digest"] = "sha256:" + hashlib.sha256(b"different").hexdigest()
    with pytest.raises(ValueError, match="conflict"):
        release.missing_assets("owner/repo", 5, "token", files)


def test_distribution_count_and_act_guard(monkeypatch, tmp_path, files):
    assert release.distributions(tmp_path) == files
    files[0].unlink()
    with pytest.raises(ValueError, match="exactly"):
        release.distributions(tmp_path)
    monkeypatch.setenv("ACT", "true")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "release_artifacts",
            "upload",
            "--version",
            "1.2.3",
            "--tag",
            "v1.2.3",
            "--sha",
            "abc",
            "--repo",
            "owner/repo",
        ],
    )
    monkeypatch.setattr(release, "request_json", lambda *args: pytest.fail("API called under ACT"))
    with pytest.raises(SystemExit):
        release.main()


def test_prepare_copies_only_missing_and_upload_never_overwrites(monkeypatch, tmp_path, files):
    monkeypatch.delenv("ACT", raising=False)
    monkeypatch.setenv("GH_TOKEN", "token")
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(release, "release_metadata", lambda *args: metadata())
    monkeypatch.setattr(release, "missing_pypi", lambda *args: files[1:])
    pending = tmp_path / "pending"
    args = [
        "release_artifacts",
        "prepare",
        "--version",
        "1.2.3",
        "--tag",
        "v1.2.3",
        "--sha",
        "abc",
        "--repo",
        "owner/repo",
        "--dist",
        str(tmp_path),
        "--pending",
        str(pending),
    ]
    monkeypatch.setattr(sys, "argv", args)
    release.main()
    assert sorted(path.name for path in pending.iterdir()) == [path.name for path in files[1:]]
    assert output.read_text() == "publish_needed=true\n"

    calls = []
    monkeypatch.setattr(release, "distributions", lambda directory: files)
    monkeypatch.setattr(release, "missing_assets", lambda *args: files[1:])
    monkeypatch.setattr(release.subprocess, "run", lambda command, **kwargs: calls.append(command))
    args[1] = "upload"
    release.main()
    assert calls == [["gh", "release", "upload", "v1.2.3", *map(str, files[1:])]]
