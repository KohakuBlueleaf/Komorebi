"""The release workflows publish only on a tag or by hand, and the scripts they run."""

import importlib.util
import json
import re
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
GATE = "github.event_name == 'push' || inputs.publish"
PYPI = ("publish-pypi", "publish-pypi-runtime")
PUBLISH = (*PYPI, "publish-github")


def load(name):
    """A script of packaging/ci/ as a module (the folder is not a package)."""
    path = ROOT / "packaging" / "ci" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def job_texts(workflow):
    """{job: its lines of the file}: the jobs are the 2-space keys under `jobs:`."""
    jobs, current, inside = {}, None, False
    for line in workflow.read_text().splitlines():
        if re.match(r"^\S", line):
            inside = line.startswith("jobs:")
        elif inside and (found := re.match(r"^  ([\w-]+):\s*$", line)):
            current = jobs.setdefault(found.group(1), [])
        elif inside and current is not None:
            current.append(line)
    return {name: "\n".join(lines) for name, lines in jobs.items()}


def test_no_workflow_has_a_schedule():
    assert WORKFLOWS
    for workflow in WORKFLOWS:
        text = workflow.read_text()
        assert not re.search(r"(?m)^\s*(schedule|cron)\b", text), workflow.name


def test_the_publish_jobs_have_the_gate_and_no_secret():
    release = ROOT / ".github" / "workflows" / "release.yml"
    jobs = job_texts(release)
    assert set(PUBLISH) <= set(jobs)
    for name in PUBLISH:
        gates = re.findall(r"(?m)^    if: (.*)$", jobs[name])
        assert gates == [GATE], name
        assert "secrets." not in jobs[name], name
    assert all(f"{name}" in jobs["publish-github"] for name in PYPI)
    text = release.read_text()
    assert 'tags: ["v*"]' in text
    assert re.search(r"publish:\n(?:        .*\n)*?        default: false\n", text)
    for name in PYPI:
        assert "id-token: write" in jobs[name]
        assert "pypa/gh-action-pypi-publish@release/v1" in jobs[name]
        assert "packages-dir: upload/" in jobs[name]
    assert "name: pypi\n" in jobs["publish-pypi"]
    assert "name: pypi-runtime\n" in jobs["publish-pypi-runtime"]
    # Each job uploads one project's wheel; only the runtime's skips existing files.
    assert "outputs.wheel" in jobs["publish-pypi"]
    assert "runtime_wheel" not in jobs["publish-pypi"]
    assert "outputs.runtime_wheel" in jobs["publish-pypi-runtime"]
    assert "skip-existing: true" in jobs["publish-pypi-runtime"]
    assert "skip-existing" not in jobs["publish-pypi"]


def test_the_release_workflow_does_not_name_a_distribution():
    jobs = job_texts(ROOT / ".github" / "workflows" / "release.yml")
    code = "\n".join(re.sub(r"\s#.*", "", t) for t in jobs.values())
    assert not re.search(r"komorebi[-_](tt|runtime)", code)


def test_workflows_parse_and_agree_with_the_text():
    yaml = pytest.importorskip("yaml")
    for workflow in WORKFLOWS:
        document = yaml.safe_load(workflow.read_text())
        triggers = document.get("on", document.get(True))  # PyYAML reads `on` as True
        assert "schedule" not in triggers, workflow.name
        assert set(document["jobs"]) == set(job_texts(workflow)), workflow.name
    release = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
    for name in PUBLISH:
        job = release["jobs"][name]
        assert job["if"] == GATE
        assert "secrets." not in json.dumps(job)
    dispatch = (release.get("on") or release[True])["workflow_dispatch"]
    assert dispatch["inputs"]["publish"]["default"] is False
    environments = [release["jobs"][name]["environment"]["name"] for name in PYPI]
    assert environments == ["pypi", "pypi-runtime"]
    for name in PYPI:
        assert release["jobs"][name]["permissions"] == {"id-token": "write"}


def make_wheel(directory, name, version, requires=(), members=None):
    """A made-up wheel with the METADATA of a distribution and the `members`."""
    stem = f"{name.replace('-', '_')}-{version}"
    lines = ["Metadata-Version: 2.4", f"Name: {name}", f"Version: {version}"]
    lines += [f"Requires-Dist: {r}" for r in requires]
    wheel = Path(directory) / f"{stem}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"{stem}.dist-info/METADATA", "\n".join(lines) + "\n\n")
        for member, data in (members or {}).items():
            archive.writestr(member, data)
    return wheel


def test_path_check_finds_a_runner_path_in_a_binary_only(tmp_path):
    paths = load("check_paths")
    work = "/home/runner/work/proj"
    elf = b"\x7fELF\x02\x01" + b"\0" * 8 + f"{work}/src/x.cpp".encode() + b"\0"
    members = {
        "pkg/lib.so.1": elf,  # a binary naming the work directory
        "pkg/image.bin": b"at /home/runner/other\0",  # a binary naming the home
        "pkg/clean.so": b"\x7fELF" + b"/home/runnerx/elsewhere\0 /home/runner",
        "pkg/notes.txt": f"{work}/doc".encode(),  # text is not a binary
    }
    wheel = make_wheel(tmp_path, "made-up", "1.0", members=members)
    found, count = paths.scan(wheel, [work, "/home/runner"])
    assert count == 3
    assert sorted((m, p) for m, p, _ in found) == [
        ("pkg/clean.so", "/home/runner"),
        ("pkg/image.bin", "/home/runner"),
        ("pkg/lib.so.1", "/home/runner"),
        ("pkg/lib.so.1", work),
    ]
    assert paths.main([str(wheel), "--forbid", work]) == 1
    clean = make_wheel(tmp_path, "clean", "1.0", members={"p/a.so": b"\x7fELF.\0"})
    assert paths.main([str(clean), "--forbid", work]) == 0


def test_forbidden_paths_are_the_given_ones_and_the_runners():
    paths = load("check_paths")
    environ = {"GITHUB_WORKSPACE": "/w/space/", "KMR_RUNTIME_TREE": "", "OTHER": "/x"}
    found = paths.forbidden_paths(["/given", "/"], environ)
    assert "/given" in found and "/w/space" in found
    assert "" not in found and "/" not in found and "/x" not in found


def two_wheels(directory, requirement, runtime_version="0.81.1.2"):
    make_wheel(directory, "made-up-runtime", runtime_version)
    requires = [f'made-up-runtime{requirement}; extra == "runtime"', "numpy"]
    requires += ['other; extra == "dev"']
    return make_wheel(directory, "made-up-tt", "0.3.0", requires)


def test_wheel_info_reads_the_names_and_checks_the_versions(tmp_path):
    info = load("wheel_info")
    main = two_wheels(tmp_path, ">=0.81.1.1,<0.81.2")
    found = info.describe(tmp_path)
    assert (found["name"], found["version"]) == ("made-up-tt", "0.3.0")
    assert found["runtime_name"] == "made-up-runtime"
    assert found["wheel"] == main.name and found["prerelease"] == "false"
    assert info.check_versions(tmp_path, "v0.3.0")[1] == []
    assert info.main(["versions", str(tmp_path), "--tag", "v0.3.1"]) == 1
    assert [str(r) for r in info.requirements(main)] == ["numpy"]
    assert [str(r) for r in info.requirements(main, ["dev"])] == ["numpy", "other"]


def test_wheel_info_refuses_a_runtime_wheel_outside_the_range(tmp_path):
    info = load("wheel_info")
    two_wheels(tmp_path, ">=0.81.1.1,<0.81.2", runtime_version="0.81.2")
    assert "requires made-up-runtime" in info.check_versions(tmp_path)[1][0]


def test_wheel_info_wants_two_wheels_that_require_each_other(tmp_path):
    info = load("wheel_info")
    make_wheel(tmp_path, "one", "1.0")
    with pytest.raises(SystemExit, match="expected 2"):
        info.identify(tmp_path)
    make_wheel(tmp_path, "two", "1.0")
    with pytest.raises(SystemExit, match="requires the other"):
        info.identify(tmp_path)
