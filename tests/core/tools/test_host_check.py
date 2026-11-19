"""host_check's log folders: the last KEEP runs are kept, and a failed run's (a
FAILED file in its folder) past them too."""

from komorebi.tools import host_check


def test_a_failed_run_s_logs_are_kept_past_the_last_runs(tmp_path):
    names = [f"20260101-0000{i:02d}" for i in range(12)]
    for name in names:
        (tmp_path / name).mkdir()
    (tmp_path / names[0] / host_check.FAILED).write_text("1 of 3 checks failed\n")
    new = host_check.run_folder(tmp_path)
    kept = sorted(p.name for p in tmp_path.iterdir())
    assert kept == [names[0], *names[3:], new.name]  # 10 passed or new, the failed one
