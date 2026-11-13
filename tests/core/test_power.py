"""The power sampler: its arithmetic, a sampler against a fake sensor file, the card
mapping against a fake sysfs tree, nvidia-smi through a fake program, and the command.
"""

import stat
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from komorebi import cli, power


def sensor_card(tmp_path, watts=50.0, update_s=0.001, temp_c=60.0):
    """A Card whose hwmon files are in `tmp_path`."""
    hwmon = tmp_path / "hwmon"
    hwmon.mkdir(exist_ok=True)
    card = power.Card(0, "0000:00:00.0", hwmon, 150.0, update_s)
    (hwmon / "power1_input").write_text(str(int(watts * 1e6)))
    (hwmon / "temp1_input").write_text(str(int(temp_c * 1000)))
    return card


def test_integral_is_the_trapezoid_with_ends_interpolated_and_held():
    points = [(0.0, 10.0), (1.0, 10.0), (2.0, 20.0)]
    assert power.integral(points, 0.0, 2.0) == pytest.approx(25.0)
    assert power.integral(points, 0.5, 1.5) == pytest.approx(11.25)
    assert power.integral(points, -1.0, 0.0) == pytest.approx(10.0)
    assert power.integral(points, 2.0, 4.0) == pytest.approx(40.0)


def test_summarize_gives_mean_peak_joules_and_temperature():
    samples = [(0.0, 10.0, 40.0), (1.0, 10.0, 50.0), (2.0, 20.0, 60.0)]
    got = power.summarize(samples, 0.0, 2.0, update_s=0.1, limit_w=100.0)
    assert not got.too_short and got.samples == 3
    assert got.joules == pytest.approx(25.0)
    assert got.mean_w == pytest.approx(12.5)
    assert got.peak_w == 20.0
    assert got.mean_temp_c == pytest.approx(50.0)
    assert got.limit_w == 100.0 and got.seconds == 2.0


def test_a_section_shorter_than_a_few_updates_reports_no_numbers():
    samples = [(0.0, 10.0, None), (0.4, 10.0, None), (0.5, 10.0, None)]
    short = power.summarize(samples, 0.0, 0.4, update_s=0.1)
    assert short.too_short and short.mean_w is None and short.joules is None
    assert short.peak_w is None and "shorter than 5 sensor updates" in short.note
    assert not power.summarize(samples, 0.0, 0.5, update_s=0.1).too_short
    assert power.summarize([(5.0, 10.0, None)], 0.0, 4.0, update_s=0.1).samples == 0


def test_a_lost_sensor_covers_only_the_time_before_the_loss():
    samples = [(tenth / 10, 10.0, None) for tenth in range(0, 11)]
    got = power.summarize(samples, 0.0, 3.0, 0.1, lost_at=1.0)
    assert got.lost and got.mean_w == pytest.approx(10.0)
    assert got.joules == pytest.approx(10.0) and "lost after 1.00 s" in got.note
    early = power.summarize(samples, 0.0, 3.0, 0.1, lost_at=0.2)
    assert early.lost and early.too_short and "lost" in early.note


def test_sampler_reports_steady_power_per_phase(tmp_path):
    card = sensor_card(tmp_path, watts=80.0)
    with power.PowerSampler(card, period_s=0.002, idle_s=0) as sampler:
        sampler.mark("a")
        time.sleep(0.05)
        sampler.mark("b")
        time.sleep(0.05)
    got = sampler.result
    assert not got.too_short and got.limit_w == 150.0
    assert got.mean_w == pytest.approx(80.0) and got.peak_w == 80.0
    assert got.joules == pytest.approx(80.0 * got.seconds)
    assert got.mean_temp_c == pytest.approx(60.0)
    assert list(got.phases) == ["a", "b"]
    assert got.phases["b"].mean_w == pytest.approx(80.0)
    assert got.as_dict()["phases"]["a"]["seconds"] > 0


def test_a_sensor_that_is_not_there_is_a_lost_sensor(tmp_path):
    card = sensor_card(tmp_path)
    (card.hwmon / "power1_input").unlink()
    with power.PowerSampler(card, period_s=0.002, idle_s=0) as sampler:
        pass
    assert sampler.result.lost and sampler.result.too_short


def test_three_failed_reads_in_a_row_end_sampling_and_one_does_not(tmp_path):
    card = sensor_card(tmp_path)
    sampler = power.PowerSampler(card)
    real = card.read
    card.read = lambda: (_ for _ in ()).throw(power.SensorLost("gone"))
    sampler.take()
    card.read = real
    sampler.take()
    assert sampler.failures == 0 and not sampler.stop.is_set()
    card.read = lambda: (_ for _ in ()).throw(power.SensorLost("gone"))
    for _ in range(power.LOST_AFTER):
        sampler.take()
    assert sampler.stop.is_set()


def test_a_phase_name_can_be_marked_once_and_only_inside(tmp_path):
    card = sensor_card(tmp_path)
    sampler = power.PowerSampler(card, period_s=0.01, idle_s=0)
    with pytest.raises(RuntimeError):
        sampler.mark("early")
    with sampler:
        sampler.mark("fill")
        with pytest.raises(ValueError, match="fill"):
            sampler.mark("fill")
    with pytest.raises(RuntimeError):
        sampler.mark("late")


def make_pci_card(root, index, pci, hwmon_name, watts_uw, limit_uw=150_000_000):
    """The sysfs entries of one card: class link, device link, hwmon directory."""
    parent = root / "devices" / "pci0000:c0" / "0000:c0:01.1"
    device = parent / pci
    entry = device / "tenstorrent" / f"tenstorrent!{index}"
    hwmon = device / "hwmon" / hwmon_name
    entry.mkdir(parents=True)
    hwmon.mkdir(parents=True)
    (entry / "device").symlink_to(Path("../.."))
    (hwmon / "name").write_text("blackhole\n")
    (hwmon / "power1_input").write_text(str(watts_uw))
    (hwmon / "power1_max").write_text(str(limit_uw))
    classdir = root / "class" / "tenstorrent"
    classdir.mkdir(parents=True, exist_ok=True)
    (classdir / f"tenstorrent!{index}").symlink_to(entry)
    return hwmon


def test_cards_are_found_through_the_device_link_not_the_hwmon_number(tmp_path):
    make_pci_card(tmp_path, 0, "0000:c1:00.0", "hwmon4", 69_000_000)
    make_pci_card(tmp_path, 1, "0000:c2:00.0", "hwmon3", 19_000_000, 120_000_000)
    make_pci_card(tmp_path, 2, "0000:c3:00.0", "hwmon5", 1_000_000)
    other = tmp_path / "devices/pci0000:c0/0000:c0:01.1/0000:c3:00.0/hwmon/hwmon5/name"
    other.write_text("other\n")
    found = power.cards(tmp_path)
    assert [card.index for card in found] == [0, 1]
    assert [card.pci for card in found] == ["0000:c1:00.0", "0000:c2:00.0"]
    assert [card.hwmon.name for card in found] == ["hwmon4", "hwmon3"]
    assert [card.limit_w for card in found] == [150.0, 120.0]
    assert found[0].read()[0] == 69.0 and found[1].read()[0] == 19.0
    assert power.cards(tmp_path / "empty") == []


def test_a_card_index_resolves_through_the_sysfs_root(monkeypatch, tmp_path):
    make_pci_card(tmp_path, 1, "0000:c2:00.0", "hwmon3", 19_000_000)
    monkeypatch.setattr(power, "SYS", tmp_path)
    assert power.PowerSampler(1).source.index == 1
    with pytest.raises(LookupError, match="card 3"):
        power.PowerSampler(3)


FAKE_SMI = """#!/bin/sh
case "$*" in
  *index,power.limit*) printf '0, 300.00\\n1, 250.00\\n' ;;
  *"-i 1"*) printf '[N/A], 41\\n' ;;
  *power.draw*) printf '66.98, 35\\n' ;;
  *) exit 1 ;;
esac
"""


@pytest.fixture
def fake_smi(monkeypatch, tmp_path):
    path = tmp_path / "nvidia-smi"
    path.write_text(FAKE_SMI)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(power, "NVIDIA_SMI", str(path))
    return path


def test_gpus_and_their_readings_come_from_nvidia_smi(fake_smi, monkeypatch, tmp_path):
    found = power.gpus()
    assert [(gpu.index, gpu.limit_w) for gpu in found] == [(0, 300.0), (1, 250.0)]
    assert found[0].read() == (66.98, 35.0)
    assert found[1].read() == (None, 41.0)
    monkeypatch.setattr(power, "NVIDIA_SMI", "false")
    with pytest.raises(power.SensorLost):
        found[0].read()
    monkeypatch.setattr(power, "NVIDIA_SMI", str(tmp_path / "missing"))
    assert power.gpus() == []


def make_rapl(root, energy_uj, wrap=1_000_000):
    package = root / "class" / "powercap" / "intel-rapl:0"
    (package / "intel-rapl:0:0").mkdir(parents=True)
    (package / "name").write_text("package-0\n")
    (package / "energy_uj").write_text(str(energy_uj))
    (package / "max_energy_range_uj").write_text(str(wrap))
    (package / "intel-rapl:0:0" / "name").write_text("core\n")
    (package / "intel-rapl:0:0" / "energy_uj").write_text("1")
    return package


def test_cpu_power_is_the_energy_counter_over_time_and_wraps(monkeypatch, tmp_path):
    package = make_rapl(tmp_path, 900_000, wrap=1_000_000)
    clock = iter([10.0, 10.5])
    monkeypatch.setattr(power, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    cpu = power.cpu_package(tmp_path)
    assert cpu is not None and cpu.read() is None
    (package / "energy_uj").write_text("100000")
    assert cpu.read()[0] == pytest.approx(0.4)
    assert power.cpu_package(tmp_path / "empty") is None


def test_select_takes_every_card_unless_told_otherwise(monkeypatch, tmp_path):
    make_pci_card(tmp_path, 0, "0000:c1:00.0", "hwmon3", 69_000_000)
    make_pci_card(tmp_path, 1, "0000:c2:00.0", "hwmon4", 19_000_000)
    monkeypatch.setattr(power, "SYS", tmp_path)
    assert [card.index for card in power.select(None, None, False)[0]] == [0, 1]
    assert [card.index for card in power.select(1, None, False)[0]] == [1]


def run_power(capsys, *args):
    code = cli.main(["power", *args])
    return code, capsys.readouterr().out


def test_kmr_power_prints_a_reading_and_a_summary(monkeypatch, tmp_path, capsys):
    hwmon = make_pci_card(tmp_path, 0, "0000:c1:00.0", "hwmon3", 69_000_000)
    (hwmon / "temp1_input").write_text("67000")
    monkeypatch.setattr(power, "SYS", tmp_path)
    code, out = run_power(capsys, "--seconds", "0.1", "--period", "0.02")
    assert code == 0
    assert "card 0 (0000:c1:00.0): 69.0 W, 67.0 C" in out
    assert "card 0 (0000:c1:00.0): 0.1" in out


def test_kmr_power_says_so_when_nothing_is_available(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(power, "SYS", tmp_path)
    monkeypatch.setattr(power, "NVIDIA_SMI", str(tmp_path / "missing"))
    code, out = run_power(capsys, "--seconds", "0.1")
    assert code == 1 and "no Tenstorrent card with a power sensor found" in out
    code, out = run_power(capsys, "--gpu", "0", "--cpu", "--seconds", "0.1")
    assert code == 1
    assert "gpu 0: not available" in out and "cpu package: not available" in out
