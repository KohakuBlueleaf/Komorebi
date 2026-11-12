"""The power sampler: its arithmetic, a sampler against a fake sensor file, the card
mapping against a fake sysfs tree, nvidia-smi through a fake program, and the command.
"""

import os
import stat
import threading
import time
from pathlib import Path

import pytest

from komorebi import cli, power

# Seconds between sensor updates of the fake sensors, small so the tests are quick.
FAST = 0.01


def sensor_card(tmp_path, watts=50.0, update_s=FAST, temp_c=60.0):
    """A Card whose hwmon files are in `tmp_path`; write_watts changes the sensor."""
    hwmon = tmp_path / "hwmon"
    hwmon.mkdir(exist_ok=True)
    card = power.Card(0, "0000:00:00.0", hwmon, 150.0, update_s)
    write_watts(card, watts)
    (hwmon / "temp1_input").write_text(str(int(temp_c * 1000)))
    return card


def write_watts(card, watts):
    """Replaces the file in one step, as sysfs shows a value (never half written)."""
    new = card.hwmon / "new"
    new.write_text(str(int(watts * 1e6)))
    os.replace(new, card.hwmon / "power1_input")


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


def test_summarize_without_temperature_has_none():
    samples = [(0.0, 10.0, None), (1.0, 10.0, None)]
    assert power.summarize(samples, 0.0, 1.0, 0.1).mean_temp_c is None


def test_a_section_shorter_than_a_few_updates_reports_no_numbers():
    samples = [(0.0, 10.0, None), (0.4, 10.0, None), (0.5, 10.0, None)]
    short = power.summarize(samples, 0.0, 0.4, update_s=0.1)
    assert short.too_short and short.mean_w is None and short.joules is None
    assert short.peak_w is None and "shorter than 5 sensor updates" in short.note
    assert not power.summarize(samples, 0.0, 0.5, update_s=0.1).too_short


def test_a_section_with_no_sample_inside_is_too_short():
    got = power.summarize([(5.0, 10.0, None)], 0.0, 4.0, update_s=0.1)
    assert got.too_short and got.samples == 0


def test_a_lost_sensor_covers_only_the_time_before_the_loss():
    samples = [(t / 10, 10.0, None) for t in range(0, 11)]
    got = power.summarize(samples, 0.0, 3.0, 0.1, lost_at=1.0)
    assert got.lost and got.mean_w == pytest.approx(10.0)
    assert got.joules == pytest.approx(10.0) and "lost after 1.00 s" in got.note
    early = power.summarize(samples, 0.0, 3.0, 0.1, lost_at=0.2)
    assert early.lost and early.too_short and "lost" in early.note


def test_sampler_reports_mean_peak_joules_marks_and_idle(tmp_path):
    card = sensor_card(tmp_path, watts=40.0)
    with power.PowerSampler(card, period_s=0.002, idle_s=0.2) as sampler:
        sampler.mark("low")
        time.sleep(0.3)
        write_watts(card, 100.0)
        time.sleep(0.03)
        sampler.mark("high")
        time.sleep(0.3)
    got = sampler.result
    assert not got.too_short and got.samples > 100
    assert got.idle_mean_w == pytest.approx(40.0)
    assert got.peak_w == 100.0 and got.limit_w == 150.0
    assert got.mean_w == pytest.approx(got.joules / got.seconds, rel=0.02)
    assert 40.0 < got.mean_w < 100.0
    assert list(got.phases) == ["low", "high"]
    assert got.phases["low"].mean_w < 70.0
    assert got.phases["high"].mean_w == pytest.approx(100.0)
    assert got.phases["low"].seconds == pytest.approx(0.33, abs=0.1)
    assert got.mean_temp_c == pytest.approx(60.0)


def test_sampler_joules_are_watts_times_seconds_for_a_steady_load(tmp_path):
    card = sensor_card(tmp_path, watts=80.0)
    with power.PowerSampler(card, period_s=0.002, idle_s=0) as sampler:
        time.sleep(0.2)
    got = sampler.result
    assert got.idle_mean_w is None
    assert got.joules == pytest.approx(80.0 * got.seconds)
    assert got.mean_w == pytest.approx(80.0)


def test_sampler_notes_a_section_that_is_too_short(tmp_path):
    card = sensor_card(tmp_path, update_s=0.1)
    with power.PowerSampler(card, period_s=0.002, idle_s=0) as sampler:
        time.sleep(0.05)
    got = sampler.result
    assert got.too_short and got.mean_w is None and got.peak_w is None
    assert "shorter than 5 sensor updates (0.50 s)" in got.format()


def test_sampler_follows_a_value_changed_by_another_thread(tmp_path):
    card = sensor_card(tmp_path, watts=20.0)

    def raise_later():
        time.sleep(0.15)
        write_watts(card, 120.0)

    other = threading.Thread(target=raise_later)
    with power.PowerSampler(card, period_s=0.002, idle_s=0) as sampler:
        other.start()
        time.sleep(0.3)
    other.join()
    assert sampler.result.peak_w == 120.0
    assert 20.0 < sampler.result.mean_w < 120.0


def test_sampler_survives_a_sensor_that_disappears(tmp_path):
    card = sensor_card(tmp_path, watts=70.0)
    with power.PowerSampler(card, period_s=0.002, idle_s=0) as sampler:
        time.sleep(0.15)
        (card.hwmon / "power1_input").unlink()
        time.sleep(0.15)
    got = sampler.result
    assert got.lost and not got.too_short
    assert got.mean_w == pytest.approx(70.0)
    assert got.seconds == pytest.approx(0.3, abs=0.1)
    assert "sensor lost after" in got.note


def test_sampler_on_a_sensor_that_was_never_there(tmp_path):
    card = sensor_card(tmp_path)
    (card.hwmon / "power1_input").unlink()
    with power.PowerSampler(card, period_s=0.002, idle_s=0.05) as sampler:
        time.sleep(0.05)
    assert sampler.result.lost and sampler.result.too_short
    assert sampler.result.idle_mean_w is None


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


def test_one_failed_read_does_not_end_sampling(tmp_path):
    card = sensor_card(tmp_path, watts=70.0)
    reads = iter([None, 1, 2])
    real = card.read

    def flaky():
        if next(reads, 0) is None:
            raise power.SensorLost("once")
        return real()

    card.read = flaky
    with power.PowerSampler(card, period_s=0.002, idle_s=0) as sampler:
        time.sleep(0.1)
    assert not sampler.result.lost and sampler.result.mean_w == pytest.approx(70.0)


def test_a_sensor_that_fails_at_the_very_end_is_lost_there(tmp_path):
    card = sensor_card(tmp_path, watts=70.0)
    with power.PowerSampler(card, period_s=0.05, idle_s=0) as sampler:
        time.sleep(0.7)
        (card.hwmon / "power1_input").unlink()
    got = sampler.result
    assert got.lost and "sensor lost after 0." in got.note


def test_the_block_exception_is_not_swallowed_and_the_thread_stops(tmp_path):
    card = sensor_card(tmp_path)
    sampler = power.PowerSampler(card, period_s=0.002, idle_s=0)
    with pytest.raises(KeyError), sampler:
        raise KeyError("x")
    assert sampler.result is not None and not sampler.thread.is_alive()


def test_the_summary_is_plain_data(tmp_path):
    card = sensor_card(tmp_path)
    with power.PowerSampler(card, period_s=0.002, idle_s=0) as sampler:
        sampler.mark("a")
        time.sleep(0.1)
    data = sampler.result.as_dict()
    assert data["phases"]["a"]["seconds"] > 0
    assert set(data) >= {"seconds", "samples", "mean_w", "peak_w", "joules", "limit_w"}


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
    other = tmp_path / "devices" / "pci0000:00" / "0000:01:00.0" / "hwmon" / "hwmon1"
    other.mkdir(parents=True)
    (other / "name").write_text("nvme\n")
    found = power.cards(tmp_path)
    assert [c.index for c in found] == [0, 1]
    assert [c.pci for c in found] == ["0000:c1:00.0", "0000:c2:00.0"]
    assert [c.hwmon.name for c in found] == ["hwmon4", "hwmon3"]
    assert [c.limit_w for c in found] == [150.0, 120.0]
    assert found[0].read()[0] == 69.0 and found[1].read()[0] == 19.0


def test_a_card_without_a_blackhole_hwmon_is_left_out(tmp_path):
    make_pci_card(tmp_path, 0, "0000:c1:00.0", "hwmon0", 1_000_000)
    make_pci_card(tmp_path, 1, "0000:c2:00.0", "hwmon1", 1_000_000)
    (
        tmp_path / "devices/pci0000:c0/0000:c0:01.1/0000:c2:00.0/hwmon/hwmon1/name"
    ).write_text("other\n")
    assert [c.index for c in power.cards(tmp_path)] == [0]


def test_cards_on_a_host_without_any(tmp_path):
    assert power.cards(tmp_path) == []


def test_a_card_index_with_no_sensor_is_a_lookup_error(monkeypatch, tmp_path):
    monkeypatch.setattr(power, "SYS", tmp_path)
    with pytest.raises(LookupError, match="card 3"):
        power.PowerSampler(3)


def test_a_card_index_resolves_through_the_sysfs_root(monkeypatch, tmp_path):
    make_pci_card(tmp_path, 1, "0000:c2:00.0", "hwmon3", 19_000_000)
    monkeypatch.setattr(power, "SYS", tmp_path)
    assert power.PowerSampler(1).source.index == 1


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


def test_gpus_and_their_readings_come_from_nvidia_smi(fake_smi):
    found = power.gpus()
    assert [(g.index, g.limit_w) for g in found] == [(0, 300.0), (1, 250.0)]
    assert found[0].read() == (66.98, 35.0)
    assert found[1].read() == (None, 41.0)


def test_a_gpu_can_be_sampled_like_a_card(fake_smi):
    with power.PowerSampler(power.gpus()[0], period_s=0.02, idle_s=0) as sampler:
        time.sleep(0.8)
    got = sampler.result
    assert not got.too_short and got.mean_w == pytest.approx(66.98)
    assert got.limit_w == 300.0 and got.mean_temp_c == pytest.approx(35.0)


def test_without_nvidia_smi_there_are_no_gpus(monkeypatch, tmp_path):
    monkeypatch.setattr(power, "NVIDIA_SMI", str(tmp_path / "missing"))
    assert power.gpus() == []


def test_a_failing_nvidia_smi_is_a_lost_sensor(fake_smi, monkeypatch):
    gpu = power.Gpu(7, None)
    monkeypatch.setattr(power, "NVIDIA_SMI", "false")
    with pytest.raises(power.SensorLost):
        gpu.read()


def make_rapl(root, energy_uj, wrap=1_000_000):
    package = root / "class" / "powercap" / "intel-rapl:0"
    (package / "intel-rapl:0:0").mkdir(parents=True)
    (package / "name").write_text("package-0\n")
    (package / "energy_uj").write_text(str(energy_uj))
    (package / "max_energy_range_uj").write_text(str(wrap))
    (package / "intel-rapl:0:0" / "name").write_text("core\n")
    (package / "intel-rapl:0:0" / "energy_uj").write_text("1")
    return package


def test_cpu_power_is_the_energy_counter_over_time_and_wraps(tmp_path):
    package = make_rapl(tmp_path, 900_000, wrap=1_000_000)
    cpu = power.cpu_package(tmp_path)
    assert cpu is not None and cpu.read() is None
    time.sleep(0.05)
    (package / "energy_uj").write_text("100000")
    watts = cpu.read()[0]
    assert 200_000 / 1e6 / 0.5 < watts < 200_000 / 1e6 / 0.04


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read everything")
def test_an_unreadable_energy_counter_means_not_available(tmp_path):
    package = make_rapl(tmp_path, 5)
    (package / "energy_uj").chmod(0)
    assert power.cpu_package(tmp_path) is None


def test_a_host_without_powercap_has_no_cpu_package(tmp_path):
    assert power.cpu_package(tmp_path) is None


def run_power(capsys, *args):
    code = cli.main(["power", *args])
    return code, capsys.readouterr().out


def test_kmr_power_prints_readings_and_a_summary_per_card(
    monkeypatch, tmp_path, capsys
):
    hwmon = make_pci_card(tmp_path, 0, "0000:c1:00.0", "hwmon3", 69_000_000)
    make_pci_card(tmp_path, 1, "0000:c2:00.0", "hwmon4", 19_000_000)
    (hwmon / "temp1_input").write_text("67000")
    monkeypatch.setattr(power, "SYS", tmp_path)
    code, out = run_power(capsys, "--seconds", "1.2", "--period", "0.02")
    assert code == 0
    assert "card 0 (0000:c1:00.0): 69.0 W, 67.0 C" in out
    assert "card 1 (0000:c2:00.0): 19.0 W" in out
    assert "mean 69.0 W (46% of 150 W)" in out and "mean 19.0 W" in out
    code, out = run_power(capsys, "--card", "1", "--seconds", "1.2", "--period", "0.02")
    assert "card 0" not in out and "card 1" in out


def test_kmr_power_says_so_when_nothing_is_available(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(power, "SYS", tmp_path)
    monkeypatch.setattr(power, "NVIDIA_SMI", str(tmp_path / "missing"))
    code, out = run_power(capsys, "--seconds", "0.1")
    assert code == 1 and "no Tenstorrent card with a power sensor found" in out
    code, out = run_power(capsys, "--gpu", "0", "--cpu", "--seconds", "0.1")
    assert code == 1
    assert "gpu 0: not available" in out and "cpu package: not available" in out


def test_kmr_power_samples_a_gpu_and_the_cpu(monkeypatch, tmp_path, capsys, fake_smi):
    make_rapl(tmp_path, 0)
    monkeypatch.setattr(power, "SYS", tmp_path)
    args = ("--gpu", "0", "--cpu", "--seconds", "0.8", "--period", "0.05")
    code, out = run_power(capsys, *args)
    assert code == 0 and "card" not in out
    assert "gpu 0: 67.0 W, 35.0 C" in out
    assert "gpu 0: 0.8" in out and "cpu package: 0.8" in out
