"""The feeder's clock: on a prebuild's stand-in card, with the card's files hidden,
`Fed` reads no clock and says so; on a real open it reads the card's."""

from komorebi.device import chips, mock
from komorebi.session import process
from komorebi.tools.feeder import fed


class Device:
    """The open device's part `Fed` asks for."""

    def worker_core(self, x, y):
        return x + 1, y + 2

    def arch(self):
        return "blackhole"

    def driver_device(self):
        return (1,)


class Session:
    grid = (2, 2)
    device = Device()


def hidden(**_):
    raise FileNotFoundError("/sys/class/tenstorrent/tenstorrent!1/tt_aiclk")


def test_on_the_stand_in_no_clock_is_read_and_the_log_says_so(monkeypatch, capsys):
    monkeypatch.setenv(mock.ACTIVE, "1")
    monkeypatch.setattr(process, "aiclk_mhz", hidden)  # as under off_card.sh
    runner = fed.Fed(Session())
    assert runner.aiclk is None
    assert runner.mhz == mock.mhz("blackhole")
    assert "aiclk not read" in capsys.readouterr().out


def test_on_a_real_open_the_cards_clock_is_read(monkeypatch):
    monkeypatch.delenv(mock.ACTIVE, raising=False)
    monkeypatch.setattr(process, "aiclk_mhz", lambda driver=None: 1350)
    runner = fed.Fed(Session())
    assert runner.aiclk == 1350
    assert runner.mhz == 1350 * chips.chip("blackhole").tick_ratio
