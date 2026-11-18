"""A card's state from the driver's files (a fake sysfs and /dev here), the open's
preflight that refuses a card not ready with CardNotReady, and the load's two causes of
a dispatcher that is not idle (no card needed)."""

from types import SimpleNamespace

import pytest
from fake_session import FakeDevice, load_session, program

import komorebi as km
from komorebi.device import mock, native
from komorebi.islands import session as island_session
from komorebi.session import process, sizes

LIVE = {"tt_card_type": "p150a", "tt_fw_bundle_ver": "19.13.1.0", "tt_aiclk": "800"}


def fake_card(tmp_path, card, files=None, node=True, beat="100"):
    """Card `card`'s sysfs entry below tmp_path/sys (its files, the heartbeat at
    `beat`, None: no heartbeat file) and its node in tmp_path/dev."""
    entry = tmp_path / "sys" / "class" / "tenstorrent" / f"tenstorrent!{card}"
    entry.mkdir(parents=True)
    for name, text in (LIVE if files is None else files).items():
        (entry / name).write_text(text + "\n")
    if beat is not None:
        (entry / "tt_heartbeat").write_text(beat + "\n")
    (tmp_path / "dev").mkdir(exist_ok=True)
    if node:
        (tmp_path / "dev" / str(card)).write_text("")
    return entry


@pytest.fixture
def driver(monkeypatch, tmp_path):
    """A fake driver: its waits advance the heartbeat of the cards listed in `live`,
    counted in `waits`; no chip lock, no mock, no card found ready before."""
    live, waits = [], []

    def wait(seconds):
        waits.append(seconds)
        for entry in live:
            beat = entry / "tt_heartbeat"
            beat.write_text(f"{int(beat.read_text()) + 3}\n")

    monkeypatch.setattr(process.time, "sleep", wait)
    monkeypatch.setattr(process, "CHIP_LOCK", str(tmp_path / "no-lock-{card}"))
    monkeypatch.setattr(process, "SYS", tmp_path / "sys")
    monkeypatch.setattr(process, "DEV", tmp_path / "dev")
    monkeypatch.setattr(process, "_READY", set())
    monkeypatch.delenv(mock.ACTIVE, raising=False)
    monkeypatch.delenv(mock.MOCK, raising=False)
    return SimpleNamespace(live=live, waits=waits, root=tmp_path)


def test_a_live_card_reads_ready(driver):
    driver.live.append(fake_card(driver.root, 0))
    state = process.card_states([0])[0]
    assert (state["type"], state["firmware"], state["aiclk"]) == (
        "p150a",
        "19.13.1.0",
        "800",
    )
    assert state["heartbeat"] == ("100", "103") and driver.waits == [0.3]
    assert state["entry"] and state["node_exists"] and state["missing"] == []
    assert state["holder"] is None and process.card_problems(state) == []


def test_a_frozen_heartbeat_is_named(driver):
    fake_card(driver.root, 0)
    state = process.card_states([0])[0]
    assert process.card_problems(state) == ["its heartbeat stayed at 100 over 0.3 s"]


def test_a_missing_file_is_reported_missing_never_guessed(driver):
    driver.live.append(fake_card(driver.root, 0, files={"tt_card_type": "p150a"}))
    state = process.card_states([0])[0]
    assert state["firmware"] is None and state["aiclk"] is None
    assert state["missing"] == ["tt_fw_bundle_ver", "tt_aiclk"]
    assert process.card_problems(state) == []  # the open needs none of them
    fake_card(driver.root, 1, beat=None)
    state = process.card_states([1])[1]
    assert state["missing"] == ["tt_heartbeat"]
    said = process.card_problems(state)
    assert said == ["its heartbeat (tt_heartbeat) could not be read"]


def test_a_missing_card_and_a_missing_node(driver):
    driver.live.append(fake_card(driver.root, 0, node=False))
    states = process.card_states([0, 3])
    assert driver.waits == [0.3]  # one wait for every card
    assert process.card_problems(states[0]) == [
        f"its device node {driver.root / 'dev' / '0'} is missing"
    ]
    assert not states[3]["entry"] and states[3]["missing"] == [
        "tt_card_type",
        "tt_fw_bundle_ver",
        "tt_aiclk",
        "tt_heartbeat",
    ]
    assert process.card_problems(states[3]) == [
        "the driver has no sysfs entry tenstorrent!3",
        f"its device node {driver.root / 'dev' / '3'} is missing",
    ]
    (driver.root / "dev" / "by-id").mkdir()
    (driver.root / "dev" / "5").write_text("")  # a node with no entry
    assert process.card_entries() == [0, 5]


def test_the_preflight_refuses_a_card_not_ready_and_waits_once_a_card(driver):
    fake_card(driver.root, 0)
    with pytest.raises(km.CardNotReady) as said:
        process.card_ready([0])
    message = str(said.value)
    assert message.startswith("card 0 is not ready: its heartbeat stayed at 100")
    assert "tt-smi -r 0" in message and said.value.card == 0
    assert said.value.state["heartbeat"] == ("100", "100")
    driver.live.append(fake_card(driver.root, 1))
    process.card_ready([1])
    process.card_ready([1])  # found ready once: no second wait
    assert len(driver.waits) == 2


def test_the_preflight_passes_the_stand_in_card(driver, monkeypatch):
    monkeypatch.setenv(mock.ACTIVE, "1")  # a prebuild: no driver files at all
    process.card_ready([0, 1])
    assert driver.waits == []


def test_both_opens_refuse_before_tt_metal(driver, monkeypatch):
    monkeypatch.setattr(process, "_OPENED", {"mesh": [], "alone": []})
    monkeypatch.setattr(process, "chip_holder", lambda card, path=None: None)
    monkeypatch.delenv("TT_VISIBLE_DEVICES", raising=False)

    def no_device(*args, **kwargs):
        raise AssertionError("the refusal must come before the device is touched")

    monkeypatch.setattr(native, "load_module", no_device)
    driver.live.append(fake_card(driver.root, 0))
    with pytest.raises(km.CardNotReady, match="card 1 is not ready: the driver has"):
        island_session.IslandSession(cards=[0, 1]).open()
    with pytest.raises(km.CardNotReady, match="card 1 is not ready"):
        km.Session(card=1).open()
    assert process._OPENED == {"mesh": [], "alone": []}  # refused before the record


def test_a_dispatcher_not_idle_names_its_cause(monkeypatch, tmp_path):
    session, log, _ = load_session(monkeypatch, tmp_path)
    sess = session.Session(device=FakeDevice(log), card=0)
    said = sess._not_idle()
    assert said.startswith("no live process holds card 0: an earlier process left")
    assert "tt-smi -r 0" in said
    monkeypatch.setattr(process, "chip_holder", lambda card, path=None: 4242)
    assert sess._not_idle().startswith("card 0 is held by process 4242 (")


def test_a_split_cards_dispatcher_not_idle_names_its_cause(monkeypatch, tmp_path):
    session, log, module = load_session(monkeypatch, tmp_path)
    made = module.Dispatch

    def busy(device, mhz):  # island 0's dispatcher runs: an earlier process left it
        dispatch = made(device, mhz)
        dispatch.idle = False
        return dispatch

    module.Dispatch = busy
    monkeypatch.setattr(process, "_hook_exit", lambda log=None: None)
    monkeypatch.setattr(sizes, "_split_rows", lambda graph, grid: [(0, 5), (5, 5)])
    device = FakeDevice(log)
    device.grid = lambda: (12, 10)
    sess = session.Session(device=device, card=0)
    with pytest.raises(km.StartError) as said:
        sess.load_islands(program(tmp_path), graph=None)
    assert str(said.value).startswith(
        "X280 cluster 0 is not idle; no live process holds card 0: an earlier process"
    )
    assert "tt-smi -r 0" in str(said.value)
