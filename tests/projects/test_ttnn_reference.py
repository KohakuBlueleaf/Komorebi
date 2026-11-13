"""The TT-NN reference benchmarks' loop sizing and summary, without a card."""

from projects import ttnn_reference as tr


def test_calls_for_sizes_a_loop_to_the_target_and_never_below_the_minimum():
    assert tr.calls_for(0.001) == 1000
    assert tr.calls_for(0.4, target_s=1.0) == 3
    assert tr.calls_for(84e-3) == 12


def test_summarize_takes_the_median_time_and_the_mean_power():
    loops = [
        {"us": u, "w": w, "peak_w": 2 * w, "temp_c": 60.0, "aiclk": c}
        for u, w, c in [
            (9.0, 100.0, [1350]),
            (5.0, 110.0, [800, 900]),
            (7.0, 120.0, []),
        ]
    ]
    got = tr.summarize(loops)
    assert got["us_per_call"] == 7.0 and got["us_per_loop"] == [9.0, 5.0, 7.0]
    assert got["mean_w"] == 110.0 and got["peak_w"] == 240.0
    assert got["aiclk_mhz"] == [800, 1016.7, 1350]


def test_summarize_keeps_a_loop_whose_power_was_not_measured():
    loops = [
        {"us": 4.0, "w": None, "peak_w": None, "temp_c": None, "aiclk": [1350]},
        {"us": 6.0, "w": 80.0, "peak_w": 90.0, "temp_c": 50.0, "aiclk": [1350]},
    ]
    got = tr.summarize(loops)
    assert got["mean_w"] == 80.0 and got["temp_c"] == 50.0
    assert got["w_per_loop"] == [None, 80.0]


def test_command_drops_the_output_path():
    argv = ["/a/b/bench_ttnn.py", "--shape", "x", "--json", "/c/d.json", "--loops", "3"]
    assert tr.command(argv) == ["bench_ttnn.py", "--shape", "x", "--loops", "3"]
