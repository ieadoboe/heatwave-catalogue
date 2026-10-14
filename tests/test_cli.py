"""Argument checks that must fail before any data is read.

    python tests/test_cli.py
"""
from hwcat import cli


def expect_error(fn, *args):
    try:
        fn(*args)
    except (ValueError, SystemExit):
        return
    raise AssertionError(f"{fn.__name__}{args} did not fail")


def test_parse_years():
    assert cli.parse_years(["1991-1993"]) == [1991, 1992, 1993]
    assert cli.parse_years(["2003"]) == [2003]
    assert cli.parse_years(["2003", "2022"]) == [2003, 2022]
    expect_error(cli.parse_years, ["2026-1991"])
    # The README quickstart lists 2003 inside the baseline range as well:
    # it must be fetched once, not twice.
    assert cli.parse_years(["1991-1993", "1992"]) == [1991, 1992, 1993]


def test_parse_region():
    assert cli.parse_region([72, -25, 35, 45]) == [72, -25, 35, 45]
    expect_error(cli.parse_region, [35, -25, 72, 45])     # north below south


def test_even_window_refused_before_reading():
    expect_error(cli.main, ["detect", "--year", "2003", "--window", "30",
                            "--workdir", "/nonexistent"])


def test_region_name_needs_its_box():
    """--region-name usa without --region would write Europe into usa files."""
    expect_error(cli.main, ["download", "--region-name", "usa", "--years", "2021",
                            "--workdir", "/nonexistent"])


if __name__ == "__main__":
    test_parse_years()
    test_parse_region()
    test_even_window_refused_before_reading()
    test_region_name_needs_its_box()
    print("ok")
