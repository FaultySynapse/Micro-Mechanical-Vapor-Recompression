"""Tests for the dimensionless fan/blower characterization."""

import pytest

from mvr import fan


def test_specific_speed_scales_with_speed():
    lo = fan.specific_speed(0.01, 20_000.0, 0.3, 3000.0)
    hi = fan.specific_speed(0.01, 20_000.0, 0.3, 6000.0)
    assert hi == pytest.approx(2 * lo, rel=1e-9)


def test_high_pressure_low_flow_is_low_specific_speed():
    # The MVR duty (high dp, low Q) sits at very low Ns -> blower territory.
    ns = fan.specific_speed(0.012, 29_000.0, 0.29, 3000.0)
    assert ns < 0.3
    assert "blower" in fan.recommended_machine(ns)


def test_machine_class_ordering():
    assert "blower" in fan.recommended_machine(0.1)
    assert fan.recommended_machine(0.5) == "centrifugal (backward-curved)"
    assert fan.recommended_machine(2.0) == "mixed-flow"
    assert fan.recommended_machine(5.0) == "axial"


def test_efficiency_peaks_in_midrange_and_floors_low():
    peak = fan.achievable_efficiency(1.5)
    low = fan.achievable_efficiency(0.01)
    assert peak > low
    assert peak == pytest.approx(0.86, abs=0.02)
    # Positive-displacement floor keeps very low Ns realistic (~0.48).
    assert low == pytest.approx(0.48, abs=0.02)


def test_characterize_duty_reports_blower_for_still():
    d = fan.characterize_duty(0.012, 29_000.0, 0.29, speed_rpm=3000.0, diameter_m=0.08)
    assert d["specific_speed"] < 0.1
    assert "blower" in d["machine"]
    assert 0.4 < d["overall_efficiency"] < 0.55
    assert d["electrical_power_w"] > 0
    assert "specific_diameter" in d


def test_offdesign_efficiency_drops_away_from_bep():
    peak = 0.5
    at_bep = fan.efficiency_offdesign(1.0, peak)
    off = fan.efficiency_offdesign(1.5, peak)
    assert at_bep == pytest.approx(peak)
    assert off < at_bep
