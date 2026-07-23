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


# --- Selectable blower with a broad operating band ---------------------------

def _blower():
    return fan.BlowerCurve(design_flow=0.012, design_pressure=29_000.0,
                           peak_efficiency=0.48)


def test_blower_head_falls_with_flow():
    b = _blower()
    assert b.pressure(0.0) > b.pressure(b.design_flow) > b.pressure(b.max_flow * 0.99)
    # Head at the design flow is the design pressure.
    assert b.pressure(b.design_flow) == pytest.approx(29_000.0, rel=1e-6)


def test_blower_efficiency_is_a_broad_hump():
    b = _blower()
    peak = b.efficiency(b.design_flow)
    assert peak == pytest.approx(0.48, rel=1e-6)
    # Efficiency stays well above half-peak across a wide flow range (broad band).
    assert b.efficiency(0.6 * b.design_flow) > 0.5 * peak
    assert b.efficiency(1.4 * b.design_flow) > 0.5 * peak


def test_blower_affinity_scaling():
    b = _blower()
    fast = b.at_speed(2.0)
    assert fast.design_flow == pytest.approx(2 * b.design_flow)      # Q ~ N
    assert fast.design_pressure == pytest.approx(4 * b.design_pressure)  # dp ~ N^2
    assert fast.peak_efficiency == b.peak_efficiency                 # eff preserved


def test_blower_operating_point_between_shutoff_and_free_delivery():
    b = _blower()
    op = b.operating_point(compression_pressure_pa=20_000.0)
    assert op["feasible"]
    assert 0.0 < op["flow_m3s"] < b.max_flow
    # At the operating flow the delivered head equals the compression demand.
    assert op["pressure_pa"] == pytest.approx(20_000.0, rel=0.01)


def test_blower_infeasible_when_demand_exceeds_shutoff():
    b = _blower()
    op = b.operating_point(compression_pressure_pa=b.pressure(0.0) * 1.1)
    assert op["feasible"] is False


def test_size_blower_for_duty_puts_bep_at_duty():
    b = fan.size_blower_for_duty(0.012, 29_000.0, 0.29)
    assert b.design_flow == 0.012
    assert b.pressure(0.012) == pytest.approx(29_000.0, rel=1e-6)
    assert 0.4 < b.peak_efficiency < 0.55
