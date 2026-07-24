"""Tests for the vessel pressure-containment spec."""

from dataclasses import replace

import pytest

from mvr import DesignParameters, solve_evaporative, vessel_pressure_spec


def _spec_at(temp_C, lift=8.0, **kw):
    p = DesignParameters(evaporator_temp_C=temp_C, temp_lift=lift,
                         noncondensable_pressure=500.0, **kw)
    return vessel_pressure_spec(solve_evaporative(p))


def test_hot_unit_is_pressure_service_and_self_vents():
    # ~110 C runs both chambers above atmospheric: positive gauge, self-venting.
    s = _spec_at(110.0)
    assert s.service == "pressure"
    assert s.gauge_pressure_pa > 0.0
    assert s.self_venting is True
    assert s.max_abs_pressure_pa > s.min_abs_pressure_pa
    # Design pressure carries a margin over the operating gauge.
    assert s.design_gauge_pressure_pa >= s.gauge_pressure_pa + 1.0


def test_design_margin_rule():
    # Design gauge = max(+10%, +50 kPa) over operating gauge.
    s = _spec_at(115.0)
    expected = max(s.gauge_pressure_pa * 1.10, s.gauge_pressure_pa + 50_000.0)
    assert s.design_gauge_pressure_pa == pytest.approx(expected)


def test_cold_unit_is_vacuum_service():
    # A sub-atmospheric evaporator (~50 C) is vacuum service, not self-venting.
    s = _spec_at(50.0, lift=5.0)
    assert s.service == "vacuum"
    assert s.vacuum_gauge_pressure_pa > 0.0
    assert s.self_venting is False
    assert s.design_gauge_pressure_pa == 0.0


def test_saturation_temp_tracks_max_pressure():
    s = _spec_at(110.0)
    # T_sat at the condenser (max) pressure sits above the evaporator temp.
    assert s.saturation_temp_C > 110.0
