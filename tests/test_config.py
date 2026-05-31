"""Pydantic validators must fail loud at load_config(), before any module is constructed."""
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from perception.config import AppConfig, load_config

DEFAULT_YAML = Path(__file__).resolve().parent.parent / "perception" / "configs" / "default.yaml"


def _load_default_dict() -> dict:
    with open(DEFAULT_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_default_yaml_loads_clean():
    cfg = load_config(DEFAULT_YAML)
    assert isinstance(cfg, AppConfig)
    assert cfg.steering.lookahead_m == 20.0
    assert cfg.classical.ema_alpha == 0.3
    assert "car" in cfg.objects.keep_classes


def test_classical_weights_must_sum_to_one():
    data = _load_default_dict()
    data["classical"]["heading_weight"] = 0.9  # 0.5 + 0.25 + 0.9 = 1.65 ≠ 1.0
    with pytest.raises(ValidationError) as excinfo:
        AppConfig.model_validate(data)
    assert "weights must sum to 1.0" in str(excinfo.value)


def test_out_of_range_conf_threshold_rejected():
    data = _load_default_dict()
    data["objects"]["conf_threshold"] = 1.5
    with pytest.raises(ValidationError):
        AppConfig.model_validate(data)


def test_extra_field_in_yaml_rejected():
    data = _load_default_dict()
    data["objects"]["mystery_knob"] = 42
    with pytest.raises(ValidationError) as excinfo:
        AppConfig.model_validate(data)
    assert "Extra inputs" in str(excinfo.value) or "extra_forbidden" in str(excinfo.value)


def test_max_angle_must_exceed_min():
    data = _load_default_dict()
    data["classical"]["min_angle_deg"] = 100.0
    data["classical"]["max_angle_deg"] = 80.0
    with pytest.raises(ValidationError) as excinfo:
        AppConfig.model_validate(data)
    assert "max_angle_deg" in str(excinfo.value)


def test_unknown_dataset_rejected():
    data = _load_default_dict()
    data["dl"]["dataset"] = "kitti"
    with pytest.raises(ValidationError):
        AppConfig.model_validate(data)
