"""Config-schema -> argparse -> config roundtrip — a method declares one dataclass, the CLI derives.

Covers the type map (float/int/str, Optional[int], list, bool via BooleanOptionalAction), kebab flag
naming (coreset_method -> --coreset-method), choices enforcement, and the add/build inverse.
"""
import argparse
from dataclasses import dataclass, field

import pytest

from core.cli_config import add_config_args, build_config


@dataclass(frozen=True)
class Cfg:
    coreset: float = 0.1
    coreset_method: str = field(default="greedy", metadata={"choices": ["greedy", "random"]})
    size: int | None = None
    channels: list[str] = field(default_factory=lambda: ["rgb"])
    arms: list[str] = field(default_factory=list)
    curriculum: bool = False


def _parse(argv):
    ap = argparse.ArgumentParser()
    add_config_args(ap, Cfg)
    return ap.parse_args(argv)


def test_defaults_roundtrip_to_the_dataclass():
    assert build_config(Cfg, _parse([])) == Cfg()


def test_typed_overrides():
    cfg = build_config(Cfg, _parse(
        ["--coreset", "0.3", "--size", "512", "--channels", "rgb", "xyz",
         "--arms", "real", "synth", "--coreset-method", "random"]))
    assert cfg.coreset == 0.3 and cfg.size == 512
    assert cfg.channels == ["rgb", "xyz"] and cfg.coreset_method == "random"
    assert cfg.arms == ["real", "synth"]           # list[str] field parses variadic


def test_optional_int_defaults_none():
    assert build_config(Cfg, _parse([])).size is None


def test_bool_flag_and_negation():
    assert build_config(Cfg, _parse([])).curriculum is False
    assert build_config(Cfg, _parse(["--curriculum"])).curriculum is True
    assert build_config(Cfg, _parse(["--no-curriculum"])).curriculum is False


def test_choices_enforced():
    with pytest.raises(SystemExit):
        _parse(["--coreset-method", "bogus"])


@dataclass
class ReqCfg:
    name: str            # no default -> the field defaults to None on the CLI (optional)


def test_field_without_default_is_none():
    ap = argparse.ArgumentParser()
    add_config_args(ap, ReqCfg)
    assert ap.parse_args([]).name is None
