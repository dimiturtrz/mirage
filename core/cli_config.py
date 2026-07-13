"""Derive an argparse surface from a typed config dataclass — so a method declares ONE config schema
and gets its CLI for free (no hand-written `--flag` per field), while the same dataclass is what code
constructs directly. One config truth for CLI, programmatic calls, and the mlflow params log.

    @dataclass(frozen=True)
    class PatchCoreCfg:
        coreset: float = 0.1
        coreset_method: str = field(default="greedy", metadata={"choices": ["greedy", "random"]})
        size: int | None = None

    CliConfig.add_config_args(ap, PatchCoreCfg)      # --coreset --coreset-method --size, typed + defaulted
    cfg = CliConfig.build_config(PatchCoreCfg, args)  # parsed namespace -> the frozen config instance

Field type -> argument: bool -> BooleanOptionalAction (respects the default); list[...] -> nargs="*";
Optional[T]/T -> type=T. `metadata={"choices": [...], "help": "..."}` flows through. Types are resolved
via get_type_hints so `from __future__ import annotations` (string annotations) still works.
"""
from __future__ import annotations

import argparse
import types
import typing
from dataclasses import MISSING, fields


class CliConfig:
    """Argparse surface derived from a typed config dataclass (helpers as staticmethods, public names kept)."""

    @staticmethod
    def _resolve(t):
        """(base_type, is_list) — unwrap Optional[T]/`T | None` and list[T] to the scalar leaf type."""
        origin = typing.get_origin(t)
        if origin in (typing.Union, types.UnionType):
            non_none = [a for a in typing.get_args(t) if a is not type(None)]
            return CliConfig._resolve(non_none[0])
        if origin in (list, tuple):                 # typed containers only (list[str], not bare list)
            return typing.get_args(t)[0], True
        return t, False

    @staticmethod
    def _default(f):
        if f.default is not MISSING:
            return f.default
        if f.default_factory is not MISSING:   # list/tuple defaults
            return f.default_factory()
        return None

    @staticmethod
    def add_config_args(ap: argparse.ArgumentParser, cfg_cls) -> argparse.ArgumentParser:
        """Add one `--kebab-name` argument per dataclass field, typed + defaulted from the schema."""
        hints = typing.get_type_hints(cfg_cls)
        for f in fields(cfg_cls):
            base, is_list = CliConfig._resolve(hints[f.name])
            flag = "--" + f.name.replace("_", "-")
            default, choices, help_ = CliConfig._default(f), f.metadata.get("choices"), f.metadata.get("help")
            if base is bool:
                ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=default, help=help_)
            elif is_list:
                ap.add_argument(flag, nargs="*", type=base, default=default, choices=choices, help=help_)
            else:
                ap.add_argument(flag, type=base, default=default, choices=choices, help=help_)
        return ap

    @staticmethod
    def build_config(cfg_cls, args: argparse.Namespace):
        """Reconstruct the config instance from a parsed namespace (the inverse of add_config_args)."""
        return cfg_cls(**{f.name: getattr(args, f.name) for f in fields(cfg_cls)})
