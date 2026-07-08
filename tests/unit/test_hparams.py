"""Unit tests for the hyperparameter configs — pydantic validation + the ModelCfg factory.

Equivalence classes: defaults; round-trip via dict (the config.json artifact); model_cfg maps the
arch fields with the given in_ch.
"""
from surfscan.training.hparams import HParams, ModelCfg


def test_defaults():
    hp = HParams()
    assert hp.model_type == "vae"
    assert hp.channels == ["xyz"]
    assert hp.size == 256 and hp.depth == 5


def test_roundtrip_via_dict():
    hp = HParams(model_type="inpaint", cats=["bagel"], epochs=3)
    hp2 = HParams(**hp.model_dump())
    assert hp2 == hp


def test_model_cfg_maps_arch_fields():
    hp = HParams(base=64, latent=128, size=128, depth=4, dropout=0.1)
    cfg = hp.model_cfg(in_ch=6)
    assert isinstance(cfg, ModelCfg)
    assert (cfg.in_ch, cfg.base, cfg.latent, cfg.size, cfg.depth, cfg.dropout) == (6, 64, 128, 128, 4, 0.1)


def test_model_cfg_defaults():
    c = ModelCfg()
    assert c.in_ch == 3 and c.base == 32 and c.depth == 5
