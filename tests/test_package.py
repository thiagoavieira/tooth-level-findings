import importlib
import pkgutil
from pathlib import Path

import pytest

import toothfindings
from toothfindings.cli import PRESETS, build_parser
from toothfindings.config import load_paths
from toothfindings.utils import read_json


def all_modules():
    return [m.name for m in pkgutil.walk_packages(toothfindings.__path__, "toothfindings.")]


@pytest.mark.parametrize("name", all_modules())
def test_module_imports_without_optional_dependencies(name):
    # heavy dependencies (torch, timm, cv2, onnxruntime, ultralytics) are imported lazily
    importlib.import_module(name)


def test_default_paths_are_inside_the_repository(paths):
    assert paths.results == paths.root / "results"
    assert paths.experiments == paths.root / "results" / "experiments"
    assert paths.lyria_json == paths.root / "data" / "lyria" / "json"
    assert (paths.experiments / "aggregates" / "cv_summary.csv").exists()


def test_overrides_and_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("TOOTHFINDINGS_DATA", str(tmp_path))
    p = load_paths(build="out")
    assert p.data == tmp_path
    assert p.crops == tmp_path / "crops"
    assert p.build == p.root / "out"
    with pytest.raises(ValueError):
        load_paths(nonsense="x")


def test_crop_path_resolution(paths):
    assert paths.crop_file("Impacted/lyria__P0001__b0.png") == paths.crops_source / "Impacted/lyria__P0001__b0.png"
    assert paths.crop_file("crops/1__t00.png", "tufts") == paths.external_work / "tufts" / "crops/1__t00.png"
    assert paths.crop_file("variants/x/a.png") == paths.crops / "variants/x/a.png"


@pytest.mark.parametrize(
    "argv",
    [
        ["stats", "all"],
        ["audit", "cascade"],
        ["train", "matrix", "--preset", "clean_5class"],
        ["eval", "prevalence", "--all"],
        ["external", "thresholds"],
    ],
)
def test_cli_parses(argv):
    a = build_parser().parse_args(argv)
    assert callable(a.func)


def _captured_run_configs(monkeypatch):
    captured = []
    monkeypatch.setattr("toothfindings.models.train.run", lambda cfg, data: captured.append(vars(cfg)))
    return captured


def test_train_run_reproduces_every_released_run_config(monkeypatch, paths):
    preset_of = {Path(p["exp"]): name for name, p in PRESETS.items()}
    captured = _captured_run_configs(monkeypatch)
    configs = sorted(paths.experiments.glob("**/run_config.yaml"))
    assert len(configs) == 552
    parser = build_parser()
    for f in configs:
        ref = read_json(f)
        exp = f.parent.parent.parent.relative_to(paths.experiments)
        argv = ["train", "run", "--preset", preset_of[exp]]
        argv += [x for k in ("mode", "fold", "enhancement", "aug", "arch", "seed") for x in (f"--{k}", str(ref[k]))]
        a = parser.parse_args(argv)
        a.func(paths, a)
        ours = captured.pop()
        assert {k: ours[k] for k in ref if k != "run_dir"} == {k: v for k, v in ref.items() if k != "run_dir"}, f


def test_train_run_keeps_explicit_batch_and_workers(monkeypatch, paths):
    captured = _captured_run_configs(monkeypatch)
    a = build_parser().parse_args(["train", "run", "--mode", "test", "--batch", "16", "--workers", "0"])
    a.func(paths, a)
    assert (captured[0]["batch"], captured[0]["workers"]) == (16, 0)


def test_subcommand_options_do_not_shadow_the_config_file():
    a = build_parser().parse_args(["--config", "my_paths.toml", "eval", "gradcam"])
    assert (a.config, a.run_config) == ("my_paths.toml", "none__noaug__inception_v3")
