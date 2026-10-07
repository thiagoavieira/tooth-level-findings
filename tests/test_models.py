"""Training, ensembling and attribution on random crops with randomly initialised backbones (CPU)."""

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from toothfindings.constants import CLASSES4  # noqa: E402
from toothfindings.evaluation.gradcam import target_conv_layer  # noqa: E402
from toothfindings.models.ensemble import Ensemble  # noqa: E402
from toothfindings.models.inference_cost import latency_ms, model_size_mb  # noqa: E402
from toothfindings.models.train import DataConfig, RunConfig, build_model, run  # noqa: E402
from toothfindings.utils import read_csv, read_json, write_csv  # noqa: E402

ARCH = "deit_small_patch16_224"


@pytest.fixture(scope="module")
def deit():
    return build_model(ARCH, 4, pretrained=False)


def random_crops(n, seed=0):
    rng = np.random.default_rng(seed)
    return [Image.fromarray(rng.integers(0, 256, (60 + i, 40, 3), dtype=np.uint8)) for i in range(n)]


def test_model_size_counts_parameters_and_buffers():
    bn = torch.nn.BatchNorm1d(10)  # 20 float parameters, 20 float buffers and an int64 counter
    assert model_size_mb(bn) == pytest.approx((20 * 4 + 20 * 4 + 8) / 1e6)


def test_latency_is_positive():
    assert latency_ms(torch.nn.Conv2d(3, 2, 1), torch.device("cpu"), size=4, iters=2, warmup=1) > 0


def test_gradcam_target_of_a_vit_is_the_patch_embedding(deit):
    # a ViT has no other Conv2d (so the DeiT-S Grad-CAM summary measures this layer)
    assert target_conv_layer(deit) is deit.patch_embed.proj


def test_gradcam_target_skips_a_1x1_classifier_head():
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 3), torch.nn.Conv2d(16, 8, 3), torch.nn.Conv2d(8, 3, 1))
    assert target_conv_layer(net) is net[1]


def test_ensemble_returns_per_seed_softmax():
    states = [build_model(ARCH, 4, pretrained=False).state_dict() for _ in range(2)]
    ens = Ensemble(None, ARCH, seeds=(0, 1), device="cpu", state_dicts=states)
    probs = ens.predict(random_crops(3), batch=2)
    assert probs.shape == (3, 2, 4) and probs.dtype == np.float32
    np.testing.assert_allclose(probs.sum(-1), 1.0, rtol=1e-5)
    assert not np.allclose(probs[:, 0], probs[:, 1])


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_one_test_mode_run_on_random_crops(tmp_path):
    splits, enhanced = tmp_path / "splits", tmp_path / "enhanced"
    rows = []
    for i, img in enumerate(random_crops(20)):
        c = CLASSES4[i % 4]
        name = f"lyria__P{i:04d}__b0.png"
        (enhanced / "none" / c).mkdir(parents=True, exist_ok=True)
        img.save(enhanced / "none" / c / name)
        rows.append([f"{c}/{name}", c, "lyria", f"P{i:04d}"])
    header = ["filepath", "class", "source", "patient_id"]
    write_csv(splits / "train_pool.csv", rows[:16], header)
    write_csv(splits / "test.csv", rows[16:], header)

    run_dir = tmp_path / "run"
    cfg = RunConfig(
        mode="test", enhancement="none", aug="aug", arch=ARCH, run_dir=str(run_dir), batch=8, epochs=1, workers=0
    )
    result = run(cfg, DataConfig(splits=splits, enhanced_root=enhanced, pretrained=False))

    assert result["test_metrics"]["per_class"].keys() == set(CLASSES4)
    assert read_json(run_dir / "metrics.json") == result
    assert read_json(run_dir / "status.json")["state"] == "done"
    used = read_csv(run_dir / "split_used.csv")
    assert sum(r["role"] == "val" for r in used) == 12  # max(3 per class, 10% of the pool)
    assert len(read_csv(run_dir / "predictions.csv")) == 4
    assert len(read_csv(run_dir / "training_log.csv")) == 1
    model = build_model(ARCH, 4, pretrained=False)
    model.load_state_dict(torch.load(run_dir / "model.pt"))
