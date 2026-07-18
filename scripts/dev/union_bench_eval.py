"""Union-bench driver for baselines whose architecture differs from our harness.

Reuses the FROZEN harness (`scripts/eval_multinorm_audit.py`) verbatim for config
load+validation, subset load, the 12 AutoAttack components (run_attack_mask), result
assembly (build_result) and mask sidecar (export_mask_sidecar). ONLY model construction
differs: baselines (robust_union MSD/MAX/AVG; E-AT softplus) have their own architectures
and are instantiated with their own class + correct [0,1] preprocessing, then run through
the identical attacks. This is the apples-to-apples "our eval" yardstick for released
baselines. (Recreated in-repo after a scratchpad wipe.)
"""
import argparse, os, sys
from pathlib import Path
from types import SimpleNamespace

# Portable root: $ATTACKDRO_ROOT if set, else derived from this file's location
# (scripts/dev/union_bench_eval.py -> repo root). The old hard-coded absolute path
# only worked on the dev PC and silently broke Colab, where external/robust_union
# lives under /content/attackdro. The model class it needs must be in the code zip.
ROOT = Path(os.environ["ATTACKDRO_ROOT"]) if os.environ.get("ATTACKDRO_ROOT") \
    else Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import torch
import eval_multinorm_audit as H
from robustdro.utils.io import save_json


def build_robust_union(ckpt_path, device):
    """locuslab/robust_union PreActResNet18 (MSD/MAX/AVG). Inputs [0,1], no normalization."""
    import torch
    sys.path.insert(0, str(ROOT / "external/robust_union/CIFAR10/models"))
    from preact_resnet import PreActResNet18
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    m = PreActResNet18(); m.load_state_dict(sd, strict=True); m.to(device).eval()
    return m


def build_eat_softplus(ckpt_path, device):
    import torch
    sys.path.insert(0, str(ROOT / "external/RAMP"))
    from model_zoo.fast_models import PreActResNet18
    m = PreActResNet18(10, activation="softplus1")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ck, dict) and "state_dict" in ck:
        ck = ck["state_dict"]
    ck = {k.replace("module.", ""): v for k, v in ck.items()}
    m.load_state_dict(ck, strict=True); m.to(device).eval()
    return m


class _Normalize(torch.nn.Module):
    """CIFAR-10 mean/std normalization prepended to a model that expects normalized input."""
    def __init__(self, mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)):
        super().__init__()
        self.register_buffer("m", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("s", torch.tensor(std).view(1, 3, 1, 1))
    def forward(self, x):
        return (x - self.m) / self.s


_RB_NORMALIZE = False   # set from --normalize; used by build_robustbench


def build_robustbench(ckpt_path, device):
    """RobustBench CIFAR-10 model. ckpt_path is the cached file; the model NAME is its stem.
    Fixes for our stack: torch 2.11 defaults weights_only=True (RB v0.2.1/1.1 predate it) ->
    patch to False around load_model (trusted source). --normalize wraps CIFAR mean/std for
    models that expect normalized input (settled empirically by the B2 reproduce-check)."""
    import torch as _t
    name = Path(ckpt_path).stem
    _orig = _t.load
    _t.load = lambda *ar, **kw: _orig(*ar, **{**kw, "weights_only": False})
    try:
        from robustbench.utils import load_model
        m = load_model(model_name=name, dataset="cifar10", threat_model="Linf")
    finally:
        _t.load = _orig
    if _RB_NORMALIZE:
        m = torch.nn.Sequential(_Normalize(), m)
    m.to(device).eval()
    return m


BUILDERS = {"robust_union_preact": build_robust_union, "eat_fast_softplus": build_eat_softplus,
            "robustbench": build_robustbench}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--run-id", required=True); p.add_argument("--checkpoint-role", required=True)
    p.add_argument("--out", required=True); p.add_argument("--arch", required=True, choices=list(BUILDERS))
    p.add_argument("--bs", type=int, default=128); p.add_argument("--device", default="cuda")
    p.add_argument("--export-masks", action="store_true")
    p.add_argument("--normalize", action="store_true",
                   help="prepend CIFAR mean/std (for robustbench models that expect normalized input)")
    a = p.parse_args()

    global _RB_NORMALIZE
    _RB_NORMALIZE = a.normalize
    import torch
    device = "cuda" if (a.device == "cuda" and torch.cuda.is_available()) else "cpu"
    cfg = H.load_audit_config(a.config); H.validate_config(cfg)
    H.assert_write_path_safe(a.out)
    out_path = H.repo_path(a.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    model = BUILDERS[a.arch](H.repo_path(a.checkpoint), device)
    x, y, _, indices_sha = H.load_subset(cfg)
    clean = H.clean_mask(model, x, y, device, a.bs)
    print(f"[union-bench] {a.run_id} clean_acc={H.mean_mask(clean):.4f} n={len(y)}", flush=True)

    per_attack, masks = {}, {}
    for attack in H.iter_attacks(cfg):
        name = attack["name"]
        mask, qm, qx = H.run_attack_mask(model, x, y, attack, float(cfg["eps"][attack["eps_key"]]), device, a.bs)
        masks[name] = mask
        per_attack[name] = {"robust_acc": H.mean_mask(mask), "n": int(mask.numel()), "available": True,
                            "skipped": False, "params": dict(attack), "actual_queries_mean": qm, "actual_queries_max": qx}
        print(f"[union-bench]   {name:14s} robust_acc={H.mean_mask(mask):.4f}", flush=True)
    for req in ("apgd_ce_linf", "apgd_ce_l2", "apgd_ce_l1"):
        if req not in masks:
            raise RuntimeError(f"Primary APGD missing: {req}")

    ck_sha = H.sha256_file(H.repo_path(a.checkpoint))
    ns = SimpleNamespace(run_id=a.run_id, checkpoint_role=a.checkpoint_role, checkpoint=a.checkpoint,
                         config=a.config, device=device, out=a.out, mask_out=None, sidecar_only=False, existing_json=None)
    result = H.build_result(args=ns, cfg=cfg, checkpoint_sha=ck_sha, indices_sha=indices_sha,
                            clean=clean, per_attack=per_attack, masks=masks, skipped=[])
    result["baseline_arch"] = a.arch; result["is_reeval_ckpt"] = True
    if a.export_masks:
        H.export_mask_sidecar(args=ns, cfg=cfg, validation_result=result, masks=masks, skipped=[],
                              checkpoint_sha=ck_sha, indices_sha=indices_sha, source_json_path=a.out)
    save_json(result, str(out_path))
    print(f"[union-bench] saved -> {out_path}  union={result['full_audit_union']:.4f}", flush=True)


if __name__ == "__main__":
    main()
