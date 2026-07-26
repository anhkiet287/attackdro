"""Backbone-space representation-collapse dump for a ramp-family checkpoint (B1 / R' / B2).

The projection head is DISCARDED at eval (only the backbone state_dict is saved in val_best),
so collapse is measured in the *deployment* representation — the pooled-512 penultimate features
the classifier reads — NOT in head space. This is exactly the space where a pull-push term could
have cheated the val proxy by collapsing features, so it is the right thing to probe.

Metrics (Wang & Isola 2020 conventions), on L2-normalized pooled-512 features:
  alignment   = E_x [ <f(x), f(x_adv)> ]           per-norm cosine clean-vs-adv (higher = adv stays near clean)
  uniformity  = log E_{i,j} exp(-2 * ||f(x_i)-f(x_j)||^2)   on clean (near 0 => collapsed; more negative = spread)
  embed_norm  = E_x ||f(x)||_2   pre-normalization pooled-512 norm (tiny/degenerate => collapse)

Reuses the FROZEN harness for model build + subset load + eps; adversarials via RAMP's own
apgd_train (same attack the patched val-selection used). Run from external/RAMP on Colab.

Usage (Colab, cwd = external/RAMP):
  ATTACKDRO_ROOT=/content/attackdro python /path/scripts/dev/collapse_dump.py \
    --checkpoint <val_best.pth> --config configs/eval/audit_cifar10_preactrn18_multinorm_v3A_testfinal.yaml \
    --name B1 --n 512 --out results/eval/union_bench/B1/collapse.json
"""
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(os.environ.get("ATTACKDRO_ROOT", "/mnt/c/Users/ADMIN/Documents/Claude/Projects/ATTACKDRO"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import eval_multinorm_audit as H
from robustdro.eval.eval_union import load_eval_checkpoint

NORMS = [("Linf", "linf"), ("L2", "l2"), ("L1", "l1")]


def pool(feat):
    """ramp forward(return_features=True) -> post-bn 8192-d (512x4x4) -> mean-pool to 512."""
    import torch  # noqa: F401
    if feat.dim() == 2 and feat.shape[1] == 8192:
        return feat.view(feat.shape[0], 512, 4, 4).mean(dim=(2, 3))
    if feat.dim() == 2 and feat.shape[1] == 512:
        return feat
    if feat.dim() == 4:
        return feat.mean(dim=(2, 3))
    raise ValueError(f"unexpected feature shape {tuple(feat.shape)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--n", type=int, default=512, help="examples for the collapse batch")
    p.add_argument("--n-iter", type=int, default=20)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    import torch
    import torch.nn.functional as F
    # RAMP lives at $RAMP_DIR (Colab clone, e.g. /content/RAMP) or external/RAMP (PC).
    for cand in [os.environ.get("RAMP_DIR"), "/content/RAMP", str(ROOT / "external/RAMP"), os.getcwd()]:
        if cand and (Path(cand) / "autopgd_train.py").exists():
            sys.path.insert(0, cand); break
    from autopgd_train import apgd_train

    device = "cuda" if (a.device == "cuda" and torch.cuda.is_available()) else "cpu"
    cfg = H.load_audit_config(a.config); H.validate_config(cfg)
    H.assert_write_path_safe(a.out)
    out_path = H.repo_path(a.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    eval_cfg = H.to_eval_cfg(cfg)
    model, _ = load_eval_checkpoint(H.repo_path(a.checkpoint), eval_cfg, model_family="ramp", device=device)
    model.eval()

    x, y, _, _ = H.load_subset(cfg)
    n = min(a.n, len(y))
    x = x[:n].to(device); y = y[:n].to(device)

    def feats(xb):  # pooled-512, no head
        return pool(model(xb, return_features=True))

    with torch.no_grad():
        f_clean = feats(x)
        z_clean = F.normalize(f_clean, dim=1)
        embed_norm = f_clean.norm(dim=1).mean().item()
        sq = torch.pdist(z_clean, p=2).pow(2)
        uniformity = torch.log(torch.exp(-2.0 * sq).mean() + 1e-12).item() if len(sq) else float("nan")

    align = {}
    for nm, ek in NORMS:
        eps = float(cfg["eps"][ek])
        # apgd_train needs grad internally; is_train=False keeps model in eval mode
        xa, *_ = apgd_train(model, x, y, norm=nm, eps=eps, n_iter=a.n_iter, is_train=False)
        with torch.no_grad():
            za = F.normalize(feats(xa), dim=1)
            align[nm] = (za * z_clean).sum(1).mean().item()

    result = {
        "name": a.name, "checkpoint": a.checkpoint, "n": n, "space": "backbone_pooled512_l2norm",
        "family": "ramp", "config_id": H.config_identity(cfg) if hasattr(H, "config_identity") else None,
        "alignment_per_norm": align,
        "alignment_worst": min(align.values()), "alignment_mean": sum(align.values()) / len(align),
        "uniformity": uniformity, "embed_norm": embed_norm,
        "note": "head discarded at eval; collapse measured in deployment (pooled-512) space",
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[collapse {a.name}] align(worst/mean)={result['alignment_worst']:.4f}/"
          f"{result['alignment_mean']:.4f}  uniformity={uniformity:.4f}  embed_norm={embed_norm:.4f}", flush=True)
    print(f"[collapse {a.name}] per-norm align: " +
          "  ".join(f"{k} {v:.4f}" for k, v in align.items()), flush=True)
    print(f"[collapse {a.name}] saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
