"""F1 pre-launch: measure the gradient scale of CLAMP's term vs the CE-on-views term, so `w` is a
MEASURED choice rather than an asserted one.

Both are auxiliary losses added to the same base CE. What matters to the optimiser is the size of the
gradient they inject into the shared encoder theta_f (backbone minus classifier, minus head), so that
is what is measured — on the SAME batches, with the SAME crafted views.

  ratio = ||d/dtheta_f (alpha*scaffold + beta*glue)||  /  ||d/dtheta_f (mean_p CE(view_p))||

Setting w = ratio makes the two auxiliaries inject the same gradient norm at that point. The ratio is
NOT constant over training, so it is reported at two states and the drift is shown, not hidden.
"""
from __future__ import annotations
import argparse, os, sys
import torch, torch.nn.functional as F
ROOT = os.environ.get("ATTACKDRO_ROOT", os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, ROOT)
import importlib.util
_s = importlib.util.spec_from_file_location("c5", os.path.join(ROOT, "scripts/dev/c5_fromscratch.py"))
c5 = importlib.util.module_from_spec(_s); _s.loader.exec_module(c5)
from robustdro.attacks.apgd_train import apgd_train      # noqa: E402


def gnorm(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum((x**2).sum() for x in g if x is not None)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None, help="omit for fresh init (the state training starts from)")
    p.add_argument("--batches", type=int, default=6)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)

    model = c5.BackboneHead().to(dev)
    tag = "fresh init (seed %d)" % a.seed
    if a.ckpt:
        ck = torch.load(os.path.join(ROOT, a.ckpt), map_location="cpu", weights_only=False)
        sd = ck.get("model", ck)
        if not any(k.startswith("b.") for k in sd):
            sd = {f"b.{k}": v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        tag = f"{a.ckpt} (head {'loaded' if not any('head' in k for k in missing) else 'RANDOM — ckpt has no head'})"

    import torchvision, torchvision.transforms as T
    ds = torchvision.datasets.CIFAR10(root=os.path.join(ROOT, "data"), train=True, download=False,
                                      transform=T.Compose([T.RandomCrop(32, padding=4),
                                                           T.RandomHorizontalFlip(), T.ToTensor()]))
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=2,
                                     generator=torch.Generator().manual_seed(a.seed))
    cls_ids = {id(q) for q in model.b.linear.parameters()}
    head_ids = {id(q) for q in model.head.parameters()}
    theta_f = [q for q in model.parameters() if id(q) not in cls_ids and id(q) not in head_ids]

    print(f"[F1] state = {tag}")
    print(f"[F1] theta_f = {len(theta_f)} tensors (backbone only; classifier and head excluded)\n")
    print(f"{'batch':<7}{'||g| pull-push':>16}{'||g| CE-views':>15}{'ratio pp/cev':>14}{'||g| base CE':>14}")
    R = []
    for bi, (x, y) in enumerate(dl):
        if bi >= a.batches: break
        x, y = x.to(dev), y.to(dev)
        model.eval()
        xb = c5.msd_v0(model, x, y, c5.EPS["Linf"], c5.EPS["L2"], c5.EPS["L1"], steps=10).detach()
        views = [apgd_train(model, x, y, nm, c5.EPS[nm], n_iter=10, is_train=True).detach()
                 for nm in c5.NORMS]
        model.train()
        l_pp, _ = c5.decoupled_pullpush(model, x, views, y, 0.5, 0.5, 0.1, "adv")
        g_pp = gnorm(l_pp, theta_f)
        l_cev = c5.ce_at_loss(model, views, y, agg="avg")
        g_cev = gnorm(l_cev, theta_f)
        l_base = F.cross_entropy(model.logits(xb), y)
        g_base = gnorm(l_base, theta_f)
        R.append((g_pp, g_cev, g_base))
        print(f"{bi:<7}{g_pp:>16.4f}{g_cev:>15.4f}{g_pp/max(g_cev,1e-12):>14.3f}{g_base:>14.4f}")
    import statistics as st
    pp = st.mean(r[0] for r in R); cev = st.mean(r[1] for r in R); base = st.mean(r[2] for r in R)
    print("-" * 66)
    print(f"{'mean':<7}{pp:>16.4f}{cev:>15.4f}{pp/max(cev,1e-12):>14.3f}{base:>14.4f}")
    print(f"\n  gradient-matched w  =  ||g_pullpush|| / ||g_CE-views||  =  {pp/max(cev,1e-12):.3f}")
    print(f"  (auxiliary/base gradient ratio: pull-push {pp/max(base,1e-12):.3f}x, "
          f"CE-views {cev/max(base,1e-12):.3f}x of the base CE)")


if __name__ == "__main__":
    main()
