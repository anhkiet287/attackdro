"""T1b — MAX/AVG base-generality 2x2 at @10k tier=full (paper table). Recomputed from masks."""
import numpy as np, os, json
ROOT=os.environ.get("ATTACKDRO_ROOT",os.getcwd()); MAIN=os.path.join(ROOT,"results/main")
NORMS=("linf","l2","l1")
def M(arm,lbl="10k"):
    p=os.path.join(MAIN,arm,lbl,"masks_multinorm_v1.npz")
    if not os.path.exists(p): return None
    d=np.load(p); return {k:d[k].astype(bool) for k in d.files if k!="metadata_json"}
def AND(m,ks):
    u=np.ones(len(m[ks[0]]),bool)
    for k in ks: u&=m[k]
    return u
def U(m): return AND(m,list(m))
def PN(m,nm): return AND(m,[k for k in m if k.endswith(nm)]).mean()
def pair(A,B,base):
    ma,mb=M(A),M(B)
    if ma is None or mb is None:
        print(f"| {base} | masks missing: {A if ma is None else B} @10k |"); return
    ua,ub=U(ma),U(mb); rng=np.random.default_rng(0); n=len(ua)
    D=np.array([ (lambda i: ua[i].mean()-ub[i].mean())(rng.integers(0,n,n)) for _ in range(10000)])
    d,l,u=D.mean(),np.quantile(D,.05),np.quantile(D,.95)
    sig="**+**" if l>0 else ("**−**" if u<0 else "ns")
    print(f"| {base} | {ub.mean():.4f} | {ua.mean():.4f} | **{d:+.4f}** | [{l:+.4f}, {u:+.4f}] | {sig} | "
          f"{PN(ma,'linf')-PN(mb,'linf'):+.4f} | {PN(ma,'l2')-PN(mb,'l2'):+.4f} | {PN(ma,'l1')-PN(mb,'l1'):+.4f} |")
print("\n### Base-generality 2x2 — @10k, tier=full (12 attacks), paired bootstrap B=10000 rng(0)\n")
print("| CE base | term OFF | term ON | Δ | 95% CI | sig | Δℓ∞ | Δℓ₂ | Δℓ₁ |")
print("|---|---|---|---|---|---|---|---|---|")
pair("M1a","M0","MSD-10 (headline)")
pair("M1a_max","M0_max","MAX")
pair("M1a_avg","M0_avg","AVG")
print("\n### Base effect (term held constant)\n")
print("| contrast | A | B | Δ | 95% CI | sig | Δℓ∞ | Δℓ₂ | Δℓ₁ |")
print("|---|---|---|---|---|---|---|---|---|")
for A,B,l in [("M0_max","M0","MAX vs MSD, term OFF"),("M1a_max","M1a","MAX vs MSD, term ON"),
              ("M0_avg","M0","AVG vs MSD, term OFF"),("M1a_avg","M1a","AVG vs MSD, term ON")]:
    pair(A,B,l)
