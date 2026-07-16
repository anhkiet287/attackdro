"""Produce external/RAMP/RAMP_claimB.py from RAMP.py with Claim-B injections.

Runs on Colab (where external/RAMP + its env live). Reads RAMP.py, applies exact-anchor
string insertions to add: (a) a projection head + its own SGD optimizer, (b) the Claim-B
representation term (reusing RAMP's own l-inf/l-1 adversarials), (c) a per-epoch worst-union
val-select checkpoint (val_best.pth). Base RAMP loss / pred-alloc / GP are UNTOUCHED.

Usage (from external/RAMP):  python /path/to/patch_ramp_claimb.py
Then run:  python RAMP_claimB.py --claimB B1 ... (see notebook).  Verify each anchor patched.
"""
import io, sys, re
from pathlib import Path

RAMP = Path(__file__).resolve().parent / "RAMP.py" if (Path(__file__).resolve().parent / "RAMP.py").exists() \
       else Path("RAMP.py")
src = RAMP.read_text()
n_applied = 0

def patch(anchor, insert_after=None, replace_with=None, tag=""):
    global src, n_applied
    if anchor not in src:
        print(f"  !! ANCHOR NOT FOUND [{tag}]: {anchor[:60]!r}"); return
    if replace_with is not None:
        src = src.replace(anchor, replace_with, 1)
    else:
        src = src.replace(anchor, anchor + insert_after, 1)
    n_applied += 1; print(f"  ok [{tag}]")

# --- (0) argparse: add Claim-B flags (anchor on RAMP's lbd arg) ---
patch("    parser.add_argument('--lbd', type=float, default=1.5)",
      insert_after=(
"\n    parser.add_argument('--claimB', type=str, default='none', choices=['none','B1','B2'])"
"\n    parser.add_argument('--cb_alpha', type=float, default=0.5)"
"\n    parser.add_argument('--cb_beta',  type=float, default=0.5)"
"\n    parser.add_argument('--cb_gamma', type=float, default=0.2)"
"\n    parser.add_argument('--cb_tau',   type=float, default=0.1)"), tag="args")

# --- (1) head + head optimizer + rep import + val data, right after RAMP's optimizer ---
patch("""    optimizer = optim.SGD(get_params_no_decay(args, model), lr=1., momentum=0.9,
        weight_decay=args.weight_decay)""",
      insert_after=(
"""
    # ===CLAIM-B=== projection head (discarded at eval) + its own optimizer + val-select data
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.environ.get('CLAIMB_REP_DIR', '.'))
    import claim_b_rep as _cb
    import torch as _t, torchvision as _tv, torchvision.transforms as _T
    _head = _cb.build_head().cuda()
    _head_opt = optim.SGD(_head.parameters(), lr=0.05, momentum=0.9, weight_decay=args.weight_decay)
    _dsv = _tv.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=_T.ToTensor())
    _Xv = _t.stack([_dsv[i][0] for i in range(49000, 50000)]); _Yv = _t.tensor([_dsv[i][1] for i in range(49000, 50000)])
    _cb_best = -1.0
    def _cb_val_worst_union(mdl):
        mdl.eval(); rob = _t.ones(1000, dtype=_t.bool)
        for _nm,_e in [('Linf',8/255),('L2',0.5),('L1',12.0)]:
            ok = _t.zeros(1000, dtype=_t.bool)
            for i in range(0, 1000, 250):
                xb=_Xv[i:i+250].cuda(); yb=_Yv[i:i+250].cuda()
                xa,_,_,_,_ = apgd_train(mdl, xb, yb, norm=_nm, eps=_e, n_iter=20, is_train=False)
                ok[i:i+250]=(mdl(xa).argmax(1)==yb).cpu()
            rob &= ok
        return rob.float().mean().item()
"""), tag="head+valdata")

# --- (2) zero head grads + sync head lr, right after RAMP zeroes its optimizer ---
patch("""                    # zero the parameter gradients
                    optimizer.zero_grad()""",
      insert_after="""
                    _head_opt.zero_grad(); _head_opt.param_groups[0].update(lr=lr)  # ===CLAIM-B==="""
      , tag="zero+lr")

# --- (3) add the rep term to the loss (replace the loss line) ---
patch("                        loss = loss_best + loss_kl * args.lbd",
      replace_with=(
"""                        # ===CLAIM-B=== representation term on RAMP's own adversarials
                        if args.claimB == 'B1':
                            _fcl = model(x, return_features=True)
                            _fs  = model(x_tr_s, return_features=True)   # l-inf
                            _ft  = model(x_tr_t, return_features=True)   # l-1
                            _rep, _ = _cb.rep_b1_pullpush(_head, _fcl, [_fs, _ft], y_tr, args.cb_alpha, args.cb_beta, args.cb_tau)
                        elif args.claimB == 'B2':
                            _fs  = model(x_tr_s, return_features=True)   # l-inf only
                            _rep, _ = _cb.rep_b2_supcon(_head, _fs, y_tr, args.cb_gamma, args.cb_tau)
                        else:
                            _rep = loss_best.new_zeros(())
                        loss = loss_best + loss_kl * args.lbd + _rep"""), tag="loss+rep")

# --- (4) step the head optimizer right after RAMP steps its optimizer ---
patch("""                    loss.backward()
                    optimizer.step()""",
      insert_after="""
                    _head_opt.step()  # ===CLAIM-B==="""
      , tag="head-step")

# --- (5) per-epoch val-select checkpoint (val_best.pth), right after the GP fusion block ---
patch("""                model_nat.load_state_dict(model_dict)
                model.load_state_dict(model_dict)""",
      insert_after="""
            # ===CLAIM-B=== worst-union val-select -> save best backbone (ramp-family auditable)
            _wu = _cb_val_worst_union(model)
            print(f'[claimB {args.claimB}] epoch {epoch} val_worst_union {_wu:.4f}', flush=True)
            if _wu > _cb_best:
                _cb_best = _wu
                _t.save(model.state_dict(), '{}/{}/val_best.pth'.format(args.save_dir, args.fname))
                _t.save({'epoch': epoch, 'val_worst_union': _wu}, '{}/{}/val_best_meta.pth'.format(args.save_dir, args.fname))"""
      , tag="valbest")

out = RAMP.parent / "RAMP_claimB.py"
out.write_text(src)
print(f"\napplied {n_applied}/6 patches -> {out}")
if n_applied < 6:
    print("WARNING: some anchors missing — inspect RAMP.py; do NOT run the full job until all 6 patched.")
