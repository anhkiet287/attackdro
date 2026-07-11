#!/usr/bin/env python
"""Generate external/RAMP/RAMP_predalloc.py from pristine external/RAMP/RAMP.py by applying the
predictive-allocation graft (EXPLORATION — forced by Kiet despite the source/target mismatch).
Keeps RAMP.py untouched (= Arm A, RAMP-full). Every patch is an exact-match replace that asserts
its anchor exists, so it fails loudly if RAMP.py drifts. Run: python scripts/dev/make_ramp_predalloc.py
"""
import os

SRC = "external/RAMP/RAMP.py"
DST = "external/RAMP/RAMP_predalloc.py"

PATCHES = [
    # 1. after the cifar train loader is built: wrap it with a dataset-index loader + import pred_alloc
    ("        train_loader, _ = data.load_cifar10_train(args, only_train=True)\n",
     "        train_loader, _ = data.load_cifar10_train(args, only_train=True)\n"
     "        # --- EXPLORATION: indexed loader for phi's per-sample binding EMA ---\n"
     "        import pred_alloc as _pa\n"
     "        from torch.utils.data import DataLoader as _DL, Dataset as _DS\n"
     "        class _IdxDS(_DS):\n"
     "            def __init__(s, b): s.b = b\n"
     "            def __len__(s): return len(s.b)\n"
     "            def __getitem__(s, i):\n"
     "                r = s.b[i]; return r[0], r[1], i\n"
     "        _idx_ds = _IdxDS(train_loader.dataset)\n"
     "        train_loader_idx = _DL(_idx_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)\n"),

    # 2. after source/target norms are chosen: init phi + FLOPs accumulators
    ("    cur_norm_target = args.target # initial target - L1\n",
     "    cur_norm_target = args.target # initial target - L1\n"
     "    # --- EXPLORATION: phi predictor over the two crafted norms {source, target} ---\n"
     "    _dev = next(model.parameters()).device\n"
     "    _phi = _pa.Phi(len(_idx_ds), [args.l_norms[cur_norm_source], args.l_norms[cur_norm_target]],\n"
     "                   device=_dev, beta=0.5, cold_epochs=1, floor_iters=int(os.environ.get('PREDALLOC_FLOOR', 2)))\n"
     "    _flops_by_epoch = {}\n"),

    # 3. reset per-epoch FLOPs counters at the top of each epoch
    ("    for epoch in range(0, args.epochs):  # loop over the dataset multiple times\n",
     "    for epoch in range(0, args.epochs):  # loop over the dataset multiple times\n"
     "        _alloc_passes = 0.0; _alloc_ref = 0.0   # EXPLORATION FLOPs accounting\n"),

    # 4. iterate the INDEXED loader in the main adversarial loop (train_clean still uses train_loader)
    ("            with tqdm(train_loader, unit=\"batch\") as tepoch:\n",
     "            with tqdm(train_loader_idx, unit=\"batch\") as tepoch:\n"),
    ("                for i, (x_loader, y_loader) in enumerate(tepoch):\n",
     "                for i, (x_loader, y_loader, idx_loader) in enumerate(tepoch):\n"),
    ("                    x, y = x_loader.cuda(), y_loader.cuda()\n",
     "                    x, y = x_loader.cuda(), y_loader.cuda()\n"
     "                    idx = idx_loader.to(_dev)\n"),

    # 5. THE graft: replace the two full crafts with phi-allocated (full on predicted, floor on other)
    ("                    x_tr_s, _, _, loss_best_s, _ = apgd_train(model, x, y, norm=args.l_norms[cur_norm_source],\n"
     "                                eps=args.l_eps[cur_norm_source], n_iter=args.l_iters[cur_norm_source], is_train=True)               \n"
     "\n"
     "                    x_tr_t, _, _, loss_best_t, _ = apgd_train(model, x, y, norm=args.l_norms[cur_norm_target],\n"
     "                                eps=args.l_eps[cur_norm_target], n_iter=args.l_iters[cur_norm_target], is_train=True)\n",
     "                    _pred = _phi.predict(idx, epoch)   # ONCE per batch (shared by both crafts)\n"
     "                    x_tr_s, loss_best_s, _ps = _pa.alloc_craft(model, x, y, idx, 0,\n"
     "                                args.l_norms[cur_norm_source], args.l_eps[cur_norm_source],\n"
     "                                args.l_iters[cur_norm_source], _phi, _pred)\n"
     "                    x_tr_t, loss_best_t, _pt = _pa.alloc_craft(model, x, y, idx, 1,\n"
     "                                args.l_norms[cur_norm_target], args.l_eps[cur_norm_target],\n"
     "                                args.l_iters[cur_norm_target], _phi, _pred)\n"
     "                    _alloc_passes += _ps + _pt\n"
     "                    _alloc_ref += (args.l_iters[cur_norm_source] + args.l_iters[cur_norm_target]) * len(y)\n"),

    # 6. after the epoch's batch loop: record + print the measured attack-FLOPs ratio
    ("        stats['loss_train_dets']['clean'][epoch] = running_loss_t / len(train_loader) #running_loss / len(train_loader)\n",
     "        stats['loss_train_dets']['clean'][epoch] = running_loss_t / len(train_loader) #running_loss / len(train_loader)\n"
     "        _ratio = _alloc_passes / max(_alloc_ref, 1.0)   # EXPLORATION: allocated vs 2*full\n"
     "        _flops_by_epoch[epoch] = _ratio\n"
     "        print('[predalloc] epoch {} attack_flops_ratio = {:.3f} (source={} target={})'.format(\n"
     "            epoch, _ratio, args.l_norms[cur_norm_source], args.l_norms[cur_norm_target]))\n"),
]


def main():
    if not os.path.exists(SRC):
        raise SystemExit(f"{SRC} missing — clone RAMP first")
    txt = open(SRC).read()
    header = ("# ===================================================================\n"
              "# RAMP_predalloc.py — AUTO-GENERATED from RAMP.py by\n"
              "# scripts/dev/make_ramp_predalloc.py. DO NOT EDIT BY HAND. EXPLORATION\n"
              "# lane (Colab): predictive per-norm allocation grafted onto RAMP's\n"
              "# source/target crafts (full on phi's predicted-binding norm, floor on the\n"
              "# other). RAMP's max + KL-pairing + weight-GP are untouched. See\n"
              "# results/exploration/generality_predictive_on_ramp.md.\n"
              "# ===================================================================\n")
    for i, (old, new) in enumerate(PATCHES, 1):
        if old not in txt:
            raise SystemExit(f"PATCH {i} anchor NOT FOUND — RAMP.py drifted. Anchor:\n{old[:120]}")
        if txt.count(old) != 1:
            raise SystemExit(f"PATCH {i} anchor is not unique ({txt.count(old)}x)")
        txt = txt.replace(old, new)
    open(DST, "w").write(header + txt)
    print(f"wrote {DST} ({len(PATCHES)} patches applied)")


if __name__ == "__main__":
    main()
