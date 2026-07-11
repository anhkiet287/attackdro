# PROJECT_MEMORY.md — canonical compact state
*Cơ chế: như memory của LLM — paste file này vào bất kỳ session mới nào để khôi phục đủ ngữ cảnh. Cập nhật TẠI CHỖ (không append như PROGRESS.md). Mỗi mục chỉ giữ thứ còn đúng ở hiện tại.*
*Last updated: 2026-07-02*

---

## 1 · IDENTITY (bất biến)
- **Owner:** Kiet — final-year CS @ HCMUT, mục tiêu **PhD Mỹ (apply 12/2026, Fall 2027)**.
- **Project:** Difficulty-aware DRO cho **worst-case robustness trên union (ℓ∞, ℓ2, ℓ1)** — phát triển từ đồ án AttackDRO++.
- **Goal tổng:** 1 workshop paper rigorous (NeurIPS ~29/8) + arXiv + code release → narrative nghiên cứu hoàn chỉnh cho hồ sơ PhD. Stretch: multi-norm robust CLIP.
- **Nguyên tắc tối thượng:** honesty > số đẹp; mechanism > leaderboard; negative result rigorous = tài sản.

## 2 · ORIGIN (từ đồ án)
- AttackDRO++ = GroupDRO với attacks-as-groups (+latent cluster từ gradient fingerprint). Thesis: chỉ ℓ∞+ℓ2, đo average-case.
- Kết quả thesis (20 seed, Wilcoxon, BH-FDR): **average +0.279pp (p=.0136), worst-case KHÔNG cải thiện** → chính là research question hiện tại: *vì sao re-weighting đẩy average mà không đẩy worst-case, và fix gì?*
- 3 lỗi positioning đã sửa: thiếu ℓ1 ✅ · đo average thay vì worst-union ✅ · thiếu baseline MSD/MAX ✅.

## 3 · LOCKED DECISIONS (chỉ đổi khi Kiet duyệt)
- **Protocol:** ε = (ℓ∞ **0.03**, ℓ2 **0.5**, ℓ1 **12**), CIFAR-10, PreActResNet-18, train==eval. File: `configs/base.yaml`.
- **Metric chính:** worst-case union robust acc (AND per-sample qua 3 norm). union ≤ min per-norm.
- **Eval policy:** mọi số từ `eval_union.py`; APGD (CE+T) để dò, full AutoAttack `standard` cho số cuối; eval parity tuyệt đối.
- **Seeds:** {0,1,2} dò · 20 seed + Wilcoxon + BH-FDR cho claim cuối.
- **Rules:** WORKFLOW_RULES.md (experiment card trước mọi run; 1 biến/lần; escalation matrix).

## 4 · VALIDATED NUMBERS (n=1000, APGD, ε∞=0.03 — nguồn: re-eval checkpoint locuslab bằng harness của ta)
| model | clean | ℓ∞ | ℓ2 | ℓ1 | **worst-∪** |
|---|---|---|---|---|---|
| **MSD (target)** | 82.1 | 44.6 | 64.5 | 46.7 | **42.5** |
| AVG (peer) | 84.6 | 40.7 | 65.5 | 47.7 | **38.7** |
| MAX | 81.7 | 39.2 | 62.1 | 26.2 | **25.0** |
| LINF | 83.9 | 47.7 | 57.7 | 7.0 | **7.0** |
| L2 | 90.7 | 22.7 | 63.1 | 21.1 | **17.5** |
| L1 | 73.7 | 0.0 | 0.1 | 0.0 | **0.0** |
- Harness **validated** vs số công bố (clean/ℓ2 khớp; ℓ∞/ℓ1/union thấp hơn vì AutoAttack mạnh hơn PGD của họ — pattern đúng).
- PGD-AT tự train (8/255, pipeline proof, KHÔNG vào bảng): clean 82.0, union **9.4** (sụp về ℓ1).
- **AttackDRO++ P1 (our recipe, seed-0 provisional):** clean 80.7 · ℓ∞ 46.2 · ℓ2 63.1 · ℓ1 41.2 · union **38.7** (= AVG). Seeds 1–2 đang chạy lấy variance.

## 5 · FINDINGS (tài sản paper)
- **F1:** published ℓ1 robustness bị overstate — APGD-ℓ1 phá nhiều hơn PGD-L1-topk (LINF 7.0 vs 16.0 report; MAX 26.2 vs 39.4).
- **F2:** model MẠNH (MSD/AVG) bị chặn bởi **ℓ∞** (không phải ℓ1); union-gap chỉ ~2pp. Model single-norm mới sụp vì ℓ1. → Beat MSD = nâng ℓ∞ hoặc thu gap mà không hy sinh norm khác.
- **F3 (CONFIRMED by seed-0 APGD — ℓ2 free/spillover):** q của GroupDRO sụp về (ℓ∞ .83, ℓ2 **.00**, ℓ1 .17) — bỏ đói hoàn toàn nhóm ℓ2. NHƯNG APGD cho **ℓ2 = 63.1** (≈ MSD 64.5, AVG 65.5) DÙ q_ℓ2=0 → **ℓ2 robustness là FREE, spillover từ train ℓ∞+ℓ1**, KHÔNG tụt thành binding norm. ⇒ starvation là **lành tính**; **q-floor / anti-starvation OFF** (CARD-4 gate ℓ2<45% không đạt). Fix #1 cũ (q-floor/CVaR) bị BÁC.
- **F4 (candidate — degeneration-to-AVG + signal misalignment):** AttackDRO++ (GroupDRO ℓ∞+ℓ2+ℓ1, seed-0 APGD) worst-∪ = **38.7 = TRÙNG KHÍT AVG 38.7**, NO-GO vs MSD (−3.8). Difficulty-aware DRO trên attack-groups **thoái hoá về averaging** cho worst-case. Cơ chế: q-by-loss (scalar/nhóm) **lệch khỏi** mục tiêu union — dồn vào ℓ∞ (loss cao nhất → ℓ∞ 46.2 **vượt** MSD 44.6) nhưng bỏ nhẹ **ℓ1 binding** (41.2, yếu nhất, thua MSD 46.7). ⇒ fix = **binding-aware signal** (CARD-3a group: q∝softmax(−robust_acc/τ); CARD-3b per-sample soft AVG↔MAX), KHÔNG phải floor/CVaR. Xác nhận kỳ vọng thesis: reweighting đẩy average, không đẩy worst-case.
- **F5 (candidate, 1-ckpt — verify trên ckpt khác, vd avg_frozen khi có):** weak-attack probe **ĐẢO NGƯỢC ranking độ khó norm**: probe PGD-20 nói weakest = ℓ∞ (50.3 < ℓ1 53.5), APGD nói weakest = **ℓ1** (41.2 < ℓ∞ 46.2); bias per-norm ℓ1 **+12.3pp** vs ℓ∞ +4.1 / ℓ2 +2.6 (seed-0 best ckpt, epoch 33). ⇒ mọi weighting calibrate bằng probe yếu **mis-target CẤU TRÚC** (không phải noise) — mở rộng F1 từ "overstate số" thành "đảo ranking". Hệ quả: (a) pre-registered prediction cho 3a pilot: ≈38.7, q dồn ℓ∞; (b) per-sample raw loss (3b) nhiều khả năng mang cùng norm-scale bias → prior cho zscore variant TĂNG (vẫn evidence-gated qua dist/* logs); (c) fix = CARD-3a-v2 val-calibrated q (APGD-CE trên val held-out). ℓ∞ = dày/đều (concentration .106); ℓ2 ≈ ℓ1 = thưa, dồn vào vùng salient (corr saliency +.72/+.86). Trục khó của union = ℓ∞ vs sparse.

## 6 · REFUTED IDEAS (đừng mở lại — kèm bài học)
- ❌ **Spatial masking / union-of-masks:** perturbation ĐÁNH TRÚNG vùng salient, không né → mask không gian không tách được robust/non-robust (vấn đề là NGỮ NGHĨA feature, không phải VỊ TRÍ). Lit: Wang&Horne, Ilyas.
- ❌ **Per-norm adapter + logit-selection (MoE):** hard-gate sập dưới adaptive attack (attacker route qua router trong budget từng norm); area đông (ADVMoE/SoE/DWF); F2: specialist không compose (LINF ℓ1=7, L1 sụp 0).
- **Meta-lesson:** robustness đến từ **objective lúc train**, không từ kiến trúc/mẹo lúc inference.

## 7 · PHASE GOALS + MILESTONES (neo deadline thật)
| Phase | Thời gian | Goal | Gate |
|---|---|---|---|
| ~~P0~~ ✅ | done | Harness validated + baseline table | — |
| **P1** 🔄 | → 08/07 | AttackDRO++ 3-seed probe (ℓ∞+ℓ2+ℓ1) vs bảng §4 | **G1:** verdict + chốt thứ tự P2 + bank tài sản Claude (Max hết 08/07) |
| **P2** | 09/07→09/08 | 1 fix ≥ MSD 42.5 HOẶC match-rẻ-hơn, KÈM mechanism evidence. Thứ tự: q-floor/CVaR → binding-aware → geometry grouping. Song song: mechanism figs + retrain AVG/MAX in-house | **G2:** chọn winner (3 kịch bản đều viết được) |
| **P3** | 10/08→**29/08** | 20-seed + full AA winner → viết → **NỘP NeurIPS workshop** (draft cho mentor 22/08) | **G3:** đã nộp (deadline cứng, scope co theo) |
| **P4** | 09→15/10 | arXiv + code release public; CLIP multi-norm CHỈ nếu còn ≥4 tuần | **G4:** chuyển ưu tiên sang apps |
| **P5** | 15/10→15/12 | SOP (arc: negative result→rigor→mechanism→fix) + thư (xin 01/11) + email prof | **PhD deadline 01/12 & 15/12** |
- Đọc verdict P1: **>42.5** = verify-rồi-GO (nghi bug trước) · **38.7–42.5** = tốt, P2 đóng gap · **<38.7** = check ℓ2 starvation, mechanism story.

## 8 · ASSETS (cái gì ở đâu)
- **Repo `ATTACKDRO`** (PC, WSL2): configs/ (base+pgd_at+attackdro) · src/robustdro/ (data, models, attacks[linf,l2,l1-topk], training[GroupDRO], eval[eval_union+AA wrapper], utils[stats,wandb]) · scripts/ (train, evaluate, validate_baseline, visualize_attack, perturbation_geometry, union_mask_test) · external/robust_union/ (6 ckpt official) · CLAUDE.md + WORKFLOW_RULES.md + PROGRESS.md.
- **Workflow:** MacBook → SSH (Tailscale) → PC 5070ti (Blackwell sm_120, **torch cu128 bắt buộc**, venv phải activate); long run trong tmux riêng; W&B (user kietna); Colab Pro backup.
- **Research Hub** (React + persistent storage): Overview/Roadmap/Board/Papers/Runs/Notes/Links + module Docs, Timeline (deadline-anchored), Assistant (BYOK AI); deploy kit Vite+Vercel (storage.js, ai.js, DEPLOY.md).
- **Docs:** Research_Proposal_UnionDRO.docx (gửi thầy+mentor, có mục Q1–Q6) · Roadmap_v2_DeadlineAnchored.md · roadmap v1.
- **Reading list** ~18 papers 4 tier (Madry, TRADES, Tramèr&Boneh, MSD, AutoAttack, E-AT / GroupDRO, GEORGE, DFR, CVaR / NCAT, PROTECTOR, CURE / TeCoA, FARE).

## 9 · OPEN / PENDING
- [x] Verdict P1 seed-0: **38.7, NO-GO vs MSD, tie AVG** (2026-07-02). Seeds 1–2 chạy nền lấy variance (tmux `p1pipe`).
- [🔄] **CARD-3a bindaware** 1-seed pilot đang train (tmux `p2pipe`, `configs/bindaware.yaml`, q∝softmax(−robust_acc/τ), τ=0.1). Kết quả ~4h (chạy song song p1).
- [🔄] **CARD-1 avg_frozen** 3 seed — queue trong `p2pipe` sau khi p1 xong (= AVG in-house, giết recipe confound).
- [🔄] **CARD-3b APPROVED (3 amendments)** — code + smoke DONE (`per_sample_soft`, T-softmax AVG↔MAX, dist-logging, zscore variant evidence-gated; F5 → prior zscore tăng). **T-sweep {0.25,0.5,1,2} armed trong tmux `p3bsweep`**, tự nổ SAU khi 3a pilot lands. Lit-check sơ bộ trong card: gần nhất arXiv 2210.00557 — đọc kỹ trước novelty claim.
- [🔄] **CARD-3a-v2 (val-calibrated q) GATE OPEN + implemented + smoked** (`configs/bindaware_v2.yaml`: val_holdout 1000, APGD-CE 30it/5ep, EMA .5). **Pilot armed tmux `p3av2`** — gate: 3a landed VÀ p1 xong (≤2 trainer cùng 3b sweep). Mục tiêu 2×2: signal (probe/val-APGD) × aggregation (group/per-sample). Pre-registered prediction cho 3a pilot: ≈38.7, q dồn ℓ∞ (F5) — báo cáo 3a PHẢI đối chiếu prediction này.
- [ ] **CARD-1 avg_frozen ×3 → COLAB** (`notebooks/colab_avg_frozen.ipynb`). **BLOCKER: repo chưa push** (last commit = scaffold ban đầu) — cần Kiet OK để commit+push toàn bộ P0–P2 code.
- [ ] CARD-2 (cvar) deprioritized; CARD-4 (qfloor) OFF (F3: ℓ2 free).
- [ ] **Kiet: tắt sleep PC (Windows, admin PowerShell):** `powercfg /change standby-timeout-ac 0` + `powercfg /change hibernate-timeout-ac 0` — seed-0 từng stall 3.25h vì máy ngủ.
- [ ] Chưa retrain MAX in-house (AVG sẽ có từ avg_frozen; recipe confound — cần cho bảng cuối P3).
- [ ] Full AutoAttack `standard` chưa test timing trên 10k.
- [ ] Chọn workshop cụ thể sau khi NeurIPS công bố danh sách (sau 11/07).
- [ ] Gửi proposal cho thầy + mentor (chèn số P1 khi có).

## 10 · UPDATE DISCIPLINE (để file này sống)
- Cập nhật khi: gate đóng · số mới validated · quyết định mới lock · idea bị refute · asset mới. Sửa tại chỗ + đổi "Last updated".
- Giữ ≤ ~150 dòng: cũ/hết đúng → xoá hoặc nén 1 dòng vào mục liên quan. Chi tiết lịch sử thuộc về PROGRESS.md, không phải đây.
- Phân công: Claude Code cập nhật §4–§9 sau mỗi milestone; Kiet duyệt §1–§3 (identity/decisions).