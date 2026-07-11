# PROJECT_MEMORY.md — canonical compact state
*Cơ chế: như memory của LLM — paste file này vào bất kỳ session mới nào để khôi phục đủ ngữ cảnh. Cập nhật TẠI CHỖ (không append như PROGRESS.md). Mỗi mục chỉ giữ thứ còn đúng ở hiện tại.*
*Last updated: 2026-07-04 (claim ladder revised + RAMP protocol-verify launched — see docs/SYNC_SNAPSHOT.md)*

---

## 1 · IDENTITY (bất biến)
- **Owner:** Kiet — final-year CS @ HCMUT, mục tiêu **PhD Mỹ (apply 12/2026, Fall 2027)**.
- **Project:** Difficulty-aware DRO cho **worst-case robustness trên union (ℓ∞, ℓ2, ℓ1)** — phát triển từ đồ án AttackDRO++.
- **Goal tổng (PIVOT 2026-07-04 → METHOD-FORWARD):** 1 workshop paper rigorous (NeurIPS ~29/8) + arXiv + code. **Target = beat-or-approach the from-scratch SOTA cluster (43.9–44.6) via composition (CARD-8b: binding-aware T-softmax thay/làm mềm L_max của RAMP), HOẶC the argued gap:** "mọi aggregation hiện tại — MSD per-step hard max · E-AT fixed geometric prescription · RAMP hard L_max + fixed λ — đều STATIC + sample-agnostic; KHÔNG cái nào thích nghi theo cấu trúc binding per-sample, state-dependent đo được (F5/F6, union<min gap, binding frequencies)." **F1–F7 KHÔNG bỏ — RE-PURPOSED thành lập luận gap (paper §2–3).** Fallback lane = paper_draft_v0.md (mechanism framing). Stretch: multi-norm robust CLIP.
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
| **MSD (classical strong baseline)** | 82.1 | 44.6 | 64.5 | 46.7 | **42.5** |
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
- **F3 (CONFIRMED by seed-0 APGD — ℓ2 free/spillover; NAY CÓ NỀN LÝ THUYẾT):** q của GroupDRO sụp về (ℓ∞ .83, ℓ2 **.00**, ℓ1 .17) — bỏ đói hoàn toàn nhóm ℓ2. NHƯNG APGD cho **ℓ2 = 63.1** (≈ MSD 64.5, AVG 65.5) DÙ q_ℓ2=0 → **ℓ2 robustness là FREE, spillover từ train ℓ∞+ℓ1**, KHÔNG tụt thành binding norm. ⇒ starvation **lành tính**; q-floor/anti-starvation OFF. **Grounding ([Croce2022-EAT] Thm 3.1 + Fig 2): ℓ2-ball (ε₂=0.5) ⊂ convex hull của ℓ1-ball ∪ ℓ∞-ball tại radii chuẩn (√(ε∞·ε₁)≈0.61) — train 2 norm cực biên cho ℓ2 "miễn phí" là ĐỊNH LÝ, không phải may mắn empirical. Weighting của ta TỰ TÁI-KHÁM-PHÁ hình học này (q_ℓ2→0 mà không được bảo trước). Differentiation: E-AT = manual geometric prior (chọn cứng ℓ∞+ℓ1); ta = LEARNED (tự hội tụ về cùng geometry) + per-sample + mechanism account.**
- **F4 (candidate — degeneration-to-AVG + signal misalignment):** AttackDRO++ (GroupDRO ℓ∞+ℓ2+ℓ1, seed-0 APGD) worst-∪ = **38.7 = TRÙNG KHÍT AVG 38.7**, NO-GO vs MSD (−3.8). Difficulty-aware DRO trên attack-groups **thoái hoá về averaging** cho worst-case. Cơ chế: q-by-loss (scalar/nhóm) **lệch khỏi** mục tiêu union — dồn vào ℓ∞ (loss cao nhất → ℓ∞ 46.2 **vượt** MSD 44.6) nhưng bỏ nhẹ **ℓ1 binding** (41.2, yếu nhất, thua MSD 46.7). ⇒ fix = **binding-aware signal** (CARD-3a group: q∝softmax(−robust_acc/τ); CARD-3b per-sample soft AVG↔MAX), KHÔNG phải floor/CVaR. Xác nhận kỳ vọng thesis: reweighting đẩy average, không đẩy worst-case. **[CARD-1 verdict 2026-07-04, recipe-controlled]: ΔDRO = P1 (39.7±0.7) − avg_frozen in-house (40.6±0.4 — files on disk, validated 2026-07-04) = −0.9 → theo decision rule CARD-1: loss-signal DRO NEUTRAL-TO-SLIGHTLY-HARMFUL dưới recipe control — F4 nâng từ 'degenerate về AVG' thành 'degenerate về DƯỚI AVG một chút'.**
- **F5 (candidate, 1-ckpt — verify trên ckpt khác, vd avg_frozen khi có):** weak-attack probe **ĐẢO NGƯỢC ranking độ khó norm**: probe PGD-20 nói weakest = ℓ∞ (50.3 < ℓ1 53.5), APGD nói weakest = **ℓ1** (41.2 < ℓ∞ 46.2); bias per-norm ℓ1 **+12.3pp** vs ℓ∞ +4.1 / ℓ2 +2.6 (seed-0 best ckpt, epoch 33). ⇒ mọi weighting calibrate bằng probe yếu **mis-target CẤU TRÚC** (không phải noise) — mở rộng F1 từ "overstate số" thành "đảo ranking". Hệ quả: (a) pre-registered prediction cho 3a pilot: ≈38.7, q dồn ℓ∞; (b) per-sample raw loss (3b) nhiều khả năng mang cùng norm-scale bias → prior cho zscore variant TĂNG (vẫn evidence-gated qua dist/* logs); (c) fix = CARD-3a-v2 val-calibrated q (APGD-CE trên val held-out). ℓ∞ = dày/đều (concentration .106); ℓ2 ≈ ℓ1 = thưa, dồn vào vùng salient (corr saliency +.72/+.86). Trục khó của union = ℓ∞ vs sparse.
- **F6 (candidate): binding norm là STATE-DEPENDENT — nó DI CƯ khi norm yếu được vá; weighting tốt phải track nó.** Evidence: (i) F5 — model P1 (ℓ1 bỏ đói): binding = ℓ1 (41.2 < ℓ∞ 46.2); (ii) 3a sau khi vá ℓ1 (47.1): binding flip → ℓ∞ (44.6); (iii) 3a-v2 val-APGD signal SẠCH trên chính model nó: weakest = ℓ∞ (.458 < ℓ1 .540) — đồng thuận độc lập; (iv) F2 — model mạnh (MSD/AVG) đều ℓ∞-bound. Giải thích vì sao q "sai hướng" của 3a hoạt động về cuối training: khi ℓ1 đã được kéo lên, dồn ℓ∞ LÀ đúng. Hệ quả thiết kế: signal phải đo lại trong training (online/every-k), không cố định từ đầu. **T-axis (sweep 4/4 + CARD-5 landed 07-04): T = núm trade ℓ∞↔(ℓ1,ℓ2); warm T → union ℓ∞-limited (F2-coherent); "cold beats hard" CHẾT (max_inhouse 43.9, ngoài band prereg phía cao) — worst-∪ gần-monotone lạnh→ấm 43.9→43.0→42.6→41.5→42.1. Related-work hook: T-dial = bản LÀM MỀM + PHÂN TÍCH của L_max term trong RAMP Eq. 2 (per-sample max được community endorse ở NeurIPS'24) → complementary, không competing.**
- **F7 (recipe confound quantified — BA evidence độc lập, 2026-07-04):** cùng "một method", chênh lệch qua recipe/nguồn: **AVG +1.9** (locuslab ckpt 38.7 vs avg_frozen in-house 40.6±0.4, files on disk) · **MAX +18.9** (locuslab ckpt 25.0 vs max_inhouse 43.9) · **xác nhận thứ 3 — external cross-validation: max_inhouse 43.9 @0.03 khớp C&H MAX retrain 44.0±0.7 ‡ @8/255 trong 1σ** ⇒ recipe ta tái tạo đúng số retrain của C&H; 19pp gap là artifact của CKPT locuslab; hard-MAX genuinely ~44 (prediction miss CARD-5 41.5–43.5 được giải thích: không phải eval error). Trộn nguồn là vô nghĩa; bảng 3-tier bắt buộc. CARD-7 msd_inhouse = điểm thứ 4, **hạ xuống OPTIONAL** (ta cite C&H 5-seed như RAMP làm).

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
- **G2′ (PIVOT gate, 2026-08-02):** CARD-8b signal? **>repro +0.5 (1 seed) → METHOD paper** (composition beats/approaches cluster) · **else → MECHANISM framing** (F1–F7 gap argument, materials còn nguyên). Quyết định lane tại đây, không trượt.
| **P4** | 09→15/10 | arXiv + code release public; CLIP multi-norm CHỈ nếu còn ≥4 tuần | **G4:** chuyển ưu tiên sang apps |
| **P5** | 15/10→15/12 | SOP (arc: negative result→rigor→mechanism→fix) + thư (xin 01/11) + email prof | **PhD deadline 01/12 & 15/12** |
- Đọc verdict P1: **>42.5** = verify-rồi-GO (nghi bug trước) · **38.7–42.5** = tốt, P2 đóng gap · **<38.7** = check ℓ2 starvation, mechanism story.

## 8 · ASSETS (cái gì ở đâu)
- **Repo `ATTACKDRO`** (PC, WSL2): configs/ (base+pgd_at+attackdro) · src/robustdro/ (data, models, attacks[linf,l2,l1-topk], training[GroupDRO], eval[eval_union+AA wrapper], utils[stats,wandb]) · scripts/ (train, evaluate, validate_baseline, visualize_attack, perturbation_geometry, union_mask_test) · external/robust_union/ (6 ckpt official) · CLAUDE.md + WORKFLOW_RULES.md + PROGRESS.md.
- **Workflow:** MacBook → SSH (Tailscale) → PC 5070ti (Blackwell sm_120, **torch cu128 bắt buộc**, venv phải activate); long run trong tmux riêng; W&B (user kietna); Colab Pro backup.
- **Research Hub** (React + persistent storage): Overview/Roadmap/Board/Papers/Runs/Notes/Links + module Docs, Timeline (deadline-anchored), Assistant (BYOK AI); deploy kit Vite+Vercel (storage.js, ai.js, DEPLOY.md).
- **Docs:** Research_Proposal_UnionDRO.docx (gửi thầy+mentor, có mục Q1–Q6) · Roadmap_v2_DeadlineAnchored.md · roadmap v1.
- **Reading list** ~18 papers 4 tier (Madry, TRADES, Tramèr&Boneh, MSD, AutoAttack, E-AT / GroupDRO, GEORGE, DFR, CVaR / NCAT, PROTECTOR, CURE / TeCoA, FARE).

## 9 · OPEN / PENDING (synced 2026-07-04 08:40 — chi tiết: docs/SYNC_SNAPSHOT.md)
- [x] Bảng in-house hiện có: P1 39.7±0.7 · avg_frozen 40.6±0.4 (Colab, files validated) · 3a 42.3/42.1 (s2 sắp) · T025 **43.03±0.05** · max_inhouse 43.9 (1s) · T-sweep 43.0/42.6/41.5/42.1.
- [x] **CLAIM LADDER REVISED (Kiet 2026-07-04, sau full RAMP read): "match/beat MSD" BỎ.** C1 = mechanism F4–F6 (unique) · +F7 eval corrections · C2 = simple cheap weighting ĐẠT cụm SOTA 43.9–44.6, không vượt. Baselines from-scratch = ‡ cited 5-seed (C&H + RAMP), KHÔNG retrain — đúng methodology RAMP.
- [🔄] **RAMP protocol-verify seed-0 ĐANG CHẠY (Kiet tự launch 07-04 08:25, đảo quyết định P3-đóng):** official cmd (80ep, --kl --max --gp --lbd 5), upstream be4971f + 1-line patch (`import copy`); eval kép armed: apgd + standard AA × {0.03, 8/255}. ETA ~14h+ (chia GPU). Ckpt: `external/RAMP/trained_models/RAMP_beta_0.5_lbd_5_0/ep_80_0.pth`.
- [🔄] finalpipe: bindaware_s2 ep49/50 → T05 s1,s2 → std-AA pack (finalists+MSD; avg_frozen cần Kiet bỏ `avg_frozen_s0_best.pt` vào checkpoints/). p7msd (msd_inhouse, giờ OPTIONAL) fire sau finalpipe.
- [🔒] CARD-6 eat_compose — gated sau mandatory queue. CARD-2 deprioritized · CARD-4 OFF (F3) · 3a-v2 PARKED.
- [x] **Trait-vs-state diagnostic RAN (CPU, 2026-07-04, `scripts/dev/binding_trait_vs_state.py`):** population binding = **STATE-like** — bắt đầu uniform (entropy 1.0) → sụp về ℓ∞ (freq .65–.87, ℓ2→0 khớp F3), decisiveness tăng .016→.27, **max_inhouse migrate ℓ1→ℓ∞** (F6 population-level), drift TV .30–.51. ⇒ củng cố F6 "measure online" + CARD-8b Opt A (online T-softmax > static pre-weight). ⚠ **Giới hạn thật:** dumps là first-batch của loader `shuffle=True`, KHÔNG có sample-index → **per-sample trait-stability KHÔNG đo được** từ dữ liệu này; muốn "predictive per-sample weighting" novelty cần instrument **fixed probe batch + indices** (1 dòng trainer + 1 re-run GPU) — gated.
- [ ] Full AA `standard` chưa test timing 10k.
- [ ] Chọn workshop sau list NeurIPS (11/07) · Gửi proposal thầy+mentor (giờ có đủ số P2 3-seed).

## 10 · UPDATE DISCIPLINE (để file này sống)
- Cập nhật khi: gate đóng · số mới validated · quyết định mới lock · idea bị refute · asset mới. Sửa tại chỗ + đổi "Last updated".
- Giữ ≤ ~150 dòng: cũ/hết đúng → xoá hoặc nén 1 dòng vào mục liên quan. Chi tiết lịch sử thuộc về PROGRESS.md, không phải đây.
- Phân công: Claude Code cập nhật §4–§9 sau mỗi milestone; Kiet duyệt §1–§3 (identity/decisions).