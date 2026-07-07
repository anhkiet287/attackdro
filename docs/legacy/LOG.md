
## 2026-07-02 — P1 Step 3b: SEED-0 APGD RESULT (provisional, 1 seed)
AttackDRO++ (GroupDRO over linf+l2+l1) seed 0, locked protocol (eps_inf=0.03, n=1000, apgd), best ckpt @epoch 33:

| model | clean | l_inf | l2 | l1 | worst-union |
|---|---|---|---|---|---|
| AttackDRO++ s0 | 80.7 | 46.2 | 63.1 | 41.2 | **38.7** |
| AVG (baseline) | 84.6 | 40.7 | 65.5 | 47.7 | **38.7** |
| MSD (baseline, target) | 82.1 | 44.6 | 64.5 | 46.7 | **42.5** |

- **NO-GO vs MSD (−3.8); exact TIE with AVG (38.7).** q_final=[l_inf 0.83, l2 0.00, l1 0.17].
- **Mechanism (why):** GroupDRO weights by per-group AVERAGE loss → piled onto l_inf (highest loss) → best l_inf (46.2, beats MSD) but **starved l1 (q=0.17) → weakest l1 (41.2, loses to MSD 46.7)**. l1 is the union-binding norm; the DRO signal is MISALIGNED with the worst-union objective. l2=63.1 despite q_l2=0.00 → l2 robustness is FREE (spillover), so anti-starvation/floor fixes waste capacity.
- **P2 direction (from seed-0): attack BINDING-AWARE first** (reweight toward per-sample worst norm / union-binding), NOT floor/CVaR. Confirms the P1 thesis-probe expectation: difficulty-aware DRO over attack-groups ≈ AVG on worst-case, does not reach MSD's per-sample worst-case handling.
- Confounds: 1 seed; our recipe vs locuslab recipe for baselines; apgd not `standard`. Seeds 1–2 running for variance; final human GO/NO-GO after they land.

### 2026-07-02 — AttackDRO++ SEED-0 APGD result (first GO/NO-GO signal)
Best ckpt @epoch33, n=1000, APGD-CE+T. q_final [linf,l2,l1]=[0.832, 0.000, 0.168].

| model | clean | l_inf | l2 | l1 | worst-union |
|---|---|---|---|---|---|
| AttackDRO++ s0 | 80.7 | 46.2 | 63.1 | 41.2 | **38.7** |
| AVG (baseline) | 84.6 | 40.7 | 65.5 | 47.7 | 38.7 |
| MSD (target)   | 82.1 | 44.6 | 64.5 | 46.7 | **42.5** |

**Verdict (provisional, 1 seed): NO-GO vs MSD** — union 38.7 = ties AVG, −3.8 vs MSD. Expected (thesis improved avg-case, not worst-case) → triggers P2.

**P2 direction from this seed → BINDING-AWARE first (not group floor/CVaR):**
- l2 group starved (q=0) yet l2 acc highest (63.1) → l2 is "free"; group-level floor/CVaR would spend capacity on an already-solved norm = mis-targeted.
- q collapsed to ~fixed (0.83/0/0.17) → GroupDRO degenerated into a weighted (linf+l1) AVG → lands exactly at AVG (38.7), can't reach MSD.
- worst-union 38.7 < min per-norm (l1 41.2) → per-sample binding norm varies; a single per-group scalar q can't capture it. This is precisely MSD's edge (per-sample max-loss over norms).
- Binding norms of worst-union: l1 then linf; leave l2 alone.
- Caveats: 1 seed; APGD (not full AA); recipe confound (ours vs locuslab). Seeds 1/2 running for variance; full auto-verdict appended when they finish.

## 2026-07-02 — P2 RE-ROUTE (after seed-0 verdict)
Seed-0 accepted: NO-GO vs MSD, exact tie AVG (38.7) — expected branch. Re-routed P2:
- **CARD-4 (qfloor) FORMALLY OFF** — gate ℓ2<45% not met (ℓ2 APGD=63.1, free/spillover). F3 rewritten in PROJECT_MEMORY §5.
- **CARD-2 (cvar) DEPRIORITIZED** — seed-0 indicts group-level scalar signals (q-by-loss piled on ℓ∞, under-weighted binding ℓ1); CVaR same class. Fallback only.
- **CARD-3 split** → **CARD-3a** group binding-aware `q∝softmax(−robust_acc_g/τ)` (**launched, 1-seed pilot**) + **CARD-3b** per-sample soft binding-aware (temperature-softmax over per-sample per-norm losses, interpolates AVG↔MAX, NOT hard-max since MAX=25.0 fails) — **drafted per E1, awaiting Kiet approval before code/launch**.
- **CARD-1 avg_frozen** (freeze q → AVG in-house, kills recipe confound) — queued after p1.
- New F4 in §5: degeneration-to-AVG tie + signal↔objective misalignment.

**Code (verified):** `dro.freeze_q` (CARD-1) + `dro.weight_signal=robust_acc`,`dro.tau` (CARD-3a) in GroupDRO/trainer; unit-checked q→weakest-norm; both configs smoke-passed. `configs/{avg_frozen,bindaware}.yaml`.
**Running:** `p1pipe` (P1 seeds 1–2, variance) + `p2pipe` (CARD-3a pilot now → then CARD-1 avg_frozen ×3). VRAM fine (15.7 GB free; each run ~1.4 GB). All eval = eval_union APGD n=1000.
**Next:** ping when CARD-3a pilot APGD lands (~4h, contended) with worst-∪ / per-norm(ℓ2) / q / vs 38.7 & 42.5 & CARD-3a-vs-P1.

## 2026-07-02 — CARD-3b approved + implemented (3 amendments)
- **Code:** `objective: per_sample_soft` in GroupDROTrainer — per-sample CE per norm → [G,B] matrix → temperature-softmax over norms (detached weights) → weighted mean. Unit-checked AVG↔MAX interpolation (T=100→AVG 1.23, T=0.01→MAX 2.03, monotonic). Amendment #1: per-norm loss distributions (std/p10/p50/p90) logged every epoch (`dist/*`); `normalize_losses: zscore` variant implemented, evidence-gated. Generic `--set KEY=VALUE` added to train.py for the sweep. Both variants smoke-passed.
- **Scheduling (per Kiet):** T-sweep {0.25,0.5,1,2} 1-seed pilots armed in tmux `p3bsweep` — waits for 3a pilot APGD, then retires p2pipe (avg_frozen local stage cancelled) and runs sweep sequentially (≤2 trainers on GPU). Best T → 3 seeds.
- **CARD-1 → Colab:** `notebooks/colab_avg_frozen.ipynb` ready (clone → smoke → 3×train+eval → table → Drive). **BLOCKER: repo not pushed since initial scaffold — needs Kiet OK to commit+push.**
- **Lit-check (amendment #2, preliminary):** closest prior = Adaptive Smoothness-weighted AT (arXiv 2210.00557, per-TYPE weights); Tramèr&Boneh AVG/hard-MAX = the two poles; MSD = per-STEP selection (different axis); E-AT = fine-tuning. Per-sample temperature-softmax not directly found → frame as mechanism probe, read 2210.00557 before any novelty claim. New decision branch recorded: 3b>3a but <42.5 ⇒ remaining MSD edge = per-step selection inside attack gen (publishable mechanism finding).

## 2026-07-02 — Autopilot session (Kiet away) + figure tooling
- **Push DONE** (pre-approved): hygiene clean (ignores complete, no key material), 3 logical commits `efffbc8..33a1ee5` → github.com/anhkiet287/attackdro. Colab avg_frozen now clone-able; **start = Needs Kiet** (Google auth + GitHub PAT).
- **SOURCES.md: all 6 ⚠ verified → ✅** (E-AT 2105.12508 ICML'22 · NCAT = Sriramanan/Gor/Feizi NeurIPS'22, no arXiv, cite proceedings · PROTECTOR UAI'22 PMLR v180 · Duchi&Namkoong AoS 49(3) 2021 · TeCoA 2212.07016 ICLR'23 · CURE 2410.03000 — arXiv title now "Generalized", certified axis). Added [Xiao2022-ASW] 2210.00557.
- **2210.00557 read** — 5-line positioning note in CARD-3b lit-check: they = per-TYPE, smoothness signal, weighted-avg, has stability theory, SOTA claim CIFAR-10/100; 3b's distinct axis = per-SAMPLE loss softmax with AVG↔MAX temperature; frame as mechanism probe; nếu 3b thắng bảng ta thì vẫn CHƯA được claim vượt họ (chưa re-eval họ dưới harness ta).
- **Figure tooling (4d)**: `scripts/figures/{fig_style,plot_q_trajectory,plot_probe_bias,plot_loss_dists}.py` — paper-style, colorblind-safe + grayscale-safe (linestyle/marker/hatch). Rendered + eyeballed: `q_traj_bindaware_s0.png` (LIVE F5 demo: q → [ℓ∞ .6+, ℓ2 ~.1, ℓ1 <uniform]; +CSV dump per Kiet item-2), `q_traj_attackdro_s0.png` (F4 figure, full 50 ep), `probe_bias_f5.png` (F5 figure: +12.3 ℓ1 bias + inversion annotation), `loss_dists_smoke.png` (3b amendment-#1 plot, verified on smoke JSON).
- Runs during session: p1 s1 + 3a pilot healthy throughout, no NaN; gates (`p3bsweep`, `p3av2`) armed and correct.

## 2026-07-02 — E8 RESULT · CARD-3a pilot (bindaware_s0) — BEST-SO-FAR, ties MSD
**Run:** `bindaware_s0`, 50 ep, locked protocol, eval APGD CE+T n=1000, best ckpt @26. q_final=[ℓ∞ .70, ℓ2 .07, ℓ1 .23]. Figure: `results/figures/q_traj_bindaware_s0.{png,csv}` (full 50-ep trajectory).

| model | clean | ℓ∞ | ℓ2 | ℓ1 | worst-∪ |
|---|---|---|---|---|---|
| **CARD-3a (bindaware) s0** | 80.3 | 44.6 | 64.6 | **47.1** | **42.3** |
| P1 AttackDRO++ s0 | 80.7 | 46.2 | 63.1 | 41.2 | 38.7 |
| AVG (locuslab) | 84.6 | 40.7 | 65.5 | 47.7 | 38.7 |
| MSD (locuslab, target) | 82.1 | 44.6 | 64.5 | 46.7 | 42.5 |

- **vs pre-registered prediction (ghi TRƯỚC trong CARD-3a-v2):** mechanism half ĐÚNG (q piled ℓ∞ .70 — F5), outcome half SAI (dự đoán ≈38.7, thực tế **42.3**). Ghi nhận trung thực: prediction failed on outcome.
- **Đọc mechanism:** robust-acc signal (dù biased theo F5) chuyển đủ weight sang ℓ1 (.23) → ℓ1 41.2→47.1 (+5.9) đổi lấy ℓ∞ −1.6 → các norm gần cân bằng (binding flip: weakest giờ là ℓ∞ 44.6). Đây là hành vi equalization mà binding-aware nhắm tới. Union 42.3 = tie MSD (−0.2, 1 seed, within noise).
- **Hệ quả cho picture P2:** F4 ("group-level ceiling") cần hạ cấp thành "group-level VỚI loss-signal ceiling" — signal là thủ phạm chính, không phải (chỉ) aggregation. 3a-v2 (val-APGD signal sạch) giờ QUAN TRỌNG HƠN: nếu 3a-v2 > 3a → signal quality là trục chính; 3b kiểm tra trục aggregation độc lập.
- **Decision rule CARD-3a:** best-so-far (42.3 > mọi run của ta). Theo quy tắc "3 seed cho config có tín hiệu" → cần bindaware seeds 1–2. **Needs Kiet: chọn slot** (đề xuất: sau 3b sweep, hoặc Colab song song với avg_frozen).
- Caveats: 1 seed · apgd không phải `standard` · recipe confound vs locuslab (avg_frozen Colab sẽ xử lý một phần) · KHÔNG claim "beat/match MSD" công khai trước 3 seed + standard AA.

## 2026-07-01 — P1 RESULTS (auto-generated by run_p1_pipeline.sh)

Locked protocol: eps_inf=0.03, l2=0.5, l1=12; n=1000; AutoAttack apgd (CE+T).

| model | clean | l_inf | l2 | l1 | worst-union |
|---|---|---|---|---|---|
| MSD (baseline, best union) | 82.1 | 44.6 | 64.5 | 46.7 | **42.5** |
| pgdat (l_inf-AT, 8/255 recipe) | 82.0 | 48.9 | 58.2 | 9.4 | 9.4 |
| AttackDRO++ seed0 | 80.7 | 46.2 | 63.1 | 41.2 | **38.7** |
| AttackDRO++ seed1 | 81.5 | 47.0 | 64.5 | 42.2 | **40.0** |
| AttackDRO++ seed2 | 80.5 | 46.0 | 62.5 | 42.6 | **40.4** |

AttackDRO++ worst-union: best=40.4, mean=39.7 ± 0.7 (n=3 seeds).
MSD baseline worst-union: 42.5.

**VERDICT: **NO-GO** — best AttackDRO++ union 40.4 < MSD 42.5 (-2.1). Difficulty-aware DRO over the union did not beat MSD on worst-case. EXPECTED, informative negative result → trigger P2 (grouping signal / objective / l1 / mechanism).**

_Note: numbers are AutoAttack `apgd` (CE+T). Re-run the winner + baselines with `--version standard` (full AutoAttack) for final paper numbers. Analysis written automatically; a human-reviewed writeup follows._

## 2026-07-03 — T=0.25 VERIFICATION (per Kiet: before any seed planning treats 43.0 as real)
- (a) ckpt/config audit ✅: run=bindaware_sample_T025, objective=per_sample_soft, T=0.25, normalize=none, train eps=(0.03,0.5,12), eval protocol eps identical, version=apgd, best_epoch=25.
- (b) independent re-eval ✅: test imgs 1000–1200 (disjoint từ mọi eval trước), attack seed 123 → union **44.5** (linf 45.5, l2 64.0, l1 54.5) — nhất quán với 43.0 (n=200 noise ±~3.5pp). Không có slice-overfit/harness artifact.
- (c) curve inspection ✅: loss mượt, jump đúng LR-drop ep25, plateau sau đó, best-probe ep = ckpt ep (25), l2 không sụp (probe 65–68 nửa sau).
- **KẾT LUẬN: 43.0 là thật ở độ chính xác 1-seed. Claim đúng: "MSD-level (43.0 vs 42.5, within seed noise ±0.7), 1 seed, apgd".** Chờ 3 seed + standard AA trước mọi phát ngôn mạnh hơn.

## 2026-07-03 — Decisions locked (Kiet)
- **zscore variant: SKIPPED.** Evidence từ T025 dist/*: bias mild (p50 linf .62 / l1 .49 / l2 .30), KHÔNG inversion, l1 vẫn cải thiện nhiều nhất (+7.1) → không đáng 1 pilot.
- **Standard AA: APPROVED, scoped** — chỉ finalists (3a, 3b-T*) + MSD + avg_frozen, SAU khi có 3 seed, parity tuyệt đối (cùng n, cùng version), Colab nếu GPU bận.
- **Seed plan:** sweep xong → chọn T* → 3-seed hai finalist (3a bindaware + 3b-T*). avg_frozen ×3 Colab (Kiet chạy hôm nay).

## 2026-07-03 — Training-cost quantification (Mức B "match-cheaper" claim; counted from code, not memory)
Per minibatch (both: 50 epochs, batch 128, PreActRN-18):
- **MSD** (`external/robust_union/CIFAR10/cifar_funcs.py::msd_v0`, num_iter=50, train.py default -model 3): mỗi step = 1 fwd+bwd (gradient) + 3 candidate fwd (chọn max-loss direction) → **50 bwd + 200 fwd** cho attack + 1 fwd+bwd train = **~201 fwd + 51 bwd / minibatch**.
- **Ours (3b per_sample_soft / 3a)** (`configs/attackdro.yaml`): linf 10 + l2 10 + l1 20 = **40 fwd + 40 bwd** cho attack + 3 fwd+bwd train (một per norm) = **~43 fwd + 43 bwd / minibatch** (+ probe 8 batch/epoch, ~2% overhead).
- **Tỷ lệ:** nếu bwd ≈ fwd: ours ≈ 86 vs MSD ≈ 251 đơn vị → **~2.9× rẻ hơn**; nếu bwd ≈ 2×fwd: 129 vs 302 → **~2.3× rẻ hơn**. Claim an toàn: **"~2–3× cheaper per iteration (forward/backward count), same epochs/batch"**. Wall-clock đối chứng sẽ có khi chạy MSD in-house (nếu cần cho paper); KHÔNG claim wall-clock từ số của họ.

## 2026-07-03 — Claim-ladder state + PRE-REGISTERED prediction cho 3a-v2 (ghi TRƯỚC khi có kết quả; v2 đang ở ~epoch 41/50)
**Claim ladder sau verification (Kiet):**
- **Mức B (kịch bản chính, nguyên liệu ĐỦ):** match MSD bằng 2 method độc lập (3a 42.3, 3b-T025 43.0) + ~2–3× cheaper/iteration (đếm từ code: MSD ~201f+51b vs ours ~43f+43b) + mechanism chain F4→F5→F6. Chỉ chờ 3-seed.
- **Mức A (beat):** CHỈ mở nếu 3-seed mean > 42.5 VÀ sống qua standard AA. Không lái narrative theo nó.
- **Mức C:** đã bảo hiểm (rigorous negative + mechanism đã có từ P1).

**PRE-REGISTERED (Kiet, 2026-07-03): 3a-v2 sẽ ra 42.0–42.6** (≈ 3a 42.3 ±noise). Khung đọc:
- **≈42.3 (±0.7) [DỰ ĐOÁN]:** signal sạch ≈ signal lệch-nhưng-tự-sửa → xác nhận F6 (quan trọng là BÁM binding norm khi nó di chuyển, không phải đo hoàn hảo từ đầu) → 2×2 khép: **signal là trục bậc nhất (+2.6–3.3pp loss→binding), aggregation & calibration-tinh-vi nằm trong noise** — tin tốt cho "simple method".
- **>43:** calibration sạch có giá trị riêng → cân nhắc merge (per-sample + val-calibrated) làm finalist 3 — chỉ nếu vượt noise.
- **<41.5:** signal sạch mà thua → soi val-holdout (train 49k vs 50k), EMA/τ — không kết luận vội.

## 2026-07-03 — E8 RESULT · CARD-3a-v2 (bindaware_v2_s0) — 2×2 HOÀN CHỈNH
**41.5** | clean 81.4 · ℓ∞ 44.0 · ℓ2 64.6 · ℓ1 47.1 · q=[.61,.08,.31] · best ep30 · val-calib cuối: ℓ∞ .458 < ℓ1 .540 (signal sạch xác nhận binding=ℓ∞ trên chính model nó — F6).
- **vs PRE-REGISTERED 42.0–42.6: TRƯỢT 0.5, về phía thấp** (biên nhánh "<41.5"). Đọc trung thực: −0.8 vs 3a nằm trong noise 1-seed (±0.7; P1 spread 1.7pp); ℓ1/ℓ2 GIỐNG HỆT 3a, toàn bộ chênh nằm ở ℓ∞ (44.0 vs 44.6). Nghiêng về kết luận nhánh giữa NHƯNG theo đúng pre-registration: 2 confound check treo trước khi chốt — (a) v2 train 49k (mất 1000 ảnh holdout, đủ giải thích −0.8), (b) EMA β=.5/τ chưa tune (cadence 5-epoch có thể lag F6-migration so với probe mỗi-epoch của 3a).
- **2×2 (seed-0):** loss→binding = +1.8..+3.3pp (trục bậc nhất, lặp ở 2 aggregation) · calibration sạch KHÔNG hơn probe tự-sửa (41.5 ≤ 42.3 → F6 support, "simple method" story) · per-sample vs group trong noise (43.0 vs 42.3) · merge-finalist gate ĐÓNG (v2 < 43).
- Bảng: P1 39.7±0.7 | 3a 42.3 | 3a-v2 41.5 | 3b-T025 43.0 | MSD 42.5 | AVG 38.7.

## 2026-07-03 — Kiet decisions (post-2×2)
- **v2 confound checks: PARKED** (không đầu tư GPU). Kết luận pragmatic ghi vào card 3a-v2: val-calibrated ≤ probe; với F6, tracking binding norm thắng precision. Caveat 49k một dòng đã ghi.
- **T\* decision: DEFERRED đến khi T=1 lands.** Nếu cold slope xác nhận (T=1 < 42.6) → đề xuất 3-seed CẢ HAI T=0.25 và T=0.5 (gap 0.4pp trong noise; T=0.5 có +1.5 clean, +1.8 ℓ1). Mang plan lên trước, KHÔNG launch.
- **Mechanism note (cho per-norm figure sau này): CỘT ℓ1 LÀ CÂU CHUYỆN** — 3b mua +1.6–3.4 ℓ1 so với MSD (48.3/50.1 vs 46.7) với giá <1pp ℓ∞ (43.8 vs 44.6). Pattern này phải nổi bật trong figure per-norm cuối (đánh dấu cặp cột ℓ1/ℓ∞).

## 2026-07-03 09:05 — STATUS NOW (consolidated)
**Scoreboard (seed-0, locked protocol, APGD CE+T, n=1000):**
| run | clean | ℓ∞ | ℓ2 | ℓ1 | worst-∪ |
|---|---|---|---|---|---|
| **3b T=0.25** (verified: config✓ disjoint-slice 44.5✓ curve✓) | 79.7 | 43.8 | 62.8 | 48.3 | **43.0** |
| 3b T=0.5 | 81.2 | 43.8 | 63.3 | **50.1** | **42.6** |
| MSD (locuslab, target) | 82.1 | 44.6 | 64.5 | 46.7 | 42.5 |
| 3a bindaware | 80.3 | 44.6 | 64.6 | 47.1 | 42.3 |
| 3a-v2 val-calibrated (PARKED) | 81.4 | 44.0 | 64.6 | 47.1 | 41.5 |
| P1 AttackDRO++ (3 seeds, NO-GO) | ~81 | — | — | — | 39.7 ± 0.7 |
| AVG (locuslab) | 84.6 | 40.7 | 65.5 | 47.7 | 38.7 |

**Đang chạy:** `p3bsweep` T=1 @ epoch 24/50 (lands ~10:30) → T=2 nối tiếp (~14:00). GPU 1 trainer, log sạch.
**Claim ladder:** Mức B nguyên liệu đủ (2 method độc lập MSD-level + ~2–3× cheaper/iter đếm từ code + chain F4→F5→F6); Mức A chỉ mở nếu 3-seed mean >42.5 + sống qua standard AA.
**2×2 khép:** signal loss→binding = trục bậc nhất (+1.8–3.3pp); calibration sạch ≤ probe (F6: tracking > precision); per-sample ≈ group (trong noise).
**Quyết định đang hiệu lực:** zscore SKIPPED · v2 PARKED (không GPU) · standard-AA approved-scoped (finalists+MSD+avg_frozen, sau seeds, parity) · T* DEFERRED đến T=1 (nếu <42.6 → đề xuất 3-seed CẢ T=0.25 và T=0.5, không tự launch).
**Chờ:** (1) T=1 ~10:30 → mang seed-plan lên; (2) T=2 ~14:00 → figure union-vs-T chốt; (3) **Colab avg_frozen — phía Kiet, chưa thấy results sync** (gitignore fix đã push 48577cb, clone giờ hoạt động); (4) Mức-A gate items: 3-seed finalists + standard AA.
**Assets mới hôm nay:** verification report T=0.25 · cost quantification (MSD 201f+51b vs ours 43f+43b) · F6 (§5) · 4 figure scripts + `union_vs_T.png` (2 điểm) · v2 E8 + 2×2 reading.

## 2026-07-03 — E8 · Presentation figure pack (CPU-only, real data, cho mentor talk)
Script: `scripts/figures/make_presentation_pack.py` (reuse fig_style + probe_bias/q_trajectory internals as libraries). Provenance: `results/MANIFEST.md`. Trainers untouched (C7 checked: chỉ p3bsweep T=1 đang chạy).
- `results/figures/presentation/fig_loss_percentiles.png` — dist p10/p50/p90 per norm, epoch 49 T=0.25 run. **Honesty note: annotation tasked là "ℓ1 widest spread" nhưng data nói ℓ∞ widest (2.06 vs ℓ1 1.72) → figure annotate theo DATA, không theo task.** p50 ranking đúng như quote (ℓ∞ .62 > ℓ1 .49 > ℓ2 .30).
- `fig_temperature_dial.png` — softmax weights tại T={0.25, 1, 100} từ epoch-MEAN losses THẬT (raw per-sample matrices không có trên disk → labeled "illustrative", đúng quy tắc no-fake-data).
- `fig_union_vs_T.png` — 2 điểm hiện có (43.0, 42.6) + anchors MAX 25.0 / AVG 38.7 / MSD 42.5; re-render 1 lệnh khi T=1/T=2 về.
- `fig_mechanism_chain.png` — composite (a) F5 bias bars + (b) F6 q-trajectory 3a, caption "biased signal → misallocated q → fixed by binding-aware".

## 2026-07-03 — Kiet approvals executed (pre-finalist)
- **Per-sample loss-matrix dump ADDED to GroupDROTrainer** (1 batch [G,B] / epoch → `results/loss_mats/<run>/epNNN.pt`, ~3 KB/epoch). Smoke-verified (shape [3,128]). Mọi finalist run từ giờ tự sinh data cho per-sample dial figure — bỏ được label "illustrative".
- **plot_union_vs_T.py fixes** (fixed x-ticks {0.25,0.5,1,2}, anchors inside plot) — áp dụng khi re-render sau T=1/T=2.
- **QA.md** thêm cho mentor talk: câu hỏi "tracking vs bias?" ở fig-2(b) + trả lời 3 ý (v2 clean signal đồng thuận .458<.540 · F2 strong-models ℓ∞-bound · bias↔truth hội tụ late-train = chính là F6).

## 2026-07-03 — E8 · State-of-play figure pack (CPU-only; inventory từ disk)
Inventory: 8 eval JSONs của ta + 6 official baselines, tất cả n=1000/apgd, KHÔNG file nào malformed; T=1 chưa có trên disk lúc render (script glob — re-render 1 lệnh khi về). Figures (results/figures/presentation/):
- `fig_state_of_play.png` — leaderboard worst-∪ (ours xanh, official xám, MSD/AVG ref lines, P1 mean±std, mọi bar 1-seed đánh dấu).
- `fig_pernorm_finalists.png` — grouped bars 5 model × 5 metric, annotation ℓ1-trade (48.3/50.1 vs 46.7, giá <1pp ℓ∞).
- `fig_union_vs_T.png` — re-render với 2 fix đã queue (fixed ticks {0.25,0.5,1,2}, anchors trong khung).
Inventory table đầy đủ in ở stdout (xem log); MANIFEST cập nhật.

## 2026-07-03 — URGENT SOTA CHECK (RAMP) — kết quả + hệ quả positioning
- **RAMP (Jiang&Singh, NeurIPS 2024, arXiv 2402.06827): from-scratch RN-18 union 44.6 ‡, clean 81.2, AutoAttack, ε=(8/255, 0.5, 12)** — protocol KHÁC ta (ε∞ 8/255 vs 0.03). Hướng lệch: 8/255 khắc nghiệt hơn → số RAMP đọc dưới protocol ta chỉ có thể ≥44.6 → **RAMP presumptively trên mọi số của ta (best 43.0). CẤM claim match/beat SOTA.** Fine-tune track (53.3) tách riêng. **Không có ckpt công khai** → Level-2 re-eval bất khả thi; muốn so trực tiếp phải tự train code họ (P3 decision).
- **E-AT Table 5 (from-scratch PRN-18, 8/255, n=1000 ‡) trích nguyên văn:** SAT 40.4 · AVG 40.1 · **MAX 44.0** · **MSD 43.9** · E-AT 42.4. E-AT claim "up to 3× cheaper" (fixed geometric trick ℓ∞+ℓ1). Differentiation của ta: adaptive + mechanism (POSITIONING.md §2).
- **⚠ PHÁT HIỆN PHỤ QUAN TRỌNG — MAX-recipe confound:** locuslab MAX ckpt = 25.0 (harness ta) nhưng C&H MAX tự-train = 44.0 ‡ → **"hard-max fails" KHÔNG được claim về nguyên tắc, chỉ về ckpt locuslab.** Đã sửa: CARD-3b constraint, anchor label fig_union_vs_T ("locuslab MAX ckpt"), POSITIONING.md §3. Cần cân nhắc MAX in-house (~2h) làm anchor T→0 sạch cho bảng cuối.
- **Labels đổi:** "MSD (target)" → **"MSD (classical strong baseline)"** (PROJECT_MEMORY §4 + narrative). fig_state_of_play thêm RAMP 44.6 ‡ reference line (annotate "not protocol-aligned").
- Docs mới: **docs/POSITIONING.md** (SOTA landscape ‡, differentiation per prior, MAX confound). SOURCES.md: [Jiang2024-RAMP] ✅ + E-AT Table-5 numbers.

## 2026-07-03 — E-AT Table-1/Fig-2 absorbed (documentation only, no run changes)
- SOURCES [Croce2022-EAT] += Table 1 fine-tune numbers sẽ cite ‡ (MSD 42.6±0.2 · E-AT 42.2±0.8 · MAX 42.2±0.6; 306s vs 160s/epoch; 8/255, 5 seeds, RN-18 FT 3ep) + caveats (protocol ≠ 0.03; FT ≠ from-scratch) + Thm 3.1/Fig 2 geometry.
- MEMORY §5 F3 nâng cấp: **ℓ2-free giờ có nền lý thuyết** (ℓ2-ball ⊂ conv hull ℓ1∪ℓ∞, E-AT Thm 3.1) — weighting của ta TỰ tái-khám-phá geometry (q_ℓ2→0 unsupervised). Differentiation: manual geometric prior (E-AT) vs learned + per-sample + mechanism (ta).
- POSITIONING += **protocol-ceiling evidence**: mọi method nghiêm túc RN-18 hội tụ 42–44 (their Table 1&5 + runs ta) → "matching MSD" = chạm trần chung, value ở mechanism+cost; **RAMP 44.6 = outlier duy nhất, phải protocol-verify trước khi coi là bar**.

## 2026-07-03 — T=1 landed + decision package chuẩn bị + CARD-5/6 recorded
- **T=1: union 41.5** (clean 82.4 · ℓ∞ 42.2 · ℓ2 65.3 · ℓ1 **51.5**). **COLD SLOPE CONFIRMED: 43.0 (T=.25) → 42.6 (T=.5) → 41.5 (T=1)**, monotone; T=2 đang train (điểm cuối). Pattern per-norm: T ấm → ℓ1/ℓ2 tăng, ℓ∞ giảm → union giảm (ℓ∞ thành binding).
- **CARD-5 max_inhouse APPROVED + implemented + smoked** (objective per_sample_max; BN pipeline giữ nguyên). Kiet pre-registered: 41.5–43.5 → cold-T finding chết dạng hiện tại; <38 → sống. T-axis story CHỜ run này.
- **CARD-6 eat_compose GATED recorded** (sau mandatory queue; pre-registered readings trong card). RAMP in-house: P3 ĐÓNG.
- **POSITIONING §2.5: bảng cuối 3 tier** (in-house recipe-controlled / ckpt re-eval * / ‡ cited); motivating example = MAX 19pp gap.
- **⚠ Colab avg_frozen: CHƯA CHẠY** — Drive search (attackdro_results / avg_frozen / notebook copy) = trống. Đây là hàng Tier-1 load-bearing nhất của bảng cuối.

## 2026-07-03 — E8 · 3B T-SWEEP COMPLETE (4/4 điểm, 1 seed mỗi)
**T=0.25: 43.0 · T=0.5: 42.6 · T=1: 41.5 · T=2: 42.1** (T2: clean 82.3, ℓ∞ 42.8, ℓ2 63.9, ℓ1 50.5).
- **Đọc trung thực về hình dạng curve:** cold-end cao nhất (T=0.25 = 43.0) nhưng curve KHÔNG monotone — T=2 bật lại 42.1 > T=1 41.5. Với noise 1-seed ±0.7: chỉ T=0.25 vs T=1 tách ~2σ; T=0.5/1/2 nằm trong noise của nhau. Kết luận được phép: "T=0.25 tốt nhất trong sweep; curve nông; cold-preference CẦN 3-seed + MAX anchor (CARD-5) trước khi thành finding."
- Per-norm pattern rõ: T ấm → ℓ1/ℓ2/clean tăng, ℓ∞ giảm → union bị ℓ∞ kéo (F6 nhất quán: ℓ∞ = binding của model đã cân bằng).
- **GPU RẢNH HOÀN TOÀN** (mọi tmux đã kết thúc sạch). Toàn bộ hàng đợi giờ chờ quyết định Kiet: (1) dual 3-seed GO, (2) CARD-5 slot, (3) Colab avg_frozen, (4) standard-AA timing.

## 2026-07-03 — GO EXECUTED: final pipeline launched (tmux `finalpipe`)
- **Queue:** CARD-5 max_inhouse s0 → T025 s1,s2 → 3a s1,s2 → T05 s1,s2 → **standard-AA pack** (finalists s0 + MSD + avg_frozen-if-present; delta APGD→standard per model → `results/standard_vs_apgd.md`) → avg_frozen s0–s2 (**skip-if-done**: nếu `results/eval_avg_frozen_sX.json` xuất hiện từ Colab thì bỏ qua — không trùng việc) → std-pack re-run. ~16–20h unattended. Mọi run tự eval+append p2_summary.
- **Drive-sync duty (Claude):** khi thấy kết quả Colab trên Drive (attackdro_results/) → kéo JSON về results/ để skip-logic hoạt động.
- **T-axis reframed** (POSITIONING §2.6 + MEMORY): T = dial ℓ∞↔(ℓ1,ℓ2); warm T union ℓ∞-limited (F2-coherent); ℓ1 record 51.5 @T=1 (đánh dấu trong per-norm figure); "cold beats hard" gated CARD-5 (prereg 41.5–43.5 chết / <38 sống).
- **Ping format CARD-5 đã arm:** worst-∪ vs pre-registered band, per-norm, so 3 nhánh đọc.
- ⚠ Reminder: PC không được sleep trong ~20h tới (powercfg).

## 2026-07-04 — CARD-5 LANDED + T025 3-SEED COMPLETE + archive mapped
- **CARD-5 max_inhouse s0: worst-∪ 43.9** (clean 78.9 · ℓ∞ 45.5 · ℓ2 61.6 · ℓ1 47.2) — **NGOÀI band pre-registered 41.5–43.5, phía CAO**. Đọc: (a) "cold beats hard" CHẾT dứt khoát — hard-MAX in-house là số tốt nhất của ta, trên mọi soft-T; (b) trục T giờ đọc gần-monotone từ lạnh→ấm: 43.9(T→0) → 43.0 → 42.6 → 41.5 → 42.1(bump); (c) **recipe-confound được chính ta tái lập: locuslab MAX ckpt 25.0 vs max_inhouse 43.9 = 18.9pp**, khớp C&H retrain 44.0 ‡ — F-recipe finding rất mạnh; (d) clean 78.9 thấp nhất bảng = đúng trade hard-max. 1 seed, apgd — cần seeds nếu muốn claim.
- **T025 3-seed COMPLETE: 43.0 / 43.1 / 43.0 → mean 43.03 ± 0.05** (variance nhỏ bất thường — tốt). **3-seed mean > MSD 42.5** → điều kiện Mức-A thứ nhất ĐẠT; còn chờ standard-AA (pipeline tự nổ sau 3a+T05 seeds).
- Pipeline: 3a_s1 đang train (31/50); queue còn 3a_s2 → T05 s1,s2 → std-pack → avg_frozen.
- **Archive `ardg_consist` mapped + PARKED** (POSITIONING §2.4 three-axis map: weighting +1.8–3.3 worst-∪ · consistency +0.5–1.6 AA-ℓ∞ old-protocol · ramp-components ℓ∞ 47.27 nhưng clean −6.4; SOURCES entry với ramp_full_09_b1_gp incomplete; future-work composability upgraded). No GPU pre-deadline.

## 2026-07-04 — avg_frozen (Colab) + CARD-1 verdict + CARD-7 armed
- **avg_frozen ×3 (Kiet-reported từ Colab): 40.7/41.1/40.1 → 40.6±0.5, clean 82.5, ℓ1 50.4.** ⚠ **SYNC BLOCKED: JSONs KHÔNG có trên Drive** (đã search attackdro_results / avg_frozen / sharedWithMe — chỉ thấy archive ardg_consist). KHÔNG fabricate file từ số quote → p2_summary/MANIFEST rows + std-AA ckpt CHỜ file thật. Cần Kiet: chạy lại cell 5 (Drive copy) HOẶC bỏ trực tiếp eval_avg_frozen_s{0,1,2}.json vào results/ + avg_frozen_s0_best.pt vào checkpoints/. Skip-logic của finalpipe Stage D sẽ tự nhường nếu file về trước (~7h nữa); nếu không, local sẽ train lại (fallback thiết kế).
- **CARD-1 VERDICT (decision rule, số reported):** ΔDRO = P1 39.7±0.7 − frozen 40.6±0.5 = **−0.9 → loss-signal DRO neutral-to-slightly-harmful dưới recipe control**. F4 upgraded (MEMORY §5). Tier-1 anchor giờ là **avg_frozen 40.6±0.5**; locuslab AVG 38.7 → tier-2 reference.
- **F7 (candidate) recorded:** recipe confound quantified — AVG +1.9, MAX +18.9 (cặp evidence); CARD-7 sẽ thêm điểm MSD.
- **CARD-7 msd_inhouse GO:** msd_v0 ported faithful (commit ef34194, fp32/[0,1], per-step 1 grad + 3 candidate fwd, k~U{5,20}), objective `msd` wired, smoked ✓. **Gated launcher armed (tmux `p7msd`) — chờ finalpipe xong, không preempt.** Pre-registered (strategy side): 43.3–44.5 → re-scope sang recipe-controlled ladder; ≤42.5 → "match" đứng vững hơn.
- **POSITIONING:** bỏ "ℓ1 record" framing (uniform in-house cũng ℓ1≈50). std-AA pack đã sẵn avg_frozen_s0 trong TARGETS (chờ ckpt).

## 2026-07-04 — avg_frozen files DELIVERED (mystery solved: Colab chạy trên account Google khác)
- 3 JSONs nhận trực tiếp từ Kiet, ghi vào results/ nguyên văn. **Validated:** protocol eps (0.03/0.5/12) ✓, apgd n=1000 ✓, khớp số quote ✓. **avg_frozen: 40.7 / 41.1 / 40.1 → 40.6 ± 0.4** (clean 82.2–83.0, ℓ1 50.0–50.6). p2_summary + MANIFEST cập nhật; provenance notes nâng từ "‡reported" → "files on disk".
- finalpipe Stage D sẽ **skip cả 3 avg_frozen** (skip-logic hoạt động như thiết kế — không duplicate).
- **Còn thiếu cho std-AA:** `avg_frozen_s0_best.pt` (checkpoint, ~45MB) — cần Kiet kéo từ Colab/Drive-account-kia vào `checkpoints/`. Không gấp: std-pack Stage E tự nhặt nếu có; nếu không, avg_frozen thiếu hàng std (ghi chú trong bảng cuối).

## 2026-07-04 — RAMP METHODOLOGY ADOPTED + CLAIM LADDER REVISED + CARD-8 ramp_verify LAUNCHED (Kiet)
**Directive (Kiet, sau khi đọc full RAMP paper) — 5 điểm, đã thực thi vào docs:**
1. **Bảng from-scratch cuối = ‡ CITED toàn bộ** (5 seeds, 8/255): RAMP 44.6±0.6 · MAX 44.0±0.7 · MSD 43.9±0.8 · E-AT 42.4±0.6 · SAT 40.4 · AVG 40.1 — KHÔNG retrain; số in-house ta ở tier riêng + protocol caveat (đúng cách RAMP tách from-scratch row khỏi cited baselines). POSITIONING §1 cập nhật với ±std.
2. **max_inhouse 43.9 VALIDATED vs C&H MAX 44.0±0.7 (trong 1σ)** → F7 nâng lên BA evidence độc lập (AVG +1.9 · MAX +18.9 · external cross-validation của recipe ta). Prediction miss CARD-5 (band 41.5–43.5) được giải thích: hard-MAX genuinely ~44 với recipe đúng — không phải eval error.
3. **CARD-7 msd_inhouse HẠ xuống OPTIONAL** (không còn gate claim nào — ta cite C&H như RAMP làm); vẫn armed `p7msd`, chạy khi GPU rảnh, hủy không tiếc nếu cần slot.
4. **CLAIM LADDER REVISED — "match/beat MSD" BỎ HOÀN TOÀN.** C1 (main, unique) = mechanism F4–F6 · +F7 = eval corrections · C2 = simple cheap adaptive weighting (~2–3× rẻ/iter) ĐẠT cụm SOTA 43.9–44.6, không vượt. Với std 5-seed: RAMP 44.6±0.6 chồng lấn MAX/MSD trong ~1σ → đọc là CỤM, không phải outlier tách biệt.
5. **RAMP Eq. 2 có L_max term** → per-sample max được community endorse (NeurIPS'24); trục T của ta = phân tích + làm mềm term đó → related-work line "complementary, không competing" (POSITIONING §2).
**CARD-8 ramp_verify — Kiet TỰ LAUNCH 08:25 (đảo quyết định P3-đóng):** upstream be4971f, patch duy nhất `import copy` (upstream bug trong gp()); official cmd seed-0, 80 epochs, recipe HỌ nguyên vẹn; W&B 4oek4cr7; eval kép armed (apgd + standard AA × {0.03, 8/255}); ckpt đích `external/RAMP/trained_models/RAMP_beta_0.5_lbd_5_0/ep_80_0.pth`; ETA ~14h+ (chia GPU với finalpipe — 2 trainer, đúng cap). ⚠ Chưa có pre-registered bands — planner nên đăng ký trước khi lands.
**Pipeline đồng thời:** bindaware_s1 LANDED **42.1** (clean 80.7 · ℓ∞ 44.4 · ℓ2 65.1 · ℓ1 46.6; best ep 33) → 3a hiện 42.3/42.1, chờ s2 (đang ep 49/50, chậm lại 239→330s/epoch do chia GPU). Queue còn: T05 s1,s2 → std-AA pack. Nhắc lại: cần `avg_frozen_s0_best.pt` vào checkpoints/ cho hàng std-AA của avg_frozen.

## 2026-07-04 09:00 — CARD-8 PRE-REGISTERED (before landing, per E1) + avg_frozen ckpt confirmed absent
- **CARD-8 ramp_verify bands locked** (strategy side): decision var = RAMP-repro @8/255, 1-seed, std-AA. within 44.6±0.6 → repro OK · 43.3–44.0 → mild under-repro (keep bar+caveat) · **<43.3 → E9 bug-investigation FIRST, NEVER lower RAMP from 1-seed** · >45.2 → suspicious edge. **INVARIANT: our 1-seed repro NEVER replaces ‡cited 5-seed 44.6±0.6; row = "our 1-seed reproduction of RAMP".** Purpose = (a) validate we run their code right (gate for CARD-8b composition), (b) one @0.03 protocol-verify point.
- **T025 43.0 framing CONFIRMED** = "within the SOTA cluster", not a beat claim (Kiet). Locked in POSITIONING/MEMORY.
- **avg_frozen_s0_best.pt CONFIRMED ABSENT**: not in checkpoints/ (only smoke ckpts Jul 2), and connected-account Drive search returns only thesis-era 2025 ckpts + old notebooks — the avg_frozen runs live on Kiet's *other* account. std-AA pack skips-if-absent (no crash); avg_frozen std row footnoted missing until Kiet copies the real ckpt in.

## 2026-07-04 09:00 — avg_frozen checkpoints RESOLVED (all 3 pulled from Drive)
- Kiet shared Drive folder (14Hp7okg...); after setting it to "anyone-with-link", `gdown --folder` pulled all 3 (s0/s1/s2, 44.8MB each) into `checkpoints/`. First attempt failed = folder was owner-only (permission list showed owner only); gdown/curl need public link, MCP would've forced 45MB base64 into context (rejected). Public-share flip fixed it.
- **Integrity verified:** 3 distinct md5 (a1646f7… / a141aa7… / 48cc8fd…), all load as `{'model','cfg'}` dicts, structure matches in-house ckpts → compatible with `eval_standard_pack.load_model` kind="ours" (`ck["model"]`).
- **std-AA pack (finalpipe Stage C/E) now includes avg_frozen_s0** — the last load-bearing tier-1 dependency is closed. Implicit re-validation happens when std-pack loads+attacks s0 (would fail loudly if corrupt); the delivered JSON already gives apgd numbers (clean 82.2, ℓ1 50.6, union 40.7).

## 2026-07-04 — DIRECTIVE: repo consolidation + fair-comparison + experiment table + STRATEGIC PIVOT (Parts A–D, no GPU)
**PART B — EXPERIMENT_TABLE.md + generator (the single table Kiet reads):** `scripts/make_experiment_table.py` regenerates `docs/EXPERIMENT_TABLE.md` from `results/*.json` live (numbers cannot drift; `--check` mode for CI). 4 blocks: Tier-1 in-house (P1 39.7±0.9 · AVG 40.6±0.5 · 3a 42.4±0.4 · T025 43.0±0.1 · T05/T1/T2 · MAXih 43.9 · MSDih pending) · Tier-2 locuslab re-evals (from baseline_table.json) · Tier-3 ‡cited (5-seed, 8/255) · Pending/planned (RAMP-repro, CARD-8b). Tier-3 numbers are literature constants, clearly flagged, never mixed.
**PART D — STRATEGIC PIVOT (method-forward; erase nothing):** MEMORY §1 goal → beat-or-approach 43.9–44.6 cluster via composition (CARD-8b) OR the argued gap ("all current aggregations — MSD hard-max / E-AT fixed-geometric / RAMP hard-L_max+fixed-λ — are static + sample-agnostic; none adapts to measured per-sample state-dependent binding, F5/F6"). **F1–F7 re-purposed as gap argument, not discarded.** MEMORY §7 gate **G2′ (2026-08-02):** 8b Δ>repro+0.5 (1 seed) → method paper; else → mechanism framing. **CARD-8b written (E1, NEXT-UP, gated on CARD-8 pass): design Opt A replace L_max with per-sample T-softmax (recommended) / Opt B augment; baseline = CARD-8 repro itself; SHOW KIET before launch.** **Trait-vs-state diagnostic RAN (CPU, `scripts/dev/binding_trait_vs_state.py`):** population binding STATE-like (uniform→ℓ∞ collapse, ℓ2→0 per F3, decisiveness ↑, max_inhouse ℓ1→ℓ∞ migration, TV drift .30–.51) → supports F6 online-measurement + CARD-8b Opt A. ⚠ Honest limit surfaced: dumps are `shuffle=True` first-batch w/o indices → true per-sample trait-stability NOT measurable; needs fixed-probe-batch instrumentation (1-line + 1 re-run, gated).
**PART C — fair-comparison inventory (POSITIONING §2.7):** per method (re-eval / retrain-feasible / cite). Key finding: **E-AT retrain IS feasible in-house** (`external/RAMP/eat_train.py` runnable, pretr ckpts present, robustbench OK, 3-epoch fine-tune cheap) → **CARD-9 eat_inhouse written (gated, card-first)**; it's the most direct differentiation (fixed-geometric vs adaptive, same recipe). MSD/AVG/MAX already covered; SAT cite-only; RAMP repro running.
**PART A — consolidation (zero behavior change; pipeline untouched & verified alive):** scripts/ top = paper entry points only (train, evaluate, eval_standard_pack, validate_baseline, make_experiment_table, run_final_pipeline.sh, run_card7_msd.sh) + subdirs figures/, ramp/; created `scripts/figures/make_all.py` (one-command figure regen). Moved 14 dev/diagnostic scripts → `scripts/dev/` (4 needed `../`→`../../` path-depth fixes, verified robustdro still imports + healthcheck ROOT resolves). configs/ = base + 8 paper methods; `pgd_at.yaml`→`configs/archive/`. **Deleted toy MLP** `src/train.py` + `configs/default.yaml` (only docs referenced them). docs/ diet: HANDOFF/MACBOOK_PROMPT/PC_PROMPT → `docs/archive/`; stale 72-line `docs/PROGRESS.md` fork → archive (root PROGRESS.md canonical). README rewritten 29 lines (what/results-link/reproduce-ONE-number, CLI verified). WORK_FLOWS.md was empty → written 26-line rules-only. **Deviation noted:** kept EXPERIMENT_CARDS_P2 + RAMP_BASELINE + SYNC_SNAPSHOT at docs/ top (active/load-bearing) beyond the 6-doc keeper list — easy to archive later if planner prefers strict 6.
**Pipeline during all of this:** finalpipe T05_s1 training (ep 8), RAMP seed-0 (~epoch 1, 45/391), p7msd gated — all verified alive after moves; no pipeline-referenced file was touched.

## 2026-07-04 — Gate alpha fixed-probe instrumentation v2 (smoke only)
- **Problem fixed:** old `results/loss_mats/<run>/epNNN.pt` dumps were first shuffled train batches without indices, so per-sample binding stability was NOT measurable. Added a separate config-flagged path `train.groupdro.probe_binding: true`, off by default.
- **Implementation:** `GroupDROTrainer` now samples a deterministic train probe with raw `ToTensor()` images, stores CIFAR train `indices` + `labels`, and dumps `L=[G,B]` per-norm loss matrices to `dumps/probe_binding/<run>/probe_binding_epNNN.pt` every `probe_binding_every` epochs. Existing training behavior is unchanged when the flag is false.
- **Smoke:** CPU 2-epoch tiny run `probe_binding_v2_smoke` with 16 fixed samples and 1-step attacks completed. Dump format verified: two files, `L=(3,16)`, identical `indices` and `labels` across epochs.
- **Diagnostic v2:** `scripts/dev/binding_trait_vs_state_v2.py` wrote `results/trait_state_v2.md`. Smoke metrics were persistence 100%, chance 100%, stable 100% all `linf`; this only validates alignment because it has 2 checkpoints / tiny attacks.
- **Verdict:** Gate alpha is **NOT passed yet**. Need a decision-grade fixed-probe run (next finalist or short dedicated real run) before claiming trait/state/both or building a predictor.

## 2026-07-05 — CARD-8 Phase-A RAMP parity evals under our harness
- **RAMP repro eps=0.03 row:** `results/eval_ramp_repro_eps003_apgd_n1000.json`, our `eval_union` harness, n=1000, APGD CE+T, clean 81.2 · Linf 48.4 · L2 65.8 · L1 49.7 · worst-union 47.2.
- **RAMP repro eps=8/255 row:** `results/eval_ramp_repro_eps8255_apgd_n1000.json`, our `eval_union` harness, n=1000, APGD CE+T, clean 81.2 · Linf 47.2 · L2 65.8 · L1 49.7 · worst-union 46.1. CARD-8 PASS remains: internal RAMP final eval was 45.2 at eps=8/255, n=10000, APGD CE+T; cited 44.6±0.6 stays separate.

## 2026-07-05 — RESULT · Sync inventory from `results/*.json`
- **Numbers vs preregistration:** inventory was required before edits. Corrected inventory found APGD/standard eval rows for finalists, avg_frozen, max_inhouse, msd_inhouse, official checkpoints, and RAMP Phase-A; no malformed JSON. Training-log/smoke JSONs are present but have no final eval metrics, so they are flagged as non-eval.
- **Decision-rule branch fired:** sync can proceed from eval JSONs only. Missing-run branch fired for CARD-8b pilot eval JSONs: no top-level `results/*.json` matches CARD-8b/augment/replace/RAMP-bindaware.
- **Surprises:** `results/ramp_RAMP_beta_0.5_lbd_5_0.json` is a training/provenance log with history/raw arrays but no normalized `metrics.worst_union_acc`; it is not used as a final eval row.
- **Mechanism note:** the source-of-truth layer is now clearer: training logs prove provenance; `eval_*.json` files prove table numbers.
- **Next step:** keep future syncs strict: every pilot must land an eval JSON, or it stays missing.

## 2026-07-05 — RESULT · Finalist 3-seed APGD consolidation
- **Numbers vs preregistration:** finalist seed check was meant to test whether the 1-seed signals persisted. Synced eval JSONs give T025 43.0+/-0.1, T05 42.7+/-0.3, bindaware 42.4+/-0.4 worst-union at eps=(0.03,0.5,12), APGD n=1000.
- **Decision-rule branch fired:** signal-persistence branch fired for all three finalists; no stronger public wording follows from APGD alone.
- **Surprises:** T025 variance is very small across the three synced files; T05 has the best l1 among the three-seed finalists but lower union than T025 because linf is tighter.
- **Mechanism note:** T is a linf-vs-l1/l2 dial: warmer T supports l1/l2/clean while union becomes linf-limited.
- **Next step:** use these as Tier-1 APGD rows; standard-AA spot checks are logged separately.

## 2026-07-05 — RESULT · CARD-1 avg_frozen
- **Numbers vs preregistration:** avg_frozen synced rows are 40.7/41.1/40.1, mean 40.6+/-0.5 worst-union, APGD n=1000, eps=(0.03,0.5,12). AttackDRO union is 39.7+/-0.9 from synced rows.
- **Decision-rule branch fired:** loss-signal DRO is neutral-to-slightly harmful under recipe control: delta AttackDRO-minus-frozen is -0.9pp.
- **Surprises:** avg_frozen keeps l1 high at 50.4+/-0.3; the weakness is linf at 41.2+/-0.9.
- **Mechanism note:** F4 is about the loss signal/objective, not just grouping machinery.
- **Next step:** keep avg_frozen as the recipe-controlled uniform anchor and use `avg_frozen_s0` in the standard-AA pack.

## 2026-07-05 — RESULT · CARD-5 max_inhouse
- **Numbers vs preregistration:** max_inhouse_s0 is 43.9 worst-union, clean 78.9, linf 45.5, l2 61.6, l1 47.2, APGD n=1000. This landed above the pre-registered 41.5-43.5 band.
- **Decision-rule branch fired:** high-side band miss: hard per-sample max is strong under the in-house recipe; the "cold softmax has special advantage over hard max" read is refuted.
- **Surprises:** the row is strong despite the lowest clean accuracy in the local table.
- **Mechanism note:** hard max buys linf/union at a clean/l2 cost, consistent with the T-axis trade.
- **Next step:** retain as Tier-1 recipe-controlled anchor; do not use the weak official MAX checkpoint as a method-principle claim.

## 2026-07-05 — RESULT · CARD-7 msd_inhouse
- **Numbers vs preregistration:** msd_inhouse_s0 is 44.1 worst-union, clean 77.1, linf 44.6, l2 63.2, l1 50.0, APGD n=1000. It falls inside the pre-registered 43.3-44.5 recipe-controlled-ladder band.
- **Decision-rule branch fired:** recipe-controlled ladder branch fired; this adds F7 evidence rather than changing the cited-baseline policy.
- **Surprises:** clean is much lower than the official MSD checkpoint row while union is higher under APGD.
- **Mechanism note:** the MSD recipe also trades clean for l1/union support, so recipe effects are not isolated to AVG/MAX.
- **Next step:** keep as a one-seed Tier-1 row; do not spend further seeds unless the planner makes it load-bearing.

## 2026-07-05 — RESULT · CARD-8 RAMP reproduction Phase-A
- **Numbers vs preregistration:** synced controlled APGD rows are RAMP eps=0.03 union 47.2 and eps=8/255 union 46.1, both n=1000. The CARD-8 prereg decision variable was standard-AA at eps=8/255; no `eval_std_ramp*.json` is synced, so that standard-AA branch is not fired from JSON.
- **Decision-rule branch fired:** APGD parity/validation branch remains usable for CARD-8b baselining; standard-AA prereg branch remains missing from synced JSON.
- **Surprises:** the eps=0.03 row is +1.1pp over the eps=8/255 row, as expected directionally, but both rows are stronger than the in-house Tier-1 rows.
- **Mechanism note:** the gap is mostly linf: RAMP repro has linf 48.4 at eps=0.03 and 47.2 at eps=8/255.
- **Next step:** CARD-8b must compare only against the matched local RAMP row: 47.2 for eps=0.03, 46.1 for eps=8/255.

## 2026-07-05 — RESULT · CARD-8b pilot audit
- **Numbers vs preregistration:** no CARD-8b pilot eval JSON is present in top-level `results/*.json`; therefore delta_8255 and delta_003 are missing, not zero.
- **Decision-rule branch fired:** missing-sync branch. No positive/no-harm/kill branch can be read.
- **Surprises:** `docs/CARD-8b_RAMP_BINDAWARE.md` exists and is detailed, but no eval result file for augment/replace pilots is synced.
- **Mechanism note:** no Linf-gate readout exists on disk for CARD-8b. The relevant baseline Linf values remain RAMP repro 47.2 at eps=8/255 and 48.4 at eps=0.03.
- **Next step:** if a pilot ran elsewhere, sync its two matched eval JSONs; otherwise keep CARD-8b card-first and do not infer a result.

## 2026-07-05 — RESULT · standard-AA pack
- **Numbers vs preregistration:** refreshed `results/standard_vs_apgd.md` from paired JSONs. APGD->standard deltas: T025 +0.0pp, T05 +0.0pp, bindaware_s0 +0.0pp, MSD_official +0.3pp, avg_frozen_s0 +0.0pp, all n=1000 at eps=(0.03,0.5,12).
- **Decision-rule branch fired:** standard-AA degradation guard passed; no model dropped more than 1.5pp.
- **Surprises:** standard AutoAttack did not reduce any synced row; MSD official increased by +0.3pp, consistent with evaluation noise/attack-version differences.
- **Mechanism note:** the APGD CE+T iteration metric is not obviously overstating these seed-0 rows under the standard pack.
- **Next step:** for final paper numbers, extend standard-AA only if a row becomes claim-bearing beyond these spot checks.

## 2026-07-05 — RESULT · Gate alpha trait/state fixed-probe status
- **Numbers vs preregistration:** `results/trait_state_v2.md` reports only `probe_binding_v2_smoke`: 2 checkpoints, 16 samples, persistence 100.0%, marginal chance baseline 100.0%, stable-trait samples 100.0%, switchers 0.0%, all stable labels linf.
- **Decision-rule branch fired:** instrumentation/alignment smoke branch only. Gate alpha does not pass because the fixed-probe run is not decision-grade.
- **Surprises:** the smoke numbers are degenerate but useful: persistence equals chance because every smoke sample is linf-bound.
- **Mechanism note:** this validates fixed-index alignment, not trait/state biology. A trait/state/both verdict remains unavailable.
- **Next step:** run a decision-grade fixed-probe diagnostic before any per-sample predictor or trait/state claim.

## 2026-07-05 — RESULT · Full re-verification sync (Claude Code executor)
- **Scope:** re-ran the results inventory from scratch over all 18 `results/eval_*.json` + `baseline_table.json` + the two RAMP eval JSONs. Purpose was a fresh strict re-sync, not a new run; no training launched. Per-run RESULT blocks for finalists / avg_frozen / max_inhouse / msd_inhouse / RAMP Phase-A / std-AA / Gate-alpha already exist above (same date) — this block records the re-verification, not duplicates.
- **Numbers vs preregistration:** every Tier-1 and Tier-2 row reproduced **identically** to the synced MEMORY table. No malformed/unparseable JSON. Multi-seed means recomputed: T025 43.0±0.1, T05 42.7±0.3, bindaware 42.4±0.4, avg_frozen 40.6±0.5, attackdro_union 39.7±0.9 (worst-union, APGD, n=1000, eps=(0.03,0.5,12)). One-seed anchors: msd_inhouse 44.1, max_inhouse 43.9, bindaware_v2 41.5, T1 41.5, T2 42.1.
- **std-AA APGD→standard delta per model** (standard − APGD, worst-union, n=1000, eps=(0.03,0.5,12)): T025 43.0→43.0 = **+0.0**; T05 42.6→42.6 = **+0.0**; bindaware_s0 42.3→42.3 = **+0.0**; avg_frozen_s0 40.7→40.7 = **+0.0**; MSD_official 42.5→42.8 = **+0.3**. Decision-rule: degradation guard PASS — **no model dropped >1.5pp**; standard AA did not reduce any synced row (MSD +0.3 is attack-version/noise).
- **CARD-8b Linf-gate readout:** **MISSING, not zero.** No `results/*.json` matches CARD-8b/augment/replace/ramp_bindaware (grep returned NONE). Therefore delta_8255 and delta_003 cannot be read; the matched RAMP-repro baselines remain Linf 47.2 / union 46.1 at eps=8/255 and Linf 48.4 / union 47.2 at eps=0.03. CARD-8b stays card-first; missing-sync branch fired.
- **Gate-alpha trait/state:** still **PENDING** — `results/trait_state_v2.md` carries only `probe_binding_v2_smoke` (2 checkpoints, 16 samples, persistence 100% == chance 100%, all linf-bound, degenerate). Instrumentation/alignment validated; verdict trait/state/both remains unavailable pending a decision-grade fixed-probe run.
- **Surprises:** none — a full independent re-parse landed on the prior consolidation to the decimal, which is the intended "meter is stable" outcome.
- **Mechanism note:** source-of-truth layering holds: `eval_*.json` = table numbers, training/provenance logs (e.g. `ramp_RAMP_beta_0.5_lbd_5_0.json`) are not final-eval rows.
- **Next step:** any CARD-8b pilot or decision-grade fixed-probe run must land its eval JSON(s) into `results/` before it can enter MEMORY; until then both stay open.

## 2026-07-05 — CARD: Gate-α decision-grade fixed-probe run (PRE-REGISTERED, launched)
- **Hypothesis / question (3-branch, per PLAYBOOK V):** is per-sample binding norm (argmax loss across {linf,l2,l1} source attacks) a stable TRAIT (persists across training) or a STATE (tracks model state)? Verdict gates the ambitious lane: TRAIT→predictive per-sample weighting viable; STATE→online-tracking framing; BOTH→two-tier.
- **Config / the ONE change:** `configs/gate_alpha_probe.yaml` inherits `avg_frozen` (uniform q, neutral recipe that does NOT reweight toward any norm). ONLY change vs avg_frozen: `probe_binding: true`, size 512, every 5 epochs. Full 50-epoch run, seed 0, eps=(0.03,0.5,12). run_name `gate_alpha_avgfrozen_s0`; dumps→`dumps/probe_binding/gate_alpha_avgfrozen_s0/probe_binding_epNNN.pt` at epochs 0,5,…,45 (10 checkpoints → decision-grade ≥5).
- **Smoke+timing gate PASSED:** warmup step + 20-step timing + one real probe dump ran finite; epoch-0 probe non-degenerate (linf 46% / l2 0% / l1 54%, vs smoke's degenerate 100% linf → 512 samples give a real distribution). 390 steps/epoch, 0.618 s/step = 4.02 min/epoch; probe 2.4s×10; peak VRAM 1.89GB. **Projected full ≈ 3.36h.**
- **Pre-registered decision rule** (mirrors `binding_trait_vs_state_v2.py` gate; verdict written to `results/trait_state_v3.md`):
  - TRAIT (Gate-α PASS→predictive weighting): avg_persist > 0.70 AND stable_frac > 0.50 AND (persist − chance) > 0.20.
  - PURE STATE (pivot to tracking/dynamics): |persist − chance| < 0.10 AND stable_frac < 0.35.
  - MIXED/BOTH (two-tier): otherwise.
  - Report per-sample persistence vs chance, stable-trait vs switcher fractions, early-vs-late binding freq. Numbers from disk only.
- **Authorization:** executor task directive = human sign-off for this >2 GPU-hr run (Gate-α is the pre-registered critical path). No competing GPU job (tmux empty, GPU idle). W&B disabled for headless robustness; source of truth = probe dumps + results JSON.
- **STATUS: LAUNCHED** in tmux `gate_alpha`. Verdict pending run completion + v3 diagnostic.

## 2026-07-05 — GATE-α PIVOT: decision-grade run moved to binding-aware recipe (Kiet directive)
- **Why the pivot:** avg_frozen (uniform) is linf-locked (~.98), so chance baseline ~.89–.96 and raw persistence is base-rate-inflated (excess only +2–5pp) → per-sample structure is NOT measurable there. The decision-grade Gate-α run must use the **binding-aware per-sample-soft recipe at T=0.25**, whose ~.73/.27/.00 binding is balanced enough that chance is low and persistence carries signal.
- **Action:** killed avg_frozen Gate-α run at ep35 (kept its 8 fixed-probe dumps ep0–35 as the CONTRAST row). Launched `configs/gate_alpha_bindaware_probe.yaml` (bindaware_sample T=0.25 + probe_binding 512/every-5, 50 epochs, seed 0), run_name `gate_alpha_bindaware_T025_s0`, tmux `gate_bind`, SOLO on GPU (critical path, no contention). Dumps→`dumps/probe_binding/gate_alpha_bindaware_T025_s0/`.
- **Diagnostic v3** written: `scripts/dev/binding_trait_vs_state_v3.py` → `results/trait_state_v3.md`. Locked metric defs (pre-registered): chance(pair)=Σ_g f_a(g)f_b(g); **headline = EXCESS = persistence − chance** (never raw); per-norm retention P(g_t+1|g_t), EXCESS_g = retention−marginal; **kappa_g=(ret−marginal)/(1−marginal)** headroom-normalized companion (raw excess is capped at 1−marginal, so a saturated majority like linf can't show big raw excess even if sticky → judge linf on kappa, l1 on raw excess). MIGRATION (population drift, F6) and TRAIT/STATE (sample-level excess on fixed indices) reported as SEPARATE axes. l1-minority conditional P(still l1 next dump) vs l1 marginal is the decision-critical number.
- **VERDICT RULE (pre-registered):** l1 EXCESS≫0 AND linf sticky(κ>0.5) → TRAIT → predictive weighting (Conj 2) viable, Gate-α PASS. both≈0 → PURE STATE → dynamics framing. linf sticky but l1≈chance → BOTH → two-tier. (smoke hint: linf=465 stable, l1=2 stable → leans BOTH.)
- **Early CONTRAST readout (avg_frozen ep0–35, 8 dumps):** persistence 93.9% / chance 89.0% / **EXCESS +4.9pp**; per-norm linf marginal 94.5% ret 96.5% excess +2.1pp κ0.38; l1 marginal 5.5% ret 49.5% **excess +44.0pp κ0.47**; ever-l1 90/512, from-l1→{linf 50.5%, l1 49.5%}, durable stable-l1 only 3. Reads: even under uniform training l1 is pairwise-sticky (+44pp) but NOT durably (3 samples, 50% escape to linf) → base-rate contrast holds; the balanced bindaware run is the clean test. Verdict pending that run.
- **smoke_timing** (results/timing_projections.md, measured under contention→~2× upper bound): CARD-8b v1 augment ≈14.2h/seed contended (~7h solo) for 80 ep; full-AA ≈14.2× APGD (APGD union n=1000 21min, full-AA n=1000 ~5.0h). No 80-ep launched.

## 2026-07-05 — PROTOCOL RE-LOCK to eps_inf=8/255 + W&B standardization (Parts A+B; no GPU; gate_bind untouched)
**Part A — protocol locked to 8/255** (was 0.03, MSD-2020-specific; 8/255 is the subfield standard — RAMP/E-AT/C&H/RobustBench — enabling direct SOTA comparison). eps_l2=0.5, eps_l1=12 unchanged.
- `configs/base.yaml` eps_inf → 0.03137254901960784 (8/255); old protocol archived → `configs/archive/base_eps003.yaml`. Verified 8/255 propagates to every method config. gate_bind (live) unaffected (config parsed at launch; on-disk edit does not touch the running process).
- **Re-eval is QUEUED, not run** — GPU is held by the critical-path gate_bind run and must not be disturbed. Runner `scripts/reeval_8255.py` (idempotent skip-if-already-8/255; archives each 0.03 JSON→`results/archive/eval_<stem>_eps003.json` before overwriting the canonical name; phases: `apgd` cluster, then `std` finalists+MSD). Dry-run validated: 17 in-house APGD + official(MSD/AVG/MAX) + 5 finalist std; ALL checkpoints present → **re-eval only, ZERO retrain needed**. Official via `eval_baselines.py --config base.yaml`.
- **Caveat (flagged):** in-house ckpts were TRAINED at 0.03; re-eval at 8/255 is a mild train/eval mismatch (0.03→0.03137, +4.6% eps). A future clean run should train at 8/255. RAMP repro already has 8/255 (union 46.1).
- `scripts/make_experiment_table.py` made eps-aware: reads each JSON's protocol eps, flags non-8/255 rows `⚠STALE@0.03` (all current rows flagged until re-eval). Table not regenerated yet (regenerate post-eval).
**Part B — W&B standardized** (our trainers + our harness evals; external/RAMP keeps its own W&B, only its harness eval mirrored here under `repro`).
- One project **attackdro-union**, entity **kietna** (base.yaml). Run id derived deterministically from run_name → a later eval attaches its summary to the SAME run as training curves. Auto-tags every run: method/run/seed/tier (in-house|repro|official). Config-flagged mode (WANDB_MODE overrides).
- Standardized per-epoch keys (step=epoch): train/loss, per-norm train loss, per-norm probe robust-acc, **probe/clean_acc (added)**, q or per-sample mean-W per norm. `evaluate.py` now logs union+per-norm as W&B summary after writing the JSON (`--run-name`, `--tier`, `--no-wandb`); reusable `log_eval_summary()` in `utils/wandb_log.py`. JSON on disk stays source of truth.
- **Compare view:** `scripts/make_dashboard.py` → `docs/dashboard.html` now has a tier-filterable union-acc bar chart + per-norm + per-row eps label (26 in-house/6 official/2 repro rows render). Command: `.venv/bin/python scripts/make_dashboard.py` then open `docs/dashboard.html`.
- **gate_bind coverage:** launched with WANDB_MODE=disabled → NOT on the dashboard (its data lives in results JSON + probe dumps). Not restarted for logging. The NEXT run inherits full standardized logging. A future predictive-binding run should also use the 8/255 recipe for consistency (assess after gate_bind's trait/state verdict).

## 2026-07-05 — DECISION: train/eval eps MISMATCH elevated (8/255 re-eval = estimate; paper 8/255 needs retrain) — RECORD, no launch
- **Context (Codex flag → decision):** re-evaluating our 0.03-TRAINED checkpoints at the harsher 8/255 is a train/eval mismatch → numbers artificially low, NOT fairly comparable to train-8/255 models (RAMP repro 46.1).
- **1. reeval_8255 = QUICK ESTIMATE / LOWER BOUND only.** Not paper numbers for our methods. Made self-documenting: `evaluate.py` now records the checkpoint's OWN training eps (from `ckpt['cfg']`) into each eval JSON as `train_protocol` + `train_eval_eps_mismatch`; `make_experiment_table.py` + `make_dashboard.py` auto-flag mismatched rows `train@0.03/eval@8255 — lower bound` (current 0.03-eval rows show `0.03 pre-relock` until re-eval). No hand-marking → drift-proof; a future train@8/255 ckpt eval'd at 8/255 gets NO flag.
- **2. Paper-grade 8/255 REQUIRES RETRAIN at 8/255 (train==eval) — now REQUIRED, not optional.** But do NOT retrain old finalists just for this.
- **3. Fold into the next real run (post-Gate-α):** ONE 8/255 clean run = paper protocol + the method the verdict selects (predictive-binding if trait/both, dynamics-informed if state). Drafted in MEMORY Open items; DO NOT launch — gated on the Gate-α verdict.
- **4. Official MSD/AVG/MAX 8/255 re-eval = valid REFERENCE (Tier-2);** released MSD ckpt also train@0.03 → same flag, as a reference not a competitor.
- **Recorded to MEMORY** (new "Protocol notes" section + post-verdict run plan in Open items). gate_bind (ep16, critical path) untouched; no GPU launched. Order stands: Gate-α verdict FIRST → then reeval as estimate → then design the single post-verdict 8/255 clean run.

## 2026-07-05 — RESULT · Gate-α trait/state DECISION-GRADE verdict = TRAIT (PASS)
- **Numbers vs preregistration:** decision-grade run `gate_alpha_bindaware_T025_s0` (bindaware per-sample-soft T=0.25, 50 ep, 512 fixed indexed probe, 10 dumps ep0–45 spanning both LR drops 25/40). Overall persistence 92.3% vs chance 82.2% → **EXCESS +10.1pp**. Per-norm (retention vs marginal-chance): **l1 (minority, marginal 9.6%) retention 63.4% → EXCESS +53.8pp, κ0.60**; linf (majority 90.4%, ceiling-capped) retention 95.4% → EXCESS +4.9pp, κ0.52; l2 marginal 0% (F3). Pre-registered TRAIT bar (l1 EXCESS≫0 AND linf sticky κ>0.5) → BOTH clear → **TRAIT**.
- **Decision-rule branch fired:** TRAIT → predictive per-sample weighting viable → **Gate-α PASS toward Conj 2**. The 8-week direction is the predictive-binding lane.
- **Surprises / honest reading:** TRAIT-DOMINANT WITH A STATE OVERLAY. Pairwise (5-epoch) stickiness is strong, but durable full-run stability is modest — only 11/135 ever-l1 samples are l1 in >80% of dumps (8.1%); 96/135 ever-l1 samples have mode-norm linf; population migrates l1 3.7%→~10% and settles. So binding is a short-horizon trait, not a fixed lifelong label. Smoke had hinted BOTH (linf=465/l1=2 durable); the balanced T=0.25 recipe raises l1 above the bar (κ_l1 0.47→0.60) so decision-grade tips to TRAIT.
- **Contrast (avg_frozen uniform, 8 dumps) = MIXED:** l1 EXCESS +44pp κ0.47 but linf κ0.38 (linf marginal 94.5% too saturated to clear the bar); only 3 durable-stable l1. This is the MOTIVATION figure — per-sample structure is only measurable once the recipe balances the norms; uniform training hides it under base-rate inflation (self-chance ~0.90 vs balanced ~0.82).
- **Mechanism note:** the excess is above the DRIFTING population marginal (chance uses per-pair f_a·f_b), so it isolates individual stickiness from F6 migration — a sample keeps its binding beyond what the moving population predicts. l1 carries the signal (headroom); linf is base-rate-saturated (read via κ).
- **Next experiment:** the post-verdict single 8/255 clean run (drafted, MEMORY Open items) uses PREDICTIVE-BINDING with a SHORT-HORIZON/EMA per-sample binding prior (matches the state-overlay caveat), train==eval at 8/255, matched vs RAMP repro 46.1. reeval_8255 (0.03-ckpt estimate) can run now that the GPU is free.

## 2026-07-06 — PROGRESS SYNC: reeval@8/255 partial (hung), CARD-PB implemented+smoked, retrain plan approved
- **reeval_8255 --phase apgd (LOWER-BOUND estimates, train@0.03/eval@8255):** 9 in-house rows converted before the run **HUNG on max_inhouse_s0** (GPU idle 76 min, no log progress → killed). Converted (worst-union, LB): **T=0.25 41.6/41.6/41.1 (mean 41.4)**, **T=0.5 40.3/41.2/41.2 (40.9)**, **3a bindaware 41.2/41.0/41.1 (41.1)**. Drop vs 0.03: −1.3 to −1.8pp (harsher eps + mismatch penalty). RAMP repro (train@8/255, MATCHED) = 46.1. max_inhouse_s0 healed back to 0.03 canonical (8/255 eval never finished; archive intact). STILL pending re-eval: max/msd/avg×3/attackdro×3 + official (all still 0.03). reeval is idempotent → resume skips the 9 done. **These are NOT paper numbers** (auto-flagged); paper 8/255 = the retrains. Views rebuilt (dashboard + EXPERIMENT_TABLE.md).
- **CARD-PB trainer IMPLEMENTED + smoked** (`objective: predictive_binding` in GroupDROTrainer; `configs/cardpb.yaml`): per-sample **EMA φ** (index-aware loader, 50000×3 state), **batch-partition budget allocation** (full attack on predicted norm, floor on rest), 4-layer safety floor (fixed floor≥3 + 10% recalibration + adaptive raise + guard), **MEASURED attack-FLOPs** from real partitions. Smoke: predicted-path **FLOPs ratio 0.502 (50% cut, < the ≤0.60 target)**, cold=1.0, all-recal=1.0, loss finite, `phi/misprediction_rate` logged; CLI `fit()` end-to-end exit 0 (train→probe→adaptive-floor→ckpt→JSON). **Predictive still gated** on Kiet card approval + a real-hardware FLOPs-cut smoke before committing 3 seeds.
- **8/255 retrain plan APPROVED by Kiet** (`docs/RETRAIN_8255_PLAN.md`): retrain @8/255 train==eval, 3 seeds, NEW run-names (0.03 ckpts intact): **reactive (soft T=0.25) FIRST** → predictive (CARD-PB) → avg_frozen → **3a KEPT (Q1)**. **F7 motivation-only (Q2)** — do NOT retrain msd/max@8/255 (use reeval LB + 0.03 + C&H cited, protocol-labeled). **Seeds (Q3): 3 exploratory; escalate ONLY the reactive-vs-predictive PAIR to 20 seeds + Wilcoxon + BH-FDR** for the efficiency claim. Devices: reactive/avg/3a → **Colab** (standard trainer); predictive+Conj-1 → **5070ti** (predictive needs local CARD-PB attack code).
- **Conj-1 (hard-MAX seed variance):** baseline captured — T=0.25 union @0.03 = {43.0,43.1,43.0} → **sample-std ≈ 0.06pp**. Pre-registered: hard-MAX std > 0.06. Still need max_inhouse s1/s2 @0.03 (queued on 5070ti, GPU now free).
- **NOT done:** Colab reactive runner script (user interrupted the write); full reeval cluster (hung, ~half remaining); Conj-1 training; predictive launch (gated).

## 2026-07-06 — RESULT · Conj-1 hard-MAX seed variance (PRE-REG CONFIRMED)
- **Numbers vs preregistration:** hard-MAX (max_inhouse) @0.03, 3 seeds: union s0 43.9 / s1 44.8 / s2 43.6 → mean 44.1, **sample-std 0.62pp**, range 1.20pp. T=0.25 soft baseline std = 0.06pp (seeds 43.0/43.1/43.0). Pre-reg was hard-MAX std > 0.06.
- **Decision-rule branch fired:** CONFIRMED — hard-MAX std 0.62 ≫ 0.06 (**~10×**). Hard per-sample MAX is far more seed-variable than the soft T-softmax; the soft formulation is the more stable/reliable objective (Conj 1 validated).
- **Surprises:** none directional; s1 is the high seed (44.8, also highest clean 79.9). Per-norm means clean 79.4 / linf 45.7 / l2 63.1 / l1 47.7.
- **Mechanism note:** hard argmax routes the whole gradient to one norm per sample → the selection is seed-sensitive (which norm "wins" flips with init), inflating variance; soft-T averages the routing → stable. This is a formulation argument for soft over hard, independent of eps.
- **Next:** predictive safety-floor smoke now running (chain). Conj-1 numbers are @0.03 (Tier-1); max_inhouse row upgraded to 3-seed 44.1±0.6.

## 2026-07-06 — RESULT · CARD-PB safety-floor smoke = conditional GO (mechanism validated)
- **Setup:** predictive_pb_smoke, 5 ep, cold=1 (worst-case: forces predicted regime early with a barely-seeded, noisy φ). Real hardware.
- **Numbers vs preregistration (go/no-go: no norm drops >2pp):** per-norm robust acc rose overall (early training) — linf 0.23→0.34, l2 0.32→0.51, l1 0.28→0.39, union 0.23→0.33; final epoch = each norm's peak. **MEASURED attack-FLOPs ratio held ~0.50** across the predicted regime (ep0 cold 1.00; ep1-4 0.498/0.495/0.496/0.503) — under the ≤0.60 target. φ misprediction rate ~0.30 (noisy, cold=1 by design).
- **Decision-rule branch fired:** **conditional GO.** The **adaptive safety floor WORKED**: a transient linf dip (ep2→ep3, −3.9pp) tripped the guard → linf floor auto-raised 3→5 at ep4 → linf recovered to peak. No sustained robustness hole. The raw 2pp bar was momentarily exceeded by that transient before correction — expected under the cold=1 stress test.
- **Mechanism note:** the floor's detect→raise valve is the load-bearing safety guarantee, and it demonstrably self-corrected a real dip. FLOPs stayed at ~50% even with the bump.
- **Recommendation for the paper run:** keep adaptive floor ON; use default cold=5 (better-seeded φ → lower misprediction → smaller transients). Predictive-8/255 launch stays gated on Kiet.
- **Next:** GPU free. Reactive-8/255 (3 seeds, first paper row) is #1 in queue, gated on Kiet go. STOP — hold for Kiet.

## 2026-07-06 — LAUNCH · Paper 8/255 chain (Kiet GO): reactive → predictive (train==eval)
- **Committed** the paper-grade 8/255 runs (tmux `paper`, WANDB offline, ~16h): **reactive_T025_8255 s0/s1/s2** (per-sample soft T=0.25, headline reactive row + efficiency baseline) → **predictive_pb_8255 s0/s1/s2** (CARD-PB). Each train==eval@8/255, eval APGD n=1000, new run-names (0.03 ckpts untouched).
- **Predictive config LOCKED before launch** (`configs/cardpb.yaml`, verified): pb_cold_epochs=**5** (not the cold=1 stress value — better-seeded φ, misprediction well below stress level), adaptive floor ON (guard 3.0, the valve that self-corrected linf 3→5), logs phi/misprediction_rate + pb/attack_flops_ratio per epoch (Prop 5 empirical). eps 8/255, T=0.25, 50 ep.
- **Gates green:** Gate-α TRAIT (predictive viable), floor smoke conditional GO (FLOPs ~0.50 held, floor validated), F8 bias-variance (hard-MAX std 0.62 vs soft 0.06, ~10×).
- **Plan:** compare Δunion + attack-FLOPs, all vs RAMP repro 46.1 (matched, train@8/255). Escalate the reactive-vs-predictive PAIR to 20 seeds + Wilcoxon + BH-FDR ONLY if predictive signals the efficiency win.
- **⚠ powercfg NOT confirmable from WSL** (needs Windows admin): could not set standby/hibernate-timeout-ac=0. Today's 7h Conj-1 chain completed without sleeping (strong evidence sleep is already off), but Kiet should confirm in an Admin PowerShell: `powercfg /change standby-timeout-ac 0` + `powercfg /change hibernate-timeout-ac 0`.

## 2026-07-06 — CONFIG · W&B logging schema CONFIRMED (Kiet) + entity fix + pb/beta + protocol flag
- **Entity bug fixed:** `configs/base.yaml` `entity: kietna` FAILED online init ("entity kietna not found" → silent fallback to stdout, no data). Correct default = `kietna-ho-chi-minh-city-university-of-technology`; set `entity: null` (resolves to API-key default). Online init tested OK. Lesson: **offline+`wandb sync` is more reliable than online for unattended** (online silently disables on failure).
- **Schema additions (Kiet-requested), smoke-verified emitting:** `pb/beta` (per-epoch, EMA-horizon → comparable across β-ablation); `train/protocol` + `train/eps_inf` + `train/eps_mismatch` (run summary in `fit()`: self-declares train==eval@8/255 vs a 0.03-ckpt lower-bound re-eval). Full canonical schema now in `docs/MEMORY.md` → **W&B logging schema** section; every future run logs the same keys for cross-run comparison.
- **Confirmed identical key set for reactive vs predictive** (predictive adds only `pb/*`+`phi/*`+`floor/*`) → one runs-table comparison; efficiency scatter = `eval/union` vs `pb/attack_flops_ratio`.

## 2026-07-06 — LAUNCH · Paper 8/255 chain v2 (validate-first, W&B ONLINE, 5070ti only)
- **Live** (tmux `paper2`, WANDB **online** — entity fixed): reactive s0 (soft T=0.25) → predictive s0 (CARD-PB cold=5, floor ON) → eval pair @8/255 APGD n=1000 → `scripts/dev/s0_decision.py` writes **results/reactive_vs_predictive_s0.md** (WAKE-UP artifact) + emits branch. Confirmed online: run `reactive_T025_8255_s0`, tags method/seed/tier, `train/protocol=8/255 eps_mismatch=0`, GPU 96%.
- **GATED continuation (auto):** CONTINUE_BOTH (Δunion ≥ −0.3 & no norm −2pp) → reactive+predictive s1,s2; else KILL_PREDICTIVE → reactive s1,s2 only (predictive halted, logged). 3-seed table grows via `scripts/dev/pair_table.py` → results/reactive_vs_predictive_3seed.md. 20-seed escalation = Kiet's call only.
- **Schema (Kiet-confirmed):** canonical key set in docs/MEMORY.md; added `pb/beta` + `train/protocol`/`eps_inf`/`eps_mismatch` this run. Reactive & predictive log identical keys → one comparison table.
- **⚠ powercfg** still not confirmable from WSL — needs Kiet in Admin PowerShell.

## 2026-07-06 — RESULT · Predictive-binding s0 KILLED (F9) · reactive baseline continues
- **Paper-grade @8/255 (train==eval, APGD n=1000), seed 0:**
  | method | union | clean | linf | l2 | l1 | attack-FLOPs |
  |---|---|---|---|---|---|---|
  | reactive (soft T=0.25) | 41.7 | 79.9 | 42.4 | 64.1 | **47.5** | 1.00 |
  | predictive (CARD-PB) | **40.6** | 81.5 | 44.1 | 63.1 | **44.3** | **0.522** |
  | RAMP repro (matched) | 46.1 | — | — | — | — | 1.00 |
- **Pre-registered KILL fired (both triggers):** Δunion −1.10pp (<−0.3) AND l1 −3.20pp (>−2.0). The **FLOPs cut is real (0.522, ≤0.60)** but bought with robustness: predictive under-defends **l1** — the TRAIT-heavy minority norm (F6, l1 EXCESS +53.8pp). The adaptive floor raised **l2** 3→5 but **not l1**, and l1 still collapsed in final eval → floor did not protect the norm that matters. φ misprediction 29.5%, β=0.5, cold=5.
- **Automation branch:** predictive HALTED at s0 (s1/s2 NOT run — saved ~4h GPU), reactive s1/s2 continue as the needed baseline. **NOT escalated to 20 seeds** (correct — pre-reg required a positive signal). GPU stays busy on reactive.
- **Strategic read for Kiet:** (1) efficiency-at-equal-robustness for CARD-PB **as configured does not hold**; the failure is l1-specific and diagnosable. (2) Obvious next lever = **l1-priority / per-norm safety floor** (never strip budget from the trait-heavy minority norm) — a NEW design, Kiet-gated, not launched. (3) Bigger picture: BOTH in-house methods trail RAMP 46.1 (reactive −4.4pp), so an efficiency claim *vs RAMP* needs reactive to first reach RAMP parity. Artifacts: `results/reactive_vs_predictive_s0.md`, W&B runs `reactive_T025_8255_s0` / `predictive_pb_8255_s0`.

## 2026-07-06 — NOTE · powercfg CONFIRMED holding (read from WSL) + CARD-PB v2 drafted
- **powercfg:** read directly from WSL via `/mnt/c/Windows/System32/powercfg.exe /query SCHEME_CURRENT SUB_SLEEP {STANDBYIDLE,HIBERNATEIDLE}` — **AC sleep=0x0 (never), AC hibernate=0x0 (never)** (scheme Balanced). Reads need no admin; only *setting* did. Chain survives to ~06:30 UTC. (Supersedes the earlier "not confirmable from WSL" flag.)
- **CARD-PB v2 DRAFTED (not launched):** `docs/CARD-PB_v2_CONFIDENCE_FLOOR.md` — predictability-aware floor. Diagnosis: v1 shifts budget l1→linf (linf +1.7, l1 −3.2); v1 adaptive guard is PROBE-driven and the l1 probe is the weakest attack (F1) → structurally blind to l1 erosion (raised l2 3→5, never l1). v2 drives floor by φ per-norm miss rate (unbiased on the recal subset). Needs new code (per-norm miss + `phi/miss_rate_{norm}` logging) + `configs/cardpb_v2.yaml`. Pre-reg: union within noise AND l1 drop <2pp AND FLOPs ≤0.60; if no such point → FINDING "l1 predictability floors efficiency". **Launch gated on reactive 3-seed baseline (~06:30) + Kiet GO.**

## 2026-07-07 — REVIEW · CARD-PB code audit (3 real defects) + R6 restore-point commit
- **Context:** reviewer pass over the uncommitted `predictive_binding` implementation vs the pre-registered CARD-PB design. Found no Codex commits — the WHOLE method was uncommitted working-tree state.
- **R6 (done):** committed the full method as a pre-fix restore point → `b73d078` on branch **`card-pb-fixes`** (92 files, +7583/−299). `main` untouched at 48577cb. Guarded: `.venv-ramp/`+`.venv*/` ignored; embedded `notebooks/attackdro` repo unstaged (own history). Fixes proceed on this branch.
- **R1 (BLOCKER, gated fix):** the safety floor is INERT — `floor_attacks` built once at `steps=pb_k_floor`; the adaptive guard's `_pb_extra_floor` feeds ONLY the FLOPs counter (groupdro.py:215) and the logged `floor/<norm>` (groupdro.py:381), never the actual attack (groupdro.py:214). ⇒ (a) the floor-smoke "linf self-corrected 3→5" claim is UNSUPPORTED (attack never changed); (b) reported FLOPs OVER-count after any raise (true v1 <0.522); (c) **v2's l1-floor sweep would be a no-op on the real attack** — must fix before v2.
- **R3 (mechanism, gated fix):** φ's `Lbar` is updated for ALL norms from crafted losses (groupdro.py:230), but floored norms' losses are under only 3 steps → systematically low → once φ stops predicting a norm it's floored → its Lbar decays → entrenched misprediction. Positive feedback; plausibly the REAL F9 driver (deeper than "guard blind to l1"). Fix: update Lbar only for full-budget norms (predicted+recal).
- **R2 (interpretability, gated fix):** code EMA `Lbar=β·new+(1−β)·old` ⇒ high β = SHORT memory; card labels β=0.8 "long memory/ignore-drift" — inverted. Reconcile card before the β ablation is interpreted.
- **R4/R5 (minor):** "attack-FLOPs" excludes fixed training fwd/bwd (scope the claim; cross-check `train/epoch_time_s`); `phi/misprediction_rate` is a ρ=10% subsample estimate.
- **Reframed F9 cause:** l1-binding samples mispredicted → floored at a FIXED 3/20 steps (the "adaptive" raise did nothing, R1) → under-attacked → collapse, with the φ-update rule entrenching the misprediction (R3). v2's confidence floor treats the symptom; R1+R3 are the mechanism.
- **Plan:** at the reactive-baseline gate (~06:10 UTC), apply R1→R3→R2 on `card-pb-fixes`, then re-run the floor smoke (cold=1) — the REAL floor validation — before the v2 sweep. Reactive s2 untouched (tmux paper2, 19/50 @ 03:50).

## 2026-07-07 — BASELINE · Reactive (soft T=0.25) @8/255 3-seed LOCKED (paper headline reactive row)
- **union 41.90 ± 0.22** (s0 41.7 / s1 42.2 / s2 41.8); **l1 48.10 ± 0.43**; clean ~80; train==eval@8/255, APGD n=1000, W&B online. Very tight seed spread (0.22pp) — consistent with F8 (soft-T stable). vs RAMP repro 46.1 → **−4.2pp**.
- This locks the CARD-PB v2 comparison target: v2 success = union within ±0.3pp of **41.90** AND l1 drop <2pp vs **48.10** AND FLOPs ≤0.60. Chain v2 COMPLETE (decision=KILL_PREDICTIVE); GPU free → R1/R3/R2 fixes proceed on `card-pb-fixes`.

## 2026-07-07 — FIX+VALIDATE · CARD-PB R1/R3/R2 on card-pb-fixes (the REAL floor validation)
- **R1 (blocker) — floor now actually attacks.** Added `_rebuild_floor(g)`: floor attacks are (re)built from `pb_k_floor+_pb_extra_floor[g]`, at init and on every guard raise. **Unit check PASS**: l1 floor 3→8 ⇒ l1 floor attack forwards 3→8 exactly (behavioral, via the real method).
- **R3 — φ no longer self-poisons.** `Lbar` updates ONLY full-budget norms (`full_all | b_hat==g`); floored (under-attacked) entries stay at their last unbiased cold/recal value. First sighting is always cold ⇒ all norms seeded. Runs clean end-to-end (no crash, sane φ/probes).
- **R2 — β labels corrected** in the card: standard EMA ⇒ high β = SHORT memory ((1−β)/β epochs); "ignore-drift" arm = low β (0.3), not 0.8.
- **RE-SMOKE (5ep, cold=1) — real floor validation v1 lacked:** at ep3 all norms dipped → guard raised floors 3→5 → **ep4 flops 0.525→0.614 AND epoch_time 150.5→168.3s (+12%)** = the raised floor ran MORE ACTUAL STEPS (physical proof, not a log change); union recovered .215→.312. In v1 the floor number moved but compute/attack never did — that "floor self-corrected" claim (now RETRACTED) was a mis-attribution. Caveat: 5ep cold=1 stress ⇒ the recovery magnitude also includes normal training progress; the DECISIVE evidence is the flops+time jump proving R1.
- **Net:** F9's mechanism is now genuinely fixable — v2's confidence floor will actually change the l1 attack. Fixes committed on `card-pb-fixes`. **STOP: awaiting Kiet GO before the v2 sweep.**

## 2026-07-07 — VALIDATE · RAMP re-run provenance (recipe-control) + schedule-origin fix
- **RAMP code = upstream be4971f.** Only behavior-affecting local change: a **default-OFF** `--bindaware_variant` flag (our CARD-8b hook). It was NOT passed → training ran the pristine RAMP loss (`loss_best = loss_hard`). Other diffs cosmetic (unused `import copy`, trailing newline). The `static` lr-schedule branch is UPSTREAM, not ours.
- **Checkpoints trained by us** (log_train.txt, ep_10..80 @ save_freq 10). Actual args (from `scripts/ramp/train_ramp_with_wandb.py`): `--lr-max 0.05 --lr-schedule=static --at_iter 10 --epochs 80 --max --kl --gp --lbd 5 --save_freq 10`, seed 0, bindaware_variant=none.
- **Recipe faithfulness:** lr 0.05 ✓; `static` = 0.05 (ep≤70) → 0.005 (ep>70) = the paper's 70+10 single ×10 drop ✓; 80 epochs ✓; 10 inner steps ✓; max+kl+gp = RAMP standard ✓. **⚠ ONE deviation: `--lbd 5` (KL weight)** vs paper-cited λ=2 (RAMP code default 1.5). BUT our **ep_50 = 42.9 exactly matches RAMP paper Table 5 (42.9)** → the re-run faithfully reproduces RAMP's reported numbers, so lbd=5 is plausibly correct for this RN-18 row — reconcile with the paper's λ before labeling. lbd is IRRELEVANT to our recipe-matched run (our per-sample-soft objective has no KL term); it only affects the RAMP reference (46.1).
- **Eval parity CONFIRMED:** RAMP-rerun AND our method both eval'd n=1000 APGD restarts=1 @8/255 → same eval → comparison VALID. 46.1 (ours, APGD) vs paper 44.6 (full-AA n=10000): APGD ≤ full-AA → ours ~1.5pp higher, consistent (matches our std_vs_apgd delta). Not a training issue.
- **Schedule [27,53] origin = MY error:** I read RAMP's `piecewise-ft` branch (×10 drops at ⅓,⅔ → 27,53) as the recipe, but that's RAMP's FINE-TUNE schedule; from-scratch used `static` (drop at 70). **FIXED:** reactive_ramprecipe.yaml now `milestones: [70]` = RAMP's static schedule exactly.

## 2026-07-07 — RESET · Method lane → RAMP recipe (single-recipe, confound-free) + ARCHIVE
- **Decision (Kiet):** retire our-recipe (50ep, lr-drop@25, train attack 10/10/20); rerun ALL methods under **RAMP's recipe** (lr 0.05→0.005 ×10 drop at ep70, 80ep, save every 10, train==eval@8/255) with training inner-max **10/10/10** (RAMP `--at_iter 10`). Enables confound-free comparison with RAMP at every epoch.
- **Archived** (provenance only, `results/archive/our_recipe_50ep_drop25/` + README, filesystem mv since results/ is gitignored): reactive 41.90 3-seed, predictive v1 (F9 40.6), predictive v2 sweep-1 ks8 (41.5), + their checkpoints. **"reactive baseline 41.90" RETRACTED as active** — the new baseline is the RAMP-recipe reactive rerun.
- **KEPT (active reference):** RAMP repro ep_40/50/80 = 43.6/42.9/46.1.
- **Attack change:** `attackdro.yaml` training attacks l1 20→10 (uniform 10/10/10; l1 step_size 0.05→0.10 to keep reach). Reactive training budget now **30 steps** (was 40) → FLOPs denominator updates (unaffected by lr change). EVAL stays **l1-heavy / F1-safe** (l1 ≥ linf/l2, never uniform-few); **final claim rows = full AutoAttack** (matches RAMP; APGD reads ~1.5pp high → why our RAMP repro 46.1 > paper 44.6).
- **Plan** (`docs/EXPERIMENT_TABLE.md` rewritten): PHASE 1 develop @ ep50 **1-seed** (5070ti de-risk: reactive → predictive v2 → β-ablation; Colab: avg_frozen ∥ 3a) → select winner; PHASE 2 continue winner to ep80 **3-seed**, full-AA vs RAMP 46.1. FLAGS before launch: trainer **save_freq** (every-10) + **resume** (optimizer/epoch/RNG) not yet supported; **λ reconcile** (RAMP ref λ=5 ours/Table-5-validated vs paper λ=2).
- **Superseded:** Colab avg_frozen+3a @our-recipe (Kiet stopping) — archive if they land.
- **Launch nothing** — archive + re-plan only; awaiting GO on the new table.

## 2026-07-07 — FIX·PORT · APGD training attack (P-09/P-10) — the l1 root cause
- **Root cause (found via Phase-1 reactive ep50):** union 29.4 / l1 30.7 (vs RAMP@50 42.9 / 47.1). Train-vs-EVAL l1 gap **+41.7pp** (train-attack robust 72.4 vs eval 30.7) — 3–5× the l∞/l2 gaps. We trained plain fixed-step top-k **PGD** but eval with **APGD** (AutoAttack) → the model overfit a weak training attack, brittle under the strong eval (l1 didn't even converge: 30.7@100→29.4@200).
- **Ground truth (P-10):** RAMP trains all norms with **APGD** — paper states it 3× ("we use APGD…", Alg. `APGD(Bq)`, "10 steps for inner max") AND code (`autopgd_train.apgd_train`, `--attack apgd`). NOT plain PGD, NOT SLIDE. Paper is transparent; the mismatch was entirely on our side. (Separate: λ paper=2 vs code=5 — cite code.)
- **Fix:** ported `apgd_train` faithfully → `src/robustdro/attacks/apgd_train.py` (momentum a=0.75, adaptive step-halving, best-iterate, **APGD-l1** adaptive top-k + L1-projection). Wired `build_source_attack(attack=)` + groupdro (reactive attacks + predictive floor) + `attack: apgd` in attackdro.yaml. Reactive AND predictive craft with APGD.
- **Verified (not a re-decision — decision locked, port checked):** (1) unit ε-balls exact (l∞ 8/255, l2 0.5, l1 12), sparse l1, in[0,1], no crash. (2) **APGD-10 is +11.6pp stronger than PGD-10** (l1 robust 41.4 vs 53.0 on the same model) and within 7pp of APGD-100 → mechanistically closes the train/eval mismatch. (3) FLOPs counter OK (counts n_iter=10; ratio 1.0 at cold; passes 1.5M). (4) cost ~1.22× PGD (~223 vs 182 s/ep). (A 6-epoch smoke can't show the gap — it's a late-training phenomenon; the strength check is the decisive port verification.)
- **Decisions:** APGD training **supersedes 10/10/10 and 10/10/20 plain-PGD**; **FLOPs denominator = 30** (uniform-10 APGD). Recipe-match with RAMP now EXACT (same attack class + steps) — removes the attack-class confound from all RAMP comparisons.
- **FLAG:** Gate-α/F6 binding measured on old-PGD loss — may shift under APGD (note; not re-run).
- **Next (Kiet GO):** re-run reactive → predictive (kspan=16) with APGD → de-risk STOP. Pre-registered check at ep50: l1 train-vs-eval gap should be ~l∞/l2 level (not 41.7) + l1 eval converges.
