# EXPERIMENT_CARDS_P2.md — cards viết sẵn cho P2 (theo WORKFLOW_RULES E1)
*Nguyên tắc: simple-first, mỗi card đổi đúng 1 thành phần, expectation + decision rule khai TRƯỚC khi chạy.*
*Thứ tự thực thi phụ thuộc verdict P1 (xem §0). Claude Code: copy card vào PROGRESS.md khi launch, điền kết quả theo E8.*

---

## §0 · ROUTING THEO VERDICT P1 (đọc trước)
Nhìn 3 thứ của seed-0 APGD: worst-∪ · **cột ℓ2** · norm binding.
- **worst-∪ > 42.5 (beat MSD):** nghi bug trước (check eval/ckpt/ε). Nếu thật → chạy đủ 3 seed, rồi CARD-1 để cô lập vì sao thắng.
- **38.7 ≤ worst-∪ ≤ 42.5:** đường chính → CARD-1 rồi CARD-2.
- **worst-∪ < 38.7 VÀ ℓ2 sụp (<45%):** kích hoạt CARD-4 (q-floor, contingency) song song CARD-1.
- **worst-∪ < 38.7 nhưng ℓ2 vẫn cao:** vấn đề không phải starvation → CARD-1 rồi CARD-3.

---

## §0.1 · P1 VERDICT + RE-ROUTE (2026-07-02, seed-0 APGD)
**Seed-0 AttackDRO++:** clean 80.7 · ℓ∞ 46.2 · **ℓ2 63.1** · ℓ1 41.2 · **worst-∪ 38.7** · q_final [ℓ∞ .83, ℓ2 .00, ℓ1 .17].
→ **NO-GO vs MSD (−3.8); EXACT TIE vs AVG (38.7).** Binding norm = **ℓ1** (weakest, 41.2). ℓ2 = 63.1 **HIGH despite q_ℓ2 = 0.00** → ℓ2 robustness is FREE (spillover), not at risk.
Routing (§0) fires the last branch's *spirit* (worst-∪ ties AVG, ℓ2 stays high): starvation is NOT the problem → objective/signal is. Re-route:
- **CARD-4 (qfloor) — FORMALLY OFF.** Activation gate (ℓ2 APGD < ~45%) NOT met (ℓ2 = 63.1). ℓ2 is free; a floor would spend capacity on an already-solved norm.
- **CARD-2 (cvar) — DEPRIORITIZED.** Seed-0 indicts *group-level scalar* signals: q-by-loss piled onto ℓ∞ (highest loss, best ℓ∞ 46.2) but under-weighted the binding ℓ1. CVaR still reweights group-level scalars → same misalignment risk. Keep as fallback, not next.
- **CARD-3 → split.** The mechanism failure is signal↔objective misalignment → attack binding-aware FIRST.
  - **CARD-3a** = group binding-aware (this doc's original CARD-3): q ∝ softmax(−robust_acc_g/τ). **LAUNCHED first, 1-seed pilot** (`configs/bindaware.yaml`, tmux `p2pipe`).
  - **CARD-3b** = per-sample SOFT binding-aware (NEW, drafted below; NOT launched — Kiet reviews first).
- **CARD-1 (avg_frozen)** — runs after the pilot in `p2pipe` (needed for the final table regardless; also the DRO⟺uniform ablation).

---

## CARD-1 · `avg_frozen` — Đóng băng q (ablation DRO ⟺ AVG in-house)  ⭐ chạy đầu
- **Hypothesis:** Tách tác dụng của q-update (DRO) khỏi recipe. Nếu AttackDRO++ ≈ frozen-uniform → DRO không đóng góp gì cho worst-∪; nếu hơn → DRO có tác dụng thật.
- **Config:** `configs/avg_frozen.yaml` kế thừa `attackdro.yaml`, override duy nhất: `dro.freeze_q: true` (q = 1/3 cố định). KHÔNG đổi gì khác.
- **Change vs previous:** 1 boolean (q update on→off) so với run AttackDRO++ P1.
- **Expected:** frozen-uniform worst-∪ ≈ 36–40 (gần AVG locuslab 38.7 nhưng recipe khác); AttackDRO++ chênh ±2pp so với nó.
- **Decision rule:** ΔDRO = AttackDRO++ − frozen. Δ > +1.5pp → DRO giúp, sang CARD-2 để đẩy thêm. |Δ| ≤ 1.5pp → DRO trung tính, mechanism story + CARD-2 (đổi objective là hy vọng chính). Δ < −1.5pp → DRO có hại, mechanism story mạnh (q-collapse phản tác dụng) + CARD-4 vào diện xét.
- **Bonus:** run này CHÍNH LÀ AVG in-house → thay AVG locuslab trong bảng cuối (giết recipe confound).
- **Cost:** ~2h/seed × 3 seed (5070ti, tmux `avgfrozen`). Seeds {0,1,2}.

## CARD-2 · `cvar` — Đổi objective: soft-DRO → CVaR   ⏸ DEPRIORITIZED (§0.1: group-level scalar, same misalignment risk; fallback only)
- **Hypothesis (H2):** worst-case cần objective nhắm tail, không phải reweight-by-loss. CVaR-α trên per-sample (hoặc per-group) loss đẩy worst-∪ tốt hơn GroupDRO softmax.
- **Config:** `configs/cvar.yaml` kế thừa attackdro; override: `dro.objective: cvar`, `dro.cvar_alpha: 0.3` (sweep {0.2,0.3,0.5} 1 seed trước, chọn 1 giá trị chạy 3 seed). Grouping GIỮ NGUYÊN.
- **Change vs previous:** objective (1 thành phần), so với AttackDRO++ P1.
- **Expected:** +1–3pp worst-∪ vs AttackDRO++ nếu H2 đúng; clean acc có thể tụt 1–2pp (tail focus).
- **Decision rule:** ≥ MSD 42.5 → ứng viên winner, chạy 3 seed đầy đủ + mechanism figs. > AttackDRO++ nhưng < 42.5 → giữ làm best-so-far, sang CARD-3. ≤ AttackDRO++ → H2 sai cho setting này, ghi nhận, sang CARD-3.
- **Cost:** sweep ~6h + 3 seed ~6h. Colab cho sweep nếu 5070ti bận.

## CARD-3a · `bindaware` — GROUP binding-aware weighting   ▶ LAUNCHED (1-seed pilot, p2pipe)
*(was CARD-3; split per §0.1. This is the group-level variant.)*
- **Hypothesis (H1/F2):** weight nhóm theo đóng góp vào worst-case (per-norm robust acc trên probe: nhóm nào robust THẤP nhất được weight cao) thay vì raw loss — nhắm thẳng norm binding, tránh bias loss-scale giữa các norm.
- **Config:** `configs/bindaware.yaml`; override: `dro.weight_signal: robust_acc` (q ∝ softmax(−robust_acc_g / τ), τ sweep {0.05,0.1}). Objective giữ GroupDRO.
- **Change vs previous:** weighting signal (1 thành phần), so với AttackDRO++ P1.
- **Expected:** khác biệt lớn nhất nếu q-by-loss đang lệch khỏi q-by-binding; nếu F3 đã "vô tình binding-aware" (q dồn ℓ∞ = norm binding) thì Δ nhỏ — bản thân điều đó là finding.
- **Decision rule:** so 3 chiều: vs AttackDRO++ (signal nào tốt hơn) · vs CARD-1 (hơn uniform không) · vs 42.5. Tốt nhất trong các card → winner; không → mechanism paper có đủ 3 objective/signal so sánh sạch.
- **Cost:** ~2h/seed; sweep τ 1 seed trước.

## CARD-3b · `bindaware_sample` — PER-SAMPLE SOFT binding-aware   ✅ APPROVED 2026-07-02 (3 amendments; implement+smoke NGAY, launch SAU khi 3a pilot xong — không chạy 3 trainer cùng lúc)
- **Hypothesis (H1′):** worst-∪ là metric **per-sample** (mỗi ảnh lấy norm phá nó mạnh nhất). Trọng số nên ở mức **per-sample × per-norm**, không phải 1 scalar q chung cho cả nhóm. Dùng softmax theo nhiệt độ trên per-sample per-norm loss để nhắm norm binding của TỪNG ảnh, **nội suy AVG↔MAX** — kỳ vọng vượt cả hai vì AVG bỏ qua norm binding còn MAX cứng thì over-focus (đã chứng minh hỏng: MAX=25.0).
- **Cơ chế:** với ảnh i, có 3 loss `ℓ_{i,linf}, ℓ_{i,l2}, ℓ_{i,l1}` (CE dưới attack mỗi norm). Weight `w_{i,g} = softmax_g(ℓ_{i,g} / T)`; loss batch = `mean_i Σ_g w_{i,g}·ℓ_{i,g}`. **T→∞ ⇒ AVG; T→0 ⇒ hard-MAX per-sample.** Chọn T trung gian để soft.
- **RÀNG BUỘC (từ F/§0.1):** KHÔNG dùng hard max (`argmax`); MAX baseline = 25.0 chứng minh hard per-sample worst-norm sập (over-concentrate, clean/ℓ2 tụt). Phải giữ soft (T đủ lớn để không thoái hoá về argmax).
- **Config:** `configs/bindaware_sample.yaml` kế thừa attackdro; override: `dro.objective: per_sample_soft`, `dro.temperature: T` (**sweep T ∈ {0.25, 0.5, 1, 2}** — 1 seed/T pilot, T tốt nhất → 3 seed [amendment #3]). Grouping/attacks GIỮ NGUYÊN. Code: forward 3 attack với `reduction='none'` → ma trận loss [B,3] → softmax hàng theo loss/T (detach weights) → weighted mean.
- **Amendment #1 — loss-distribution instrumentation (BẮT BUỘC trong pilot):** log per-norm per-sample loss distribution mỗi epoch (mean/std/p10/p50/p90 mỗi norm). Nếu thấy **scale bias** giữa norm (norm loss-scale lớn chiếm softmax bất kể binding) → thêm variant `dro.normalize_losses: zscore` (z-score per-group trước softmax) — **declared exception cho 1-variable rule** (đã khai trước, chỉ kích hoạt khi có bằng chứng scale bias từ pilot). **[F5 update 2026-07-02]: prior cho zscore TĂNG** — F5 cho thấy attack yếu bóp méo per-norm signal ở mức ranking; per-sample raw loss từ 3 attack train-time (PGD-20/topk) nhiều khả năng mang cùng norm-scale bias → soi `dist/*` logs của sweep pilots kỹ. Vẫn evidence-gated như đã khai.
- **Change vs previous:** objective aggregation (group-scalar-q → per-sample-softmax), 1 thành phần, so với AttackDRO++ P1 / CARD-3a.
- **Expected:** nếu per-sample binding là đòn đúng → +2–4pp worst-∪ vs AVG/AttackDRO++ (38.7), tiệm cận/vượt MSD 42.5; clean có thể tụt 1–2pp khi T nhỏ. Nếu T nhỏ mà ℓ2 sụp / clean tụt mạnh ⇒ quá lạnh (gần MAX) → tăng T.
- **Decision rule:** so 4 chiều: vs AVG/AttackDRO++ 38.7 · vs CARD-3a · vs MSD 42.5 · sweep-T monotonic check (AVG↔MAX). ≥42.5 → winner, 3 seed + figs. **[Amendment #2] 3b > 3a nhưng < 42.5 → phần MSD edge còn lại nằm ở per-STEP selection BÊN TRONG attack generation** (MSD chọn norm mỗi PGD step; 3b chỉ chọn sau khi attack xong) → ghi nhận là **mechanism finding (publishable)**: "binding-aware sau-attack thu hẹp nhưng không đóng gap; phần còn lại do per-step norm selection". ≈38.7 ⇒ union không bị chặn bởi lựa-chọn-norm-per-sample mà bởi độ mạnh tuyệt đối từng norm (→ ℓ1 source mạnh hơn / geometry grouping).
- **[Amendment #2] Lit-check trước novelty claim — ĐÃ CHẠY 2026-07-02 (sơ bộ, cần đọc kỹ trước khi claim):**
  - Tramèr&Boneh 2019 (arXiv 1904.13000): AVG + hard-MAX per-sample — 2 cực của 3b, KHÔNG có soft/temperature.
  - MSD (arXiv 1909.04068): chọn norm **per-STEP bên trong PGD** — khác trục với 3b (per-sample sau attack).
  - **E-AT** (Croce&Hein, arXiv 2105.12508): fine-tune nhanh sang union — không phải soft weighting.
  - **⚠ GẦN NHẤT: "Adaptive Smoothness-weighted AT for Multiple Perturbations" (arXiv 2210.00557)** — adaptive weights giữa perturbation types, nhưng **per-TYPE (group-level), signal = smoothness**; 3b là **per-SAMPLE, signal = loss, temperature nội suy AVG↔MAX**. PHẢI đọc kỹ + cite trước khi claim novelty; positioning: "per-sample soft aggregation" vs their "per-type adaptive weights".
  - Kết luận sơ bộ: temperature-softmax per-sample over norms chưa thấy được publish trực tiếp, nhưng novelty hẹp → frame là **mechanism probe** (AVG↔MAX axis), không phải "new method" trừ khi thắng rõ.
- **Cost:** sweep 4×T ~1 seed mỗi (~8h GPU-contended) + 3 seed cho T thắng (~6h).
- **SCHEDULING (Kiet 2026-07-02):** implement + smoke NGAY; **launch sweep SAU khi 3a pilot APGD lands** (giữ ≤2 trainer/GPU). CARD-1 avg_frozen ×3 chuyển sang **Colab**.

## CARD-3a-v2 · `bindaware_v2` — VAL-CALIBRATED q   ✅ GATE OPEN (diagnostic 2026-07-02), evidence-gated per Kiet
- **FREE DIAGNOSTIC (đã chạy, seed-0 best ckpt @epoch 33, không tốn GPU):**
  | norm | probe (PGD-20, in-training) | APGD (CE+T, 100it) | bias |
  |---|---|---|---|
  | ℓ∞ | 50.3 | 46.2 | +4.1 |
  | ℓ2 | 65.7 | 63.1 | +2.6 |
  | ℓ1 | 53.5 | 41.2 | **+12.3** |
  → ℓ1 bias vượt trội (đúng dự đoán F1: PGD-ℓ1-topk yếu → overstate) VÀ **RANKING INVERSION**: probe nói weakest = ℓ∞ (50.3<53.5), APGD nói weakest = **ℓ1** (41.2<46.2). ⇒ **GATE OPEN.**
- **⚠ PRE-REGISTERED PREDICTION cho 3a pilot (ghi TRƯỚC khi có kết quả):** CARD-3a đang chạy dùng CHÍNH probe này làm signal → q của nó sẽ dồn vào ℓ∞ (norm probe cho là yếu nhất) thay vì ℓ1 (binding thật) → 3a pilot dự đoán **underperform / ≈38.7**, với mechanism = signal-bias, KHÔNG phải "binding-aware sai hướng". Nếu 3a ra đúng thế, đó là evidence cho 3a-v2, không phải bằng chứng chống binding-aware.
- **Hypothesis:** q calibrate bằng attack họ-eval (APGD-CE rút gọn) trên val held-out khớp ranking với eval thật → binding-aware signal ĐÚNG norm.
- **Config:** `configs/bindaware_v2.yaml` kế thừa bindaware; override: `weight_signal: val_apgd`, `dataset.val_holdout: 1000` (1000 ảnh CUỐI train set, loại khỏi training), `val_every: 5` (epoch), `val_iters: 30` (APGD-CE only — KHÔNG full DDN/CW ensemble vì cost; DDN-ℓ2 chỉ thêm nếu có evidence sau), `ema_beta: 0.5` (EMA trên acc trước softmax), `tau: 0.1`. q = softmax(−acc_ema/τ).
- **Change vs previous:** signal SOURCE (in-training PGD probe trên test → APGD-CE trên val held-out), 1 thành phần vs CARD-3a.
- **CAVEATS (bắt buộc disclose khi viết):** (1) q được tune bằng attack CÙNG HỌ với eval — hợp lệ vì val≠test và chỉ 3 scalar/5-epoch rò rỉ, nhưng PHẢI ghi rõ trong paper; (2) vẫn group-level → trần F4 (degeneration-to-AVG) vẫn áp dụng — 3a-v2 **bổ trợ, không thay thế 3b**; (3) train set nhỏ hơn 1000 ảnh (49k) — chênh lệch nhỏ nhưng ghi nhận.
- **Decision rule:** so vs 3a pilot (signal đúng có cứu group-level không) · vs 38.7 · vs 42.5. 3a-v2 > 3a ⇒ signal-bias là thủ phạm chính của 3a → merge signal này vào 3b (per-sample + val-calibrated). 3a-v2 ≈ 3a ≈ 38.7 ⇒ F4 ceiling xác nhận mạnh (group-level chết dù signal đúng) → toàn lực 3b.
- **Cost:** calibration ~35s/5 epochs (không đáng kể); run = ~2h/seed như 3a.
- **ORDERING (Kiet 2026-07-02, chốt):** sau khi 3a pilot lands: **slot 1 = 3b sweep** (armed, `p3bsweep`) · **slot 2 = 3a-v2 pilot chạy SONG SONG** (armed, `p3av2` — gate: 3a landed VÀ p1 attackdro xong; nếu p1 còn chạy thì 3a-v2 đợi, giữ ≤2 trainer). **Mục tiêu: đủ ma trận 2×2** — signal (probe vs val-APGD) × aggregation (group vs per-sample). Merged variant (per-sample + val-calibrated) CHỈ build nếu một trục cho tín hiệu.

## CARD-4 · `qfloor` — CONTINGENCY   ⛔ FORMALLY OFF (§0.1: gate not met, ℓ2 APGD = 63.1 ≥ 45%)
- **Kích hoạt KHI VÀ CHỈ KHI:** cột ℓ2 của AttackDRO++ APGD < ~45% (ℓ2 thật sự sụp vì starvation) — không chạy trước đó. **[2026-07-02: ℓ2 = 63.1 → gate KHÔNG đạt, card OFF. ℓ2 free/spillover, không cần sàn.]**
- **Hypothesis:** hard concentration cần sàn; q_g ≥ floor giữ ℓ2 không chết mà vẫn dồn lực vào norm khó.
- **Config:** `dro.q_floor: 0.1` (1 dòng trong losses.py: clamp sau softmax + renormalize).
- **Decision rule:** ℓ2 hồi ≥ baseline VÀ worst-∪ ≥ AttackDRO++ → giữ như một thành phần của winner; ngược lại bỏ, và "floor không cứu được" cũng là data point.
- **Cost:** ~2h/seed.

---

## Ghi chú chung
- Mọi card: eval bằng `eval_union` APGD n=1000; số rời repo phải là `standard`.
- Mechanism instrumentation BẬT cho mọi run: log q-trajectory (hoặc weight-trajectory), per-norm probe mỗi epoch → figs cho paper.
- 1 seed để dò → chỉ chạy đủ 3 seed cho config có tín hiệu. 20 seed chỉ cho winner cuối (P3).
- Sau mỗi card: RESULT block (E8) vào PROGRESS.md + cập nhật PROJECT_MEMORY §5/§9 nếu có finding.

