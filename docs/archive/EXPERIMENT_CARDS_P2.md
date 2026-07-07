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
- **RÀNG BUỘC (từ F/§0.1):** KHÔNG dùng hard max (`argmax`); MAX baseline = 25.0 chứng minh hard per-sample worst-norm sập (over-concentrate, clean/ℓ2 tụt). Phải giữ soft (T đủ lớn để không thoái hoá về argmax). **[SỬA 2026-07-03, xem POSITIONING §3]: 25.0 là số của locuslab MAX *checkpoint*; Croce&Hein Table 5 MAX tự-train = 44.0 ‡ (@8/255) → KHÔNG được nói "hard-max fails về nguyên tắc", chỉ được nói "ckpt locuslab MAX yếu". Trục soft>hard cần anchor MAX in-house (T→0, recipe ta) trước khi claim.**
- **Config:** `configs/bindaware_sample.yaml` kế thừa attackdro; override: `dro.objective: per_sample_soft`, `dro.temperature: T` (**sweep T ∈ {0.25, 0.5, 1, 2}** — 1 seed/T pilot, T tốt nhất → 3 seed [amendment #3]). Grouping/attacks GIỮ NGUYÊN. Code: forward 3 attack với `reduction='none'` → ma trận loss [B,3] → softmax hàng theo loss/T (detach weights) → weighted mean.
- **Amendment #1 — loss-distribution instrumentation (BẮT BUỘC trong pilot):** log per-norm per-sample loss distribution mỗi epoch (mean/std/p10/p50/p90 mỗi norm). Nếu thấy **scale bias** giữa norm (norm loss-scale lớn chiếm softmax bất kể binding) → thêm variant `dro.normalize_losses: zscore` (z-score per-group trước softmax) — **declared exception cho 1-variable rule** (đã khai trước, chỉ kích hoạt khi có bằng chứng scale bias từ pilot). **[F5 update 2026-07-02]: prior cho zscore TĂNG** — F5 cho thấy attack yếu bóp méo per-norm signal ở mức ranking; per-sample raw loss từ 3 attack train-time (PGD-20/topk) nhiều khả năng mang cùng norm-scale bias → soi `dist/*` logs của sweep pilots kỹ. Vẫn evidence-gated như đã khai. **[VERDICT 2026-07-03, Kiet]: zscore SKIPPED** — evidence từ T025: bias mild (p50 ℓ∞ .62/ℓ1 .49/ℓ2 .30), không inversion, ℓ1 vẫn cải thiện nhiều nhất (48.3) → gate không mở.
- **Change vs previous:** objective aggregation (group-scalar-q → per-sample-softmax), 1 thành phần, so với AttackDRO++ P1 / CARD-3a.
- **Expected:** nếu per-sample binding là đòn đúng → +2–4pp worst-∪ vs AVG/AttackDRO++ (38.7), tiệm cận/vượt MSD 42.5; clean có thể tụt 1–2pp khi T nhỏ. Nếu T nhỏ mà ℓ2 sụp / clean tụt mạnh ⇒ quá lạnh (gần MAX) → tăng T.
- **Decision rule:** so 4 chiều: vs AVG/AttackDRO++ 38.7 · vs CARD-3a · vs MSD 42.5 · sweep-T monotonic check (AVG↔MAX). ≥42.5 → winner, 3 seed + figs. **[Amendment #2] 3b > 3a nhưng < 42.5 → phần MSD edge còn lại nằm ở per-STEP selection BÊN TRONG attack generation** (MSD chọn norm mỗi PGD step; 3b chỉ chọn sau khi attack xong) → ghi nhận là **mechanism finding (publishable)**: "binding-aware sau-attack thu hẹp nhưng không đóng gap; phần còn lại do per-step norm selection". ≈38.7 ⇒ union không bị chặn bởi lựa-chọn-norm-per-sample mà bởi độ mạnh tuyệt đối từng norm (→ ℓ1 source mạnh hơn / geometry grouping).
- **[Amendment #2] Lit-check trước novelty claim — ĐÃ CHẠY 2026-07-02 (sơ bộ, cần đọc kỹ trước khi claim):**
  - Tramèr&Boneh 2019 (arXiv 1904.13000): AVG + hard-MAX per-sample — 2 cực của 3b, KHÔNG có soft/temperature.
  - MSD (arXiv 1909.04068): chọn norm **per-STEP bên trong PGD** — khác trục với 3b (per-sample sau attack).
  - **E-AT** (Croce&Hein, arXiv 2105.12508): fine-tune nhanh sang union — không phải soft weighting.
  - **✅ ĐÃ ĐỌC (2026-07-02): "Adaptive Smoothness-weighted AT for Multiple Perturbations" (Xiao, Qin, Fan, Wu, Wang, Luo; arXiv 2210.00557). POSITIONING NOTE (5 dòng):**
    1. Họ weight **per-TYPE (group-level)** giữa ℓ1/ℓ2/ℓ∞ — cùng granularity với CARD-3a/AttackDRO++, KHÁC 3b (per-sample).
    2. Signal của họ = **smoothness của loss landscape** per adversary ("ℓ1, ℓ2, ℓ∞ adversaries give different contributions to the smoothness") — không phải loss value (P1), robust-acc (3a), hay per-sample loss (3b).
    3. Aggregation = **weighted average per-type**, không có temperature/AVG↔MAX axis; có stability-based excess-risk theory — mạnh hơn ta về lý thuyết ở trục đó.
    4. SOTA claim trên CIFAR-10/100 mixed perturbations → khi so sánh phải cite; nếu 3b thắng bảng ta, KHÔNG được claim vượt họ mà chưa chạy số họ dưới harness ta (golden rule #3).
    5. Novelty còn lại của 3b: **per-SAMPLE softmax + temperature nội suy AVG↔MAX + binding-aware framing gắn F4/F5** — hẹp nhưng distinct; frame "mechanism probe", không phải "new SOTA method".
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
- **RESULT + VERDICT (2026-07-03, Kiet): 41.5** (ℓ1/ℓ2 giống hệt 3a; chênh −0.8 toàn ở ℓ∞). Pre-registration 42.0–42.6 trượt biên dưới 0.5. **Confound checks (49k data, EMA/τ) = PARKED, không đầu tư thêm** — kết luận pragmatic: **val-calibrated ≤ probe; với F6, TRACKING binding norm thắng PRECISION đo nó**. Caveat 1 dòng: v2 train 49k (mất 1000 ảnh holdout) — riêng nó có thể giải thích −0.8, nên "≤" là kết luận an toàn duy nhất. **Không cấp GPU slot nào thêm cho v2.**
- **Cost:** calibration ~35s/5 epochs (không đáng kể); run = ~2h/seed như 3a.
- **ORDERING (Kiet 2026-07-02, chốt):** sau khi 3a pilot lands: **slot 1 = 3b sweep** (armed, `p3bsweep`) · **slot 2 = 3a-v2 pilot chạy SONG SONG** (armed, `p3av2` — gate: 3a landed VÀ p1 attackdro xong; nếu p1 còn chạy thì 3a-v2 đợi, giữ ≤2 trainer). **Mục tiêu: đủ ma trận 2×2** — signal (probe vs val-APGD) × aggregation (group vs per-sample). Merged variant (per-sample + val-calibrated) CHỈ build nếu một trục cho tín hiệu.

## CARD-5 · `max_inhouse` — MAX anchor in-house   ✅ LANDED 2026-07-04
- **Mục đích:** anchor T→0 SẠCH cho trục T (locuslab MAX ckpt 25.0 = artifact; C&H retrain 44.0 ‡ — POSITIONING §3). Tier-1 row cho bảng cuối.
- **Config:** `configs/max_inhouse.yaml` — 1 thay đổi: `objective: per_sample_max` (hard argmax per sample; BN vẫn nhận cả 3 forwards như per_sample_soft — cô lập đúng trục soft-vs-hard). q logged = selection frequency.
- **PRE-REGISTERED (Kiet, TRƯỚC khi chạy): 41.5–43.5** → cold-T finding ("soft>hard") CHẾT ở dạng hiện tại; **<38** → finding SỐNG với anchor sạch.
- **RESULT + VERDICT (2026-07-04): worst-∪ 43.9** (clean 78.9 · ℓ∞ 45.5 · ℓ2 61.6 · ℓ1 47.2) — **NGOÀI band pre-registered, phía CAO**. (a) "Cold beats hard" CHẾT dứt khoát; trục T gần-monotone lạnh→ấm 43.9→43.0→42.6→41.5→42.1. (b) **Prediction miss GIẢI THÍCH ĐƯỢC (sau full RAMP read): 43.9 @0.03 khớp C&H MAX retrain 44.0±0.7 ‡ @8/255 trong 1σ — hard-MAX genuinely ~44 với recipe đúng, KHÔNG phải eval error; đồng thời = external cross-validation cho recipe in-house ta → F7 evidence thứ 3.** (c) clean 78.9 thấp nhất bảng = đúng trade hard-max. 1 seed, apgd.

## CARD-6 · `eat_compose` — E-AT composition pilot   🔒 GATED (approved in principle 2026-07-03; chạy CHỈ SAU mandatory queue: seeds + MAX anchor + standard AA)
- **Setup:** locuslab LINF ckpt → (a) vanilla E-AT fine-tune 3 epochs (recipe họ, eval protocol eps ta) vs (b) cùng budget, **per-sample soft weighting** thay fixed extreme-norms alternation. 1 seed mỗi nhánh, ~phút-scale.
- **PRE-REGISTERED readings:** (b) > (a) +1pp → composability demonstrated, C2 nâng thành "orthogonal module" · (b) ≈ (a) → protocol-ceiling evidence · (b) < (a) → geometric prior thắng adaptive trong fine-tune regime — limitation trung thực, report.
- **RAMP composition: future-work** — NHƯNG note 2026-07-04: CARD-8 sẽ cho ta ckpt RAMP local (ep_80) → composition pilot khả thi về mặt kỹ thuật nếu còn budget sau mandatory queue; vẫn KHÔNG tự launch.

## CARD-7 · `msd_inhouse` — MSD objective, OUR recipe   ⬇ DOWNGRADED to OPTIONAL (Kiet 2026-07-04, sau full RAMP read) — vẫn armed `p7msd`, chạy khi GPU rảnh
- **Downgrade rationale:** bảng from-scratch giờ CITE C&H 5-seed ‡ (đúng methodology RAMP) → run này KHÔNG còn gate claim nào; giá trị còn lại = 1 điểm MSD recipe-controlled (nice-to-have cho tier-1) + F7 evidence thứ 4. Giữ trong queue vì gần như free (GPU overnight); nếu cần slot cho việc quan trọng hơn → hủy không tiếc.
- **Mục đích (gốc):** điểm F7 + hàng MSD tier-1 cho bảng recipe-controlled.
- **Config:** `configs/msd_inhouse.yaml` — objective `msd`, faithful port `msd_v0` từ robust_union **commit ef3419493188e2560b57fab87bdbf42ce8c0287e** (per-step: 1 grad + 3 candidate fwd, chọn max-loss per-sample; steps=50, alphas .003/.05/.05, k~U{5,20}); còn lại recipe ta. Smoked ✓. 1 seed, ~6h.
- **PRE-REGISTERED (strategy side, giữ nguyên):** 43.3–44.5 → claim re-scope sang recipe-controlled ladder; ≤42.5 → "match" đứng vững hơn. (Với claim ladder mới, đọc chủ yếu là: có tái tạo được cụm ~44 dưới recipe ta không.)

## CARD-8 · `ramp_verify` — RAMP protocol-verification, THEIR code + recipe   🔄 RUNNING (Kiet tự launch 2026-07-04 08:25 — đảo quyết định P3-đóng sau khi đọc full paper)
- **Mục đích:** protocol-verify outlier duy nhất không có public ckpt (RAMP 44.6±0.6 ‡). KHÔNG phải retrain-baseline (C&H vẫn ‡ cited) — row từ run này = "our reproduction of RAMP (1 seed)", tier riêng.
- **Setup:** upstream clone `uiuc-focal-lab/RAMP` @be4971f vào `external/RAMP/`; patch DUY NHẤT = `import copy` thiếu trong `utils.py::gp()` (upstream bug, zero algorithmic change — docs/RAMP_BASELINE.md). Official cmd seed-0: `RAMP.py --lr-max 0.05 --lr-schedule=static --at_iter 10 --epochs 80 --kl --max --final_eval --gp --lbd 5` — recipe HỌ nguyên vẹn, wrapper W&B của ta chỉ log (scripts/ramp/train_ramp_with_wandb.py).
- **Eval kép armed (scripts/ramp/):** apgd + standard AA × {ε∞=0.03 (protocol ta), 8/255 (protocol họ)} → vừa so được với 44.6±0.6 reported, vừa có số protocol-ta để định vị tier.
- **Ckpt:** `external/RAMP/trained_models/RAMP_beta_0.5_lbd_5_0/ep_80_0.pth`. ETA ~14h+ (80 ep, đang chia GPU với finalpipe). W&B run 4oek4cr7.
- **PRE-REGISTERED (strategy side, 2026-07-04 — registered TRƯỚC khi lands, per E1). Decision variable = RAMP-repro @8/255, 1-seed, std-AA:**
  - within **44.6±0.6** → reproduce OK; 44.6 là bar thật.
  - **43.3–44.0** → mild under-repro; GIỮ 44.6 làm bar, note caveat.
  - **<43.3** → **điều tra bug TRƯỚC (E9)**; KHÔNG BAO GIỜ dùng 1-seed repro để hạ RAMP; nếu clean thì protocol-ceiling mạnh lên nhưng cited 44.6±0.6 VẪN là official bar.
  - **>45.2** → nghi protocol/eval edge; verify trước khi dùng.
- **INVARIANT (bất biến, ghi vào bảng cuối):** 1-seed repro của ta **KHÔNG BAO GIỜ thay** số ‡cited 5-seed 44.6±0.6; row luôn là **"our 1-seed reproduction of RAMP"**. Mục đích = (a) validate ta chạy code họ ĐÚNG trước mọi composition (CARD-8b), (b) một điểm protocol-verify @0.03.
- **CARD-8b (composition, future):** chỉ khi CARD-8 = reproduce OK VÀ còn budget sau mandatory queue. Không tự launch. → full card ngay dưới.

## CARD-8b · `ramp_bindaware` — RAMP loss + binding-aware T-softmax   🟡 NEXT-UP, GATED · ⚠ SHOW KIET TRƯỚC KHI LAUNCH (E1, drafted 2026-07-04)
- **Vị trí chiến lược:** đây là run quyết định lane của G2′ (2026-08-02). Đây là hiện thân của PIVOT method-forward: thay vì chỉ *phân tích* gap (F5/F6: mọi aggregation hiện tại static + sample-agnostic), ta *vá* nó — ghép observable per-sample binding structure của ta vào loss SOTA (RAMP).
- **Gate (bắt buộc, cả 3):** (1) CARD-8 repro PASS band (@8/255 ∈ 44.6±0.6, hoặc ≥43.3 mild-repro) — nếu CARD-8 <43.3 thì ĐIỀU TRA BUG trước, KHÔNG chạy 8b trên nền chưa verify; (2) mandatory queue xong (finalist seeds + std-AA); (3) **Kiet duyệt design option + strategy-side pre-registration TRƯỚC launch.**
- **Cơ chế RAMP hiện tại (điểm can thiệp):** RAMP Eq. 2 dùng **hard L_max** term (per-sample max loss qua các norm) + logit-pairing (KL) + fixed λ (lbd=5) trên gradient-projection. L_max = "chọn norm tệ nhất mỗi sample, trọng số cứng". ĐÂY là chỗ static/sample-agnostic mà T-dial của ta địa chỉ hoá.
- **DESIGN OPTIONS (Kiet chọn 1 — hoặc chạy A rồi B nếu A có signal):**
  - **Opt A — REPLACE (thay L_max bằng per-sample soft T-softmax):** `L = Σ_norm w_i · ℓ_i`, `w = softmax(ℓ / T)` per-sample, T lấy từ sweep của ta (T=0.25 best in-house). Giữ nguyên KL + GP + λ của RAMP. Cô lập đúng 1 biến: hard-max → adaptive-soft. Pre-registered đọc: >repro +0.5 → adaptive thắng static ⇒ method paper.
  - **Opt B — AUGMENT (giữ L_max, thêm binding-aware regularizer nhẹ):** L_ramp + β · (per-sample binding-consistency term). Ít rủi ro phá RAMP, nhưng khó quy signal về đúng trục adaptive-weighting. Dùng nếu A phá training (clean sụp).
  - **Khuyến nghị Claude:** chạy **Opt A** trước (isolation sạch nhất, đúng câu chuyện "adaptive thay static"); B là fallback nếu A unstable.
- **Cô lập biến:** baseline so sánh = **CARD-8 RAMP-repro chính nó** (cùng code/recipe/seed), CHỈ đổi L_max→T-softmax. Δ = pure effect của adaptive weighting trên nền SOTA. KHÔNG so với số ‡ (khác protocol/seed-count).
- **PRE-REGISTERED bands (strategy side điền TRƯỚC launch — placeholder, chờ Kiet):** đề xuất: Δ vs repro **> +0.5 (1 seed) → METHOD lane** (G2′ pass) · **−0.5..+0.5 → within noise → MECHANISM lane** (adaptive ≈ static ở regime này = honest limitation, vẫn là finding) · **< −0.5 → static thắng adaptive trong loss-shaping regime** — report trung thực, mechanism lane.
- **Cost:** RAMP train ~14h/seed (80 ep) → 8b = 1 seed pilot cùng cost. Cần slot GPU sau mandatory queue; KHÔNG preempt. Ckpt gốc RAMP (CARD-8 ep_80) tái dùng được cho warm-start experiments nếu muốn rẻ hơn.
- **Eval:** parity tuyệt đối — eval_union / standard pack, cả 0.03 và 8/255 như CARD-8.

## CARD-9 · `eat_inhouse` — E-AT fine-tune, in-house row   🟡 GATED · card-first, ⚠ pre-registration TRƯỚC launch (drafted 2026-07-04, Part C.2)
- **Mục đích:** hàng E-AT tier-1 in-house — mắt xích còn thiếu để bảng recipe-controlled đối xứng, và là differentiation trực tiếp nhất (E-AT = fixed geometric prescription: train ℓ∞+ℓ1, ℓ2 free-qua-Thm-3.1; ta = adaptive per-sample weighting). Có in-house E-AT nghĩa là so "fixed geometric vs adaptive" TRONG CÙNG recipe.
- **Feasibility (Part C xác nhận):** `external/RAMP/eat_train.py` runnable, `RAMP/models/pretr_{Linf,L1,L2}.pth` = starting ckpts sẵn, `robustbench` import OK. Fine-tune **3 epochs** = rẻ (~phút→chục-phút scale, KHÔNG phải 14h như RAMP from-scratch).
- **Setup (dự kiến — Kiet chốt):** chạy `eat_train.py` recipe HỌ từ pretr start; eval QUA eval_union/standard pack của ta (parity invariant), cả 0.03 và 8/255. 1 seed pilot trước.
- **Gate:** sau mandatory queue (finalist seeds + std-AA); KHÔNG preempt CARD-8/8b. **Card-first: Kiet + strategy-side pre-register bands TRƯỚC khi launch.**
- **PRE-REGISTERED (placeholder — chờ strategy side):** in-house E-AT @0.03 kỳ vọng ~42–43 (khớp ‡42.4 @8/255, đọc ≥ ở 0.03). Đọc chính: so với T025 43.0 / max_inhouse 43.9 in-house — E-AT ≈ hay < adaptive của ta TRONG cùng recipe? Đây là bằng chứng "adaptive > fixed-geometric" nếu ta ≥ E-AT.
- **KHÔNG tự launch.**

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

