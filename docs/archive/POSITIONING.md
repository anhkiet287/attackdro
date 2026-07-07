# POSITIONING.md — related-work positioning + SOTA landscape (living doc)
*Last updated: 2026-07-04. Mọi số ‡ = cited từ paper gốc, chưa re-eval dưới harness ta.*

## 1 · SOTA landscape, from-scratch RN-18/PreActRN-18, union (ℓ∞, ℓ2, ℓ1)
**BASELINE METHODOLOGY (Kiet lock 2026-07-04, sau khi đọc full RAMP paper — adopt đúng cách RAMP làm):** bảng from-scratch cuối CITE ‡ toàn bộ baselines (5 seeds, ε∞=8/255) từ C&H 2022 + RAMP 2024, **KHÔNG retrain** chúng; số in-house của ta (recipe ta, ε∞=0.03) nằm **tier RIÊNG với protocol caveat tường minh** — chính xác như RAMP tách from-scratch row của họ khỏi cited baselines.

**⚠ Protocol khác nhau — KHÔNG so trực tiếp với bảng ta:** mọi số dưới đây ở **ε∞ = 8/255 (0.0314)**, ta lock **0.03**. 8/255 > 0.03 ⇒ eval của họ *khắc nghiệt hơn một chút* trên trục ℓ∞ ⇒ số của họ đọc dưới protocol ta chỉ có thể **cao hơn hoặc bằng**. Hướng lệch này KHÔNG cứu được mình: nếu họ ≥ ta ở 8/255 thì họ vẫn ≥ ta ở 0.03.

| method | union ‡ (5 seeds, ε∞=8/255) | nguồn | ghi chú |
|---|---|---|---|
| **RAMP (NeurIPS'24)** | **44.6 ± 0.6** (clean 81.2) | [Jiang2024-RAMP] | **presumptive SOTA from-scratch**; no public ckpt |
| MAX (C&H retrain) | 44.0 ± 0.7 | [Croce2022-EAT] Table 5 | ≠ locuslab MAX ckpt (25.0 dưới harness ta!) — xem §3 |
| MSD (C&H retrain) | 43.9 ± 0.8 | [Croce2022-EAT] Table 5 | |
| E-AT | 42.4 ± 0.6 | [Croce2022-EAT] Table 5 | claim "up to 3× cheaper" |
| SAT | 40.4 · AVG 40.1 | [Croce2022-EAT] Table 5 | |
| *(ours, ε∞=0.03, in-house tier)* | *max_inhouse 43.9 · T025 43.0±0.05 (3 seeds) · 3a 42.3* | bảng ta | *tier riêng — protocol caveat, chỉ định vị* |

**CLAIM LADDER (REVISED xuống trung thực, Kiet lock 2026-07-04 — "match/beat MSD" BỎ HOÀN TOÀN):**
- **C1 (main, unique):** mechanism account F4–F6 — vì sao loss-signal DRO thoái hoá ≤AVG, weak-probe đảo ranking norm, binding norm di cư và tracking>precision. Không prior work nào có chain này.
- **+F7 (eval corrections, service to community):** locuslab MAX ckpt 19pp artifact · published ℓ1 overstated (F1) · probe-bias structure (F5).
- **C2 (positioning, khiêm tốn):** simple cheap adaptive weighting (~2–3× rẻ hơn MSD/iteration, đếm từ code) **đạt tới cụm SOTA 43.9–44.6, KHÔNG vượt nó** (max_inhouse 43.9, T025 43.0±0.05 @0.03 vs cited cluster @8/255).
- MSD giữ label **"classical strong baseline"**; mọi claim "match/beat SOTA" vẫn CẤM.

**Protocol-ceiling evidence (2026-07-03, cập nhật 07-04 với ±std):** fine-tune Table 1: union MSD **42.6±0.2** · E-AT 42.2±0.8 · MAX 42.2±0.6 (‡, 8/255, RN-18) — cộng from-scratch Table 5 (42.4±0.6 – 44.0±0.7) và các run của ta (42.3–43.9 @0.03): **mọi method nghiêm túc trên RN-18 hội tụ về dải 42–44.6** bất kể cơ chế (per-step selection, geometric trick, binding-aware, per-sample soft/hard). Với std 5-seed giờ có trên bảng: **RAMP 44.6±0.6 CHỒNG LẤN MAX 44.0±0.7 / MSD 43.9±0.8 trong ~1σ — không còn đọc là outlier tách biệt mà là đỉnh của một CỤM 43.9–44.6**. Ý nghĩa cho paper: giá trị nằm ở CƠ CHẾ (C1) + GIÁ (~2–3× rẻ/iter) + việc một weighting ĐƠN GIẢN chạm cụm này (C2) — không phải ở leaderboard delta.

## 2 · Differentiation vs từng prior
- **vs E-AT** (cùng chung claim "rẻ hơn MSD"): E-AT = **fixed geometric trick** (chỉ train 2 norm cực biên ℓ∞+ℓ1, ℓ2 miễn phí qua convex-hull theorem). Ta = **adaptive weighting** (binding-aware signal / per-sample soft) + **mechanism account** (F4–F6: vì sao loss-signal degenerate về AVG, binding norm di cư, tracking>precision). E-AT không giải thích *cơ chế*; ta có chain finding. Số của ta (43.0 @0.03, 1 seed) vs E-AT 42.4 @8/255 — không so trực tiếp được, gần nhau.
- **vs RAMP**: khác trục — RAMP đổi LOSS (logit pairing + NT gradient projection), ta đổi WEIGHTING/AGGREGATION trên attack chuẩn. **RAMP's Eq. 2 chứa một L_max term (per-sample max over norms) → per-sample max là lựa chọn được community-endorse ở mức NeurIPS'24; trục T của ta chính là PHÂN TÍCH + LÀM MỀM term đó (T→0 = hard L_max, T ấm = AVG-blend) — complementary, không competing (related-work line cho paper).** RAMP không có mechanism story về binding-norm migration. Có thể compose (RAMP loss + binding-aware weighting) — future work, không claim. **[Upgrade 2026-07-04]: composability line giờ có backing THỰC — thesis-era self-implementation của RAMP components (`ramp_full_10_b5_gp`, archive §2.4) đã chạy được cùng codebase family; và "consistency × binding-aware" có preliminary orthogonal-gain evidence (+0.5–1.6 AA-ℓ∞ trên trục độc lập).**
- **vs Xiao2022-ASW**: đã có positioning note trong CARD-3b (per-TYPE smoothness vs per-SAMPLE loss).
- **vs MSD**: per-STEP selection bên trong PGD; ta per-sample sau attack, ~2–3× rẻ hơn/iteration (đếm code).

## 2.4 · THREE-AXIS MAP (thesis-era consistency archive `ardg_consist`, mapped 2026-07-04 — PARKED, no GPU pre-deadline)
Ba trục can thiệp độc lập đã có data nội bộ (⚠ archive dùng OLD protocol: ℓ∞+ℓ2 only, avg-case, RN-18+normalize — **KHÔNG so được với bảng hiện tại**, chỉ dùng làm orthogonality evidence):
1. **Weighting/aggregation** (P2 hiện tại): loss→binding-aware +1.8–3.3pp worst-∪ (bảng chính).
2. **Consistency regularization** (thesis-era, feat-L2 penultimate): **+0.5–0.8 AA-ℓ∞ C10 / +1.6 C100** so với dual-attack β=0 control — gain trên trục KHÁC weighting → preliminary evidence cho "consistency × binding-aware" orthogonal-gain (future work, KHÔNG chạy trước deadline).
3. **Loss-shaping** (thesis-era self-implementation của RAMP components — `ramp_full_10_b5_gp` = ramp_kl_correct + GP): **AA-ℓ∞ 47.27 NHƯNG clean −6.4, mean-rob −3** → strong ℓ∞-shift, union-relevance UNCLEAR. Backing cho future-work line "composable with RAMP" bằng chính implementation của ta, không chỉ trích dẫn.

## 2.7 · FAIR-COMPARISON BASELINE INVENTORY (Part C, 2026-07-04) — code/ckpt provenance per method
Parity invariant: MỌI baseline tích hợp chỉ được eval QUA `eval_union`/standard pack của ta.

| method | (a) re-eval ckpt | (b) retrain in-house khả thi? | (c) ‡cite | trạng thái |
|---|---|---|---|---|
| **MSD** | ✅ locuslab `robust_union/CIFAR10/Selected/MSD.pt` (tier-2: 42.5) | ✅ CARD-7 msd_inhouse (msd_v0 faithful, OPTIONAL) | ✅ C&H Tab5 43.9±0.8 | done + optional in-house |
| **AVG** | ✅ locuslab AVG.pt (tier-2: 38.7) | ✅ avg_frozen ×3 done (40.6±0.4) | ✅ C&H 40.1 | **F7 pair đủ** |
| **MAX** | ✅ locuslab MAX.pt (tier-2: 25.0 — artifact) | ✅ max_inhouse done (43.9) | ✅ C&H Tab5 44.0±0.7 | **F7 3-evidence đủ** |
| **E-AT** | ❌ KHÔNG có final ckpt công khai (chỉ `RAMP/models/pretr_{Linf,L1,L2}.pth` = điểm khởi đầu fine-tune) | ✅ **KHẢ THI** — `external/RAMP/eat_train.py` chạy được, pretr ckpts sẵn, robustbench import OK; fine-tune 3-epoch = RẺ | ✅ C&H Tab5 42.4±0.6 | → **CARD-9 gated** (card-first) |
| **RAMP** | ❌ no public ckpt | ✅ **đang train** (CARD-8, their code) | ✅ 44.6±0.6 | repro running |
| **SAT** | ❌ | code KHÔNG bundle (ngoài scope) | ✅ C&H 40.4 | cite-only |

**Đọc:** ba baseline "mạnh" (MSD/AVG/MAX) đã có ĐỦ (re-eval + in-house). E-AT là mắt xích còn thiếu để bảng in-house đối xứng — và nó RẺ (fine-tune 3 epoch) + là differentiation trực tiếp nhất của ta (E-AT = fixed geometric prescription vs ta = adaptive). → đáng làm 1 hàng in-house E-AT (CARD-9), gated sau mandatory queue, card-first + pre-registration.

## 2.5 · CẤU TRÚC BẢNG CUỐI CHO PAPER (Kiet lock 2026-07-03): 3 tier theo mức kiểm soát
1. **Tier 1 — in-house, recipe-controlled** (mạnh nhất): mọi method train bằng CHÍNH recipe/attacks/protocol của ta — avg_frozen (AVG in-house), MAX in-house, 3a, 3b-T*, P1. So sánh trong tier này là claim chính.
2. **Tier 2 — checkpoint re-evals**: ckpt official re-eval dưới harness ta (locuslab MSD/AVG/MAX/LINF/L2/L1). Ký hiệu *.
3. **Tier 3 — ‡ cited**: số báo cáo từ paper gốc, protocol có thể khác (RAMP 44.6, E-AT Table 1/5) — chỉ để định vị, không so trực tiếp.
**Motivating example cho cấu trúc này: MAX 19pp gap** (locuslab ckpt 25.0 tier-2 vs C&H retrain 44.0 tier-3) — cùng "một method" lệch 19pp qua recipe/nguồn ⇒ trộn tier là vô nghĩa.
**RAMP: quyết định P3-đóng bị ĐẢO (Kiet 2026-07-04, sau khi đọc full paper): protocol-verify seed-0 ĐANG CHẠY** — official code (upstream be4971f, patch duy nhất `import copy` thiếu trong `gp()`, zero algorithmic change), official cmd 80ep `--kl --max --gp --lbd 5`, recipe HỌ nguyên vẹn. Eval kép đã armed (scripts/ramp/): apgd + standard AA, mỗi cái ở CẢ 0.03 (protocol ta → so tier được) VÀ 8/255 (protocol họ → so 44.6±0.6 reported). Đây KHÔNG phải retrain-baseline (C&H vẫn ‡ cited) — đây là verify outlier duy nhất không có public ckpt. Note: RAMP row từ run này = "our reproduction of RAMP (1 seed)", tier riêng, không thay số ‡ của họ.

## 2.6 · T-AXIS REFRAME (Kiet lock 2026-07-03, sau sweep 4/4 điểm)
Sweep đầy đủ (1 seed): T=0.25 **43.0** · 0.5 **42.6** · 1 **41.5** · 2 **42.1** — KHÔNG monotone; chỉ 0.25↔1 tách ~2σ.
- **Frame đúng: T là cái NÚM trên trade ℓ∞ ↔ (ℓ1, ℓ2).** T ấm → ℓ1/ℓ2/clean tăng (T=1: ℓ1 **51.5 = kỷ lục ℓ1**, ℓ2 65.3), ℓ∞ giảm → union bị **ℓ∞-limited** ở warm T. Nhất quán F2 (model mạnh nào cũng ℓ∞-bound) + F6 (binding di cư về ℓ∞ khi ℓ1 được vá).
- **"Cold beats hard" CHẾT (CARD-5 landed 2026-07-04): max_inhouse 43.9 — NGOÀI band pre-registered 41.5–43.5 phía CAO.** Hard-MAX in-house là số tốt nhất của ta; trục T đọc gần-monotone lạnh→ấm: 43.9 (T→0) → 43.0 → 42.6 → 41.5 → 42.1. T vẫn là cái núm trade hợp lệ (cho clean/ℓ1/ℓ2), nhưng cho worst-∪ thì lạnh hơn = tốt hơn.
- ~~ℓ1 record 51.5~~ **[RÚT 2026-07-04]:** avg_frozen in-house (uniform) cũng đạt ℓ1≈50 (files on disk) → ℓ1≈50 không phải đặc sản của weighting; KHÔNG dùng "ℓ1 record" framing. Trục trade đúng (đã reframe): union/ℓ∞ vs clean/ℓ1/ℓ2.

## 3 · ⚠ MAX-RECIPE CONFOUND (phát hiện 2026-07-03, quan trọng cho narrative)
Bảng ta: locuslab MAX ckpt = **25.0** union (harness ta, 0.03). E-AT Table 5: **MAX tự train lại = 44.0** (@8/255, harness họ). Chênh ~19pp không thể do protocol/harness — **checkpoint MAX của locuslab yếu bất thường so với MAX được train tử tế**.
- Hệ quả 1: **anchor "hard-MAX fails (25.0)" trong fig_union_vs_T và constraint CARD-3b phải được nói lại**: "the locuslab MAX *checkpoint*" chứ KHÔNG phải "hard-max *về nguyên tắc*". C&H's MAX 44.0 cho thấy hard per-sample max có thể rất mạnh với recipe đúng.
- Hệ quả 2 [RESOLVED 2026-07-04]: MAX in-house ĐÃ CHẠY (CARD-5) = **43.9 @0.03 — khớp C&H retrain 44.0±0.7 ‡ @8/255 trong 1σ**. Đây là **cross-validation ĐỘC LẬP của recipe in-house ta bằng số external**: hard per-sample max GENUINELY ~44 với recipe đúng (prediction miss 41.5–43.5 của strategy side được giải thích — không phải eval error, mà hard-MAX thật sự mạnh vậy). "Soft>hard" chết; xem §2.6.
- Hệ quả 3: mọi chỗ đã viết "MAX=25.0 proves hard-max fails" (CARD-3b constraint, QA) cần sửa thành phiên bản có điều kiện. [ĐÃ sửa các doc chính 07-03→07-04.]
