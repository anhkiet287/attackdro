# SOURCES.md — nguồn cho mọi claim trong project
*Quy tắc trích nguồn: mỗi số liệu/claim trong docs phải trỏ về (a) nguồn NỘI BỘ (kết quả tự chạy, ghi trong PROGRESS.md) hoặc (b) nguồn NGOÀI (paper/repo/trang chính thức). Ký hiệu độ tin: ✅ = đã verify link/venue · ⚠️ = cần double-check trước khi cite trong paper.*
*Last updated: 2026-07-02*

---

## 1 · NGUỒN NỘI BỘ (số của ta)
| Dữ liệu | Nguồn | Ghi chú |
|---|---|---|
| Bảng baseline 6 model (MSD 42.5, AVG 38.7, …) | `results/baseline_table.{md,json}` + PROGRESS.md 2026-07-01 | n=1000, APGD, ε∞=0.03; re-eval checkpoint official bằng harness của ta |
| Checkpoint baselines | ✅ github.com/locuslab/robust_union (Google Drive links trong repo) | 6 ckpt CIFAR-10: MSD/LINF/L2/L1/AVG/MAX |
| Số "reported" để so (MSD union 46.1, LINF ℓ1 16.0, MAX ℓ1 39.4…) | ⚠️ Bảng trong paper MSD [Maini2020] / repo robust_union | Claude Code đối chiếu khi validate — **check lại đúng bảng+trang trước khi đưa vào paper** |
| PGD-AT của ta (union 9.4) | PROGRESS.md 2026-07-02, `results/` | Train 8/255, eval 0.03 — pipeline proof, không vào bảng |
| F3: q ≈ (.82, .00, .18) | PROGRESS.md 2026-07-02 | Probe in-training, provisional |
| Kết quả thesis (avg +0.279pp, p=.0136) | Đồ án tốt nghiệp (file PDF trong project) | 20 seed, Wilcoxon, BH-FDR |
| Diagnostic geometry (corr ℓ1–saliency +.86…) | Output `perturbation_geometry.py` 2026-07-01 | ⚠️ Chạy trên ckpt load lệch 3 key — re-run trước khi cite |
| Thesis-era consistency archive | Drive folder `ardg_consist` (shared): `final_eval.json` mỗi run; **`ramp_full_09_b1_gp` INCOMPLETE — chỉ có config** | ⚠️ OLD protocol (ℓ∞+ℓ2, avg-case, normalize) — chỉ dùng làm orthogonality evidence (POSITIONING §2.4). W&B: `ardg-consist-c100` / `ardg-consist-c10`. Status: explored, mapped, PARKED (no GPU pre-deadline) |

## 2 · BIBLIOGRAPHY (theo tier của reading list)

### Nền tảng AT & eval
- **[Madry2018]** ✅ Madry, Makelov, Schmidt, Tsipras, Vladu. *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR 2018. arXiv:1706.06083
- **[Zhang2019-TRADES]** ✅ Zhang et al. *Theoretically Principled Trade-off between Robustness and Accuracy.* ICML 2019. arXiv:1901.08573
- **[Croce2020-AA]** ✅ Croce, Hein. *Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks* (AutoAttack). ICML 2020. arXiv:2003.01690
- **[Ilyas2019]** ✅ Ilyas et al. *Adversarial Examples Are Not Bugs, They Are Features.* NeurIPS 2019. arXiv:1905.02175

### Union / multi-norm (lõi của project)
- **[Tramer2019]** ✅ Tramèr, Boneh. *Adversarial Training and Robustness for Multiple Perturbations* (AVG/MAX). NeurIPS 2019. arXiv:1904.13000
- **[Maini2020-MSD]** ✅ Maini, Wong, Kolter. *Adversarial Robustness Against the Union of Multiple Perturbation Models.* ICML 2020. arXiv:1909.04068 · code: github.com/locuslab/robust_union
- **[Croce2021-L1APGD]** ✅ Croce, Hein. *Mind the Box: l1-APGD for Sparse Adversarial Attacks on Image Classifiers.* ICML 2021 (PMLR v139, 2201–2211). arXiv:2103.01208 — **nguồn chính cho F1**: paper này chỉ ra prior work overestimate ℓ1-robustness; ℓ1-AutoAttack = APGD-ℓ1 + FAB-ℓ1 + Square-ℓ1
- **[Croce2022-EAT]** ✅ Croce, Hein. *Adversarial Robustness against Multiple and Single lp-Threat Models via Quick Fine-Tuning of Robust Classifiers* (E-AT). ICML 2022. arXiv:2105.12508 · code: github.com/fra31/robust-finetuning *(verified 2026-07-02)*
  - **Table 5 (from-scratch PreActRN-18, ε=(8/255, 0.5, 12), n=1000 ‡):** l∞-AT 6.3 · l2-AT 20.9 · l1-AT 22.1 · SAT 40.4 · AVG 40.1 · **MAX 44.0** · **MSD 43.9** · **E-AT 42.4** (union). *(extracted verbatim from PDF 2026-07-03 — LƯU Ý: ε∞=8/255 ≠ protocol ta 0.03, và MAX/MSD ở đây là HỌ TỰ TRAIN với recipe của họ, khác locuslab ckpt.)*
  - Cost claim của họ: E-AT "costs up to three times less" than other multi-norm AT (fixed geometric trick: chỉ train 2 norm cực biên l∞+l1).
  - **Table 1 (fine-tune RN-18, 3 epochs, ε∞=8/255, 5 seeds — SỐ SẼ CITE ‡):** union **MSD 42.6±0.2 · E-AT 42.2±0.8 · MAX 42.2±0.6**; time/epoch **MSD 306s vs E-AT 160s**. **Caveats khi cite:** (a) ε∞=8/255 ≠ protocol ta 0.03 — không so trực tiếp; (b) đây là FINE-TUNE track, ≠ from-scratch (Table 5); (c) ±std của họ từ 5 seeds — chuẩn so sánh noise tốt.
  - **Thm 3.1 + Fig 2 (geometry):** ℓ2-ball ⊂ convex hull(ℓ1-ball ∪ ℓ∞-ball) tại radii chuẩn — nền lý thuyết cho F3 (ℓ2 free); xem MEMORY §5 F3 + POSITIONING §2.
- **[Jiang2024-RAMP]** ✅ Jiang, Singh. *RAMP: Boosting Adversarial Robustness Against Multiple lp Perturbations for Universal Robustness.* NeurIPS 2024. arXiv:2402.06827 · code: github.com/uiuc-focal-lab/RAMP *(verified 2026-07-03)*
  - **From-scratch RN-18 CIFAR-10: union 44.6, clean 81.2** (‡ reported, AutoAttack, ε=(8/255, 0.5, 12)). Fine-tuning: tới 53.3 (track riêng, không so với from-scratch).
  - **KHÔNG có checkpoint công khai** (repo chỉ có training scripts) → Level-2 re-eval dưới harness ta CHƯA khả thi; muốn so trực tiếp phải tự train bằng code họ.
  - Method: logit pairing loss (distribution-shift lens) + NT→AT gradient projection.
- **[NCAT2022]** ✅ Sriramanan, Gor, Feizi. *Toward Efficient Robust Training against Union of ℓp Threat Models.* NeurIPS 2022. OpenReview: 6qdUJblMHqy · code: github.com/GaurangSriramanan/NCAT *(verified 2026-07-02; không có arXiv riêng — cite proceedings)*
- **[PROTECTOR2022]** ✅ Maini, Chen, Li, Poor. *Perturbation Type Categorization for Multiple Adversarial Perturbation Robustness.* UAI 2022, PMLR v180. proceedings.mlr.press/v180/maini22a *(verified 2026-07-02 — check author list lần cuối khi cite: Maini là chắc, đồng tác giả lấy từ trang PMLR)*
- **[Xiao2022-ASW]** ✅ Xiao, Qin, Fan, Wu, Wang, Luo. *Adaptive Smoothness-weighted Adversarial Training for Multiple Perturbations with Its Stability Analysis.* arXiv:2210.00557 *(added 2026-07-02 — closest prior cho CARD-3b: per-TYPE smoothness-weights; positioning note trong EXPERIMENT_CARDS_P2 §CARD-3b; venue chưa rõ, check trước khi cite)*
- **[CURE2024]** ✅ Jiang, Cheung, Singh. *Towards **Generalized** Certified Robustness with Multi-Norm Training* (CURE). arXiv:2410.03000, 2024. OpenReview: fXb7MgySp8 · code: github.com/uiuc-focal-lab/CURE *(verified 2026-07-02 — LƯU Ý: title arXiv đã đổi "Universal"→"Generalized"; đây là CERTIFIED robustness, khác trục empirical của ta)*

### DRO / worst-group (nền method)
- **[Sagawa2020-GroupDRO]** ✅ Sagawa, Koh, Hashimoto, Liang. *Distributionally Robust Neural Networks for Group Shifts.* ICLR 2020. arXiv:1911.08731
- **[Sohoni2020-GEORGE]** ✅ Sohoni et al. *No Subclass Left Behind: Fine-Grained Robustness in Coarse-Grained Classification Problems.* NeurIPS 2020. arXiv:2011.12945 — tổ tiên trực tiếp của AttackDRO++
- **[Kirichenko2023-DFR]** ✅ Kirichenko, Izmailov, Wilson. *Last Layer Re-training is Sufficient for Robustness to Spurious Correlations.* ICLR 2023. arXiv:2204.02937
- **[Levy2020-CVaR]** ✅ Levy, Carmon, Duchi, Sidford. *Large-Scale Methods for Distributionally Robust Optimization.* NeurIPS 2020. arXiv:2010.05893 — nền cho fix CVaR (P2)
- **[Duchi2021]** ✅ Duchi, Namkoong. *Learning Models with Uniform Performance via Distributionally Robust Optimization.* Annals of Statistics 49(3):1378–1406, 2021. arXiv:1810.08750 *(verified 2026-07-02)*

### Robust foundation models (stretch P4)
- **[Mao2023-TeCoA]** ✅ Mao et al. *Understanding Zero-Shot Adversarial Robustness for Large-Scale Models.* ICLR 2023. arXiv:2212.07016 *(verified 2026-07-02; "TeCoA" = text-guided contrastive adversarial training loss trong paper)*
- **[Schlarmann2024-FARE]** ✅ Schlarmann, Singh, Croce, Hein. *Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings.* ICML 2024. arXiv:2402.12336

### Ý tưởng đã refute (cite trong Related Work nếu cần)
- **[Athalye2018]** ✅ Athalye, Carlini, Wagner. *Obfuscated Gradients Give a False Sense of Security.* ICML 2018. arXiv:1802.00420 — nền cho meta-lesson "kiến trúc inference sập dưới adaptive attack"
- **[Vaishnavi2019]** ⚠️ Vaishnavi et al. — foreground-attention mask AT; ⚠️ **[WangHorne2020]** robust features ≠ spatial invariance; ⚠️ **[AIB2024]**, **[DWF2025]** (arXiv:2512.20821), **[ADVMoE]**, **[SoE]**, **[Immunity2024]** (arXiv:2402.18787) — MoE-for-robustness. Verify từng cái nếu đưa vào Related Work.

### Deadline / hành chính
- **[NeurIPS2026-CFW]** ✅ neurips.cc/Conferences/2026/CallForWorkshops — workshop contributions suggested 29/08/2026, notification bắt buộc trước 29/09/2026; hội nghị 06–12/12/2026 (Sydney). *Từng workshop có CFP riêng — check sau khi công bố danh sách (11/07).*
- **[PhD-deadlines]** ⚠️ Mốc 01/12 & 15/12 là quy ước phổ biến — **check từng trường khi chốt list (P5)**.

## 3 · CLAIM → NGUỒN (map cho các docs hiện có)
| Claim (trong MEMORY/proposal/roadmap) | Nguồn |
|---|---|
| "Worst-case union là headline metric của subfield" | [Tramer2019], [Maini2020-MSD] |
| Định nghĩa union = AND per-sample; union ≤ min per-norm | [Maini2020-MSD] §threat model; suy diễn De Morgan (tự chứng minh, không cần cite) |
| ε chuẩn CIFAR-10: (0.03, 0.5, 12) | [Maini2020-MSD] + repo robust_union (protocol của checkpoint ta dùng) |
| "AutoAttack mạnh hơn PGD tự report; defense hay bị overstate" | [Croce2020-AA] |
| **F1: published ℓ1 overstated** | Số nội bộ (§1) + [Croce2021-L1APGD] (paper độc lập cùng kết luận) |
| **F2: MSD/AVG bị chặn bởi ℓ∞** | Số nội bộ (§1) — finding CỦA TA, chưa thấy nguồn ngoài; cite bảng của ta |
| "GroupDRO upweight nhóm loss cao nhất" | [Sagawa2020-GroupDRO] |
| "AttackDRO++ ~ GEORGE áp cho attacks" | [Sohoni2020-GEORGE] + đồ án |
| "MSD ~50 attack steps/iteration (đắt)" | ⚠️ [Maini2020-MSD] — check con số chính xác trong paper trước khi viết |
| "Robustness từ objective, không từ mẹo inference" (meta-lesson) | [Athalye2018] |
| "Non-robust features predictive nhưng giòn" | [Ilyas2019] |
| Mốc NeurIPS workshop 29/08, notify 29/09 | [NeurIPS2026-CFW] |

## 4 · CHECKLIST TRƯỚC KHI PAPER (P3)
- [ ] Verify mọi mục ⚠️ (ID arXiv, venue, tác giả) — đặc biệt NCAT, PROTECTOR, E-AT, Duchi, TeCoA
- [ ] Đối chiếu số "reported" với đúng bảng+trang trong [Maini2020-MSD]
- [ ] Re-run diagnostic geometry trên ckpt load sạch trước khi cite corr numbers
- [ ] Cite commit hash của robust_union khi nói "official checkpoints"
- [ ] Bảng cuối: ký hiệu nguồn từng số (†self-trained / *re-eval ckpt / ‡cited)

