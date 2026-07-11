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
- **[Croce2022-EAT]** ⚠️ Croce, Hein. *Adversarial Robustness against Multiple and Single lp-Threat Models via Quick Fine-Tuning of Robust Classifiers* (E-AT). ICML 2022. arXiv:2105.12508 (check ID)
- **[NCAT2022]** ⚠️ *Toward Efficient Robust Training against Union of lp Threat Models.* NeurIPS 2022 — **verify tên tác giả + link OpenReview trước khi cite**
- **[PROTECTOR2022]** ⚠️ Maini et al. *Perturbation Type Categorization for Multiple Adversarial Perturbation Robustness.* UAI 2022 — verify venue/ID
- **[CURE2024]** ⚠️ *Towards Universal Certified Robustness with Multi-Norm Training.* 2024 — verify venue/ID

### DRO / worst-group (nền method)
- **[Sagawa2020-GroupDRO]** ✅ Sagawa, Koh, Hashimoto, Liang. *Distributionally Robust Neural Networks for Group Shifts.* ICLR 2020. arXiv:1911.08731
- **[Sohoni2020-GEORGE]** ✅ Sohoni et al. *No Subclass Left Behind: Fine-Grained Robustness in Coarse-Grained Classification Problems.* NeurIPS 2020. arXiv:2011.12945 — tổ tiên trực tiếp của AttackDRO++
- **[Kirichenko2023-DFR]** ✅ Kirichenko, Izmailov, Wilson. *Last Layer Re-training is Sufficient for Robustness to Spurious Correlations.* ICLR 2023. arXiv:2204.02937
- **[Levy2020-CVaR]** ✅ Levy, Carmon, Duchi, Sidford. *Large-Scale Methods for Distributionally Robust Optimization.* NeurIPS 2020. arXiv:2010.05893 — nền cho fix CVaR (P2)
- **[Duchi2021]** ⚠️ Duchi, Namkoong. *Learning Models with Uniform Performance via Distributionally Robust Optimization.* Annals of Statistics 2021. arXiv:1810.08750 (check ID)

### Robust foundation models (stretch P4)
- **[Mao2023-TeCoA]** ⚠️ Mao et al. *Understanding Zero-Shot Adversarial Robustness for Large-Scale Models.* ICLR 2023. arXiv:2212.07016 (check ID)
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

