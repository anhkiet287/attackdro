# Results manifest

Index only. Result directories are immutable provenance and are not modified by this manifest. `safe_to_archive` means eligible for a later explicitly approved archival review; it does not authorize movement or deletion.

| run | canonical? | retention_class | checkpoint_role | grade | safe_to_archive |
|---|---|---|---|---|---|
| analysis | no | derived-support | N/A | derived/read-only | TBD |
| archive | no | historical-archive | mixed | mixed historical | already-archived |
| audit | supporting | audit-active | mixed canonical checkpoints | multinorm audit v1 | no |
| b2_static_cycle_ramp80_apgd_8255_t49k_v1k | yes | canonical-active | val_best + last | APGD 20/20/100, val_select/test_monitor, n=1000 | no |
| b3_static_weighted_linf_ramp80_apgd_8255_t49k_v1k | yes | canonical-active | val_best + last | APGD 20/20/100, val_select/test_monitor, n=1000 | no |
| b4_predictive_refresh_routeA_diag_ramp80_apgd_8255_t49k_v1k | yes | canonical-active | val_best + last | APGD 20/20/100, val_select/test_monitor, n=1000 | no |
| branch_diag_b4_small_v1 | yes | canonical-active-diagnostic | branch endpoints; no retained checkpoint role | APGD 20/20/100, val_select, n=1000 | no |
| diagnostics | no | diagnostic-provenance | mixed | diagnostic-only | review |
| eval_curve | no | derived-support | epoch checkpoints | APGD decision-curve | review |
| exploration | no | tangential/archive-candidate | mixed | exploratory | review |
| idea1_failrate_ramp80_apgd_8255 | no | tangential/archive-candidate | legacy best/last mapping | APGD 20/20/100 develop eval where available | review |
| predictive_apgd_8255 | no | tangential/archive-candidate | TBD | exploratory APGD | review |
| predictive_ks24_apgd_8255 | no | tangential/archive-candidate | TBD | exploratory APGD | review |
| predictive_ks2_apgd_8255 | no | tangential/archive-candidate | TBD | exploratory APGD | review |
| predictive_ks4_apgd_8255 | no | tangential/archive-candidate | TBD | exploratory APGD | review |
| predictive_ks8_apgd_8255 | no | tangential/archive-candidate | TBD | exploratory APGD | review |
| ramp | supporting | external-reference-active | external/final or epoch-specific | mixed reproduce/decision grade | no |
| reactive_apgd_8255 | no | tangential/archive-candidate | TBD | exploratory APGD | review |
| reactive_softT_full10_ramp80_apgd_8255_t49k_v1k | yes | canonical-active | val_best + last | APGD 20/20/100, val_select/test_monitor, n=1000 | no |
| reports | no | derived-support | N/A | summaries only | TBD |

Tangential curriculum and recovery assets, where present beneath `exploration/`, inherit the `tangential/archive-candidate` classification. Unknown metadata remains `TBD` pending a separate provenance audit.
