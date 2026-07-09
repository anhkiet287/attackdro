# AttackDRO

Difficulty-aware adversarial training for CIFAR-10 union robustness over linf, l2, and l1.

Active research state lives in [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md). The human dashboard is [`docs/dashboard.html`](docs/dashboard.html), generated from `PROJECT_STATE.md` plus result JSONs.

Current paper protocol: CIFAR-10, PreActResNet-18, raw `[0,1]` pixels, eps `(8/255, 0.5, 12)`, RAMP schedule, APGD train `10/10/10`, develop eval `20/20/100`, final eval full AutoAttack.

Active work is on `card-pb-fixes` and cleanup branches derived from it; old docs/configs are archived under `docs/legacy/` or `configs/legacy/` and should not be treated as current protocol.
