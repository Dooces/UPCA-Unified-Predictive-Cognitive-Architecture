# UPCA v4.1 — Unified Predictive Cognitive Architecture

> Closed-form, falsifiable signals for laughter `L(t)`, qualia intensity `Q(t)`, and grounded meaning via fast/slow free energy + ethical prior `η`, integrated in a single control loop.

[**Whitepaper (PDF)**](UPCA_v4.1_whitepaper.pdf) • [**Minimal simulator**](upca_min.py)

**DOI:** _pending_ • **License:** MIT

## Cite
Brent, B. (2025). UPCA v4.1: A Unified Predictive Cognitive Architecture (whitepaper & code). _Zenodo_. DOI: _pending_

## Summary
- **Laughter**: \(L(t)=\sigma(\alpha\,\Delta\dot F_{\text{social}}+\beta\,\Delta\ddot F_{\text{semantic}}-\gamma\,\Gamma_{\text{threat}})\)
- **Qualia**: \(Q(t)=\sum_i \Gamma_i(t)\,|dF_i/dt|+\lambda\,H[q(s_i|o)]\)
- **Grounding**: minimize \(F_{\text{fast}}\) + \(\epsilon F_{\text{slow}}\) + \(\gamma \epsilon_\eta\); ablations predict measurable failures.

## Run
```bash
python upca_min.py
