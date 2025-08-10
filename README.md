# UPCA — Unified Predictive Cognitive Architecture

> A minimal, testable blueprint for cognition that puts **free energy**, **precision (Γ)**, and an **ethical prior (η)** in the same control loop.  
> This repo includes a runnable toy sim that shows the predicted signatures for **laughter** \(L(t)\), **qualia intensity** \(Q(t)\), and **η-driven policy shifts**.

---

## TL;DR

- **Closed-form signals** with falsifiable predictions  
  - **Laughter**: \( L(t)=\sigma\big(\alpha\,\Delta\dot F_\text{social}(t) + \beta\,\Delta\ddot F_\text{semantic}(t) - \gamma\,\Gamma_\text{threat}(t)\big) \)  
  - **Qualia intensity**: \( Q(t)=\sum_i \Gamma_i(t)\,\big|\tfrac{dF_i}{dt}\big| + \lambda\,H\!\left[q(s_i\mid o)\right] \)  
  - **Ethical drive**: \( \varepsilon_\eta = D_{KL}\!\big(q(y\mid\pi)\,\|\,p(y\mid C,\eta)\big) \) that **actually reshapes policy**.
- **One control architecture** (UPCA) that couples:
  - Fast sensory free energy \(F_\text{fast}\)
  - Slow abstraction/MDL \(F_\text{slow}\)
  - Ethical error \(\varepsilon_\eta\)
- **Runnable now**: a small simulation showing (i) laughter dips when threat precision spikes, (ii) \(Q(t)\) tracks precision-weighted error dynamics, (iii) policies flip when \(\eta\) flips.

---

## What’s in this repo

- `upca_min.py` — **Minimal simulator** of the UPCA control loop. Prints a table each step:
  - `F` (free energy proxy), `L` (laughter trigger), `Q` (qualia scalar), actions `a0/a1`, and `eps_eta`.
  - You’ll see **L → 0** when `Γ_threat` spikes, and **policy shift** after an `η` flip (demo line in code).
- `scaffold.py` — A graph learner baseline. Useful as an external **memory scaffold**, but **not the UPCA loop**. Keep it around for storage/pruning, not for FE control.
- `README.md` — this file.

> If you only do one thing: run `python upca_min.py` and skim the console log. You should see the qualitative signatures described above.

---

## Quickstart

```bash
# Python 3.10+ recommended
pip install -r requirements.txt  # if provided, else stdlib-only is fine
python upca_min.py
