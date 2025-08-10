
# upca_min.py
# Minimal, runnable UPCA core (single file)
# - Variational Free Energy (linear-Gaussian, closed-form MAP)
# - Precision gating Γ on observation error
# - Ethics prior η with KL penalty ε_η that changes action choice
# - Laughter trigger L(t) and Qualia scalar Q(t) per provided formulas
# Only dependency: numpy

import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class UPCAMin:
    def __init__(self, n_s=3, n_o=3, n_a=2, y_dim=2, seed=0):
        rng = np.random.RandomState(seed)
        # Generative model
        self.A = np.array([[1.0, 0.2, 0.0],
                           [0.0, 1.0, 0.3],
                           [0.0, 0.1, 1.0]])  # obs mapping (n_o x n_s)
        self.B = np.array([[0.95, 0.0, 0.0],
                           [0.0, 0.97, 0.0],
                           [0.0, 0.0, 0.98]])  # state transition (n_s x n_s)
        self.U = rng.randn(n_s, n_a) * 0.2       # action effect on state

        # Noise precisions (inverse covariances)
        self.Lambda_o = np.diag([10.0, 10.0, 10.0])  # base obs precision
        self.Lambda_s = np.diag([5.0, 5.0, 5.0])     # prior precision on dynamics residual

        # Precision gate Γ (per observation channel, multiplicative on Lambda_o)
        self.Gamma = np.ones(n_o)  # will change over time on a channel to demo gating

        # Ethics prior η over outcome y = W_y s'
        self.Wy = np.array([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0]])  # y_dim x n_s
        self.eta_mu = np.array([0.0, 0.0])          # preferred outcomes baseline
        self.Lambda_eta = np.diag([2.0, 2.0])       # ethics precision

        # Weights
        self.gamma_ethics = 1.5  # multiplies epsilon_eta in action objective

        # State
        self.s_prev = np.zeros(n_s)
        self.a_prev = np.zeros(n_a)
        self.t = 0

        # Monitors for L(t) and Q(t)
        self.F_social_hist = []     # use obs channels [0,1] as "social"
        self.F_semantic_hist = []   # dynamics residual energy
        self.L_hist = []
        self.Q_hist = []
        self.o_prev = None
        self.last_obs_term = None # per-channel F_i for Q(t)

        # Precompute for speed
        self.I_s = np.eye(n_s)

    def _posterior_precision(self):
        # Posterior precision: A^T Γ Λ_o A + Λ_s
        Gamma_mat = np.diag(self.Gamma)
        return self.A.T @ (Gamma_mat @ self.Lambda_o) @ self.A + self.Lambda_s

    def infer_state(self, o_t):
        # Compute MAP s_t that minimizes F_t (quadratic form)
        Gamma_mat = np.diag(self.Gamma)
        Lambda_post = self._posterior_precision()
        # RHS: A^T Γ Λ_o o + Λ_s (B s_prev + U a_prev)
        rhs = self.A.T @ (Gamma_mat @ self.Lambda_o @ o_t) + self.Lambda_s @ (self.B @ self.s_prev + self.U @ self.a_prev)
        # Solve linear system
        # Add small ridge for numerical stability
        ridge = 1e-6 * np.eye(Lambda_post.shape[0])
        s_map = np.linalg.solve(Lambda_post + ridge, rhs)
        return s_map

    def free_energy_terms(self, s_t, o_t):
        # Observation error term (per-channel)
        pred_o = self.A @ s_t
        err_o = o_t - pred_o
        # Per-channel contribution F_i = 0.5 * Gamma_i * (err_i^2 * lambda_o_i)
        per_chan = 0.5 * self.Gamma * np.diag(self.Lambda_o) * (err_o**2)
        F_obs = np.sum(per_chan)

        # Dynamics ("semantic") error
        dyn_res = s_t - (self.B @ self.s_prev + self.U @ self.a_prev)
        F_sem = 0.5 * dyn_res.T @ self.Lambda_s @ dyn_res
        return per_chan, F_obs, F_sem, err_o, dyn_res

    def epsilon_eta(self, s_next, context=None):
        # Outcome features
        y = self.Wy @ s_next
        diff = y - self.eta_mu  # assume q(y|π) ~ delta(y), KL reduces to quadratic form
        eps = 0.5 * diff.T @ self.Lambda_eta @ diff
        return eps, y

    def predict_next(self, s_t, a):
        s_next = self.B @ s_t + self.U @ a
        o_next = self.A @ s_next  # mean prediction
        return s_next, o_next

    def expected_F_fast(self, s_t, a):
        # Single-step proxy for expected free energy: F at predicted next step (mean field)
        s_next, o_next = self.predict_next(s_t, a)
        # Use same Γ, expected obs equals o_next; plug into F using that as "o"
        per_chan, F_obs, F_sem, _, _ = self.free_energy_terms(s_next, o_next)
        return F_obs + F_sem, per_chan

    def choose_action(self, s_t, action_set, context=None):
        # Evaluate surrogate objective: E[F_fast] + gamma * epsilon_eta
        best_val = +1e9
        best = action_set[0]
        best_breakdown = None
        for a in action_set:
            Fef, per_chan = self.expected_F_fast(s_t, a)
            s_next, _ = self.predict_next(s_t, a)
            eps_eta, y = self.epsilon_eta(s_next, context)
            val = Fef + self.gamma_ethics * eps_eta
            if val < best_val:
                best_val = val
                best = a
                best_breakdown = (Fef, eps_eta, y)
        return best, best_val, best_breakdown

    def L_Q(self, per_chan_now, F_sem_now):
        # For L(t):
        # F_social = sum of per-chan over "social" channels [0,1]
        F_social_now = float(np.sum(per_chan_now[:2]))
        self.F_social_hist.append(F_social_now)
        self.F_semantic_hist.append(float(F_sem_now))
        # Derivatives via finite difference
        def deriv(seq):
            if len(seq) < 2:
                return 0.0
            return seq[-1] - seq[-2]
        def second_deriv(seq):
            if len(seq) < 3:
                return 0.0
            return seq[-1] - 2*seq[-2] + seq[-3]
        dF_social = deriv(self.F_social_hist)
        ddF_sem = second_deriv(self.F_semantic_hist)

        # Threat precision gate Γ_threat := Γ of channel 0 for demo
        Gamma_threat = self.Gamma[0]
        # Coeffs (tunable)
        alpha, beta, gamma = 2.0, 4.0, 3.0
        L_t = sigmoid(alpha * dF_social + beta * ddF_sem - gamma * Gamma_threat)

        # For Q(t): sum_i Γ_i * |dF_i/dt| + λ * entropy(q(s|o))
        if self.last_obs_term is None:
            dF_i = np.zeros_like(per_chan_now)
        else:
            dF_i = per_chan_now - self.last_obs_term
        self.last_obs_term = per_chan_now.copy()
        Q_obs_part = float(np.sum(self.Gamma * np.abs(dF_i)))
        # Posterior covariance entropy approx: H ~ 0.5 * log |Σ| (ignore constants)
        # Σ = (A^T Γ Λ_o A + Λ_s)^{-1}
        Lambda_post = self._posterior_precision()
        sign, logdet = np.linalg.slogdet(Lambda_post)
        # Entropy of covariance inverse: log |Σ| = - log |Λ_post|
        H_q = -0.5 * logdet
        lam = 0.1
        Q_t = Q_obs_part + lam * H_q

        self.L_hist.append(float(L_t))
        self.Q_hist.append(float(Q_t))
        return float(L_t), float(Q_t)

    def step(self, o_t, action_set, context=None):
        # Inference
        s_t = self.infer_state(o_t)
        per_chan, F_obs, F_sem, err_o, dyn_res = self.free_energy_terms(s_t, o_t)
        L_t, Q_t = self.L_Q(per_chan, F_sem)

        # Choose action
        a_star, obj, breakdown = self.choose_action(s_t, action_set, context)
        (Fef_pred, eps_eta, y_pred) = breakdown

        # Update state for next time
        self.s_prev = s_t.copy()
        self.a_prev = a_star.copy()
        self.t += 1

        F_total = F_obs + F_sem
        return {
            "t": self.t,
            "s": s_t,
            "a": a_star,
            "F_total": float(F_total),
            "F_obs": float(F_obs),
            "F_sem": float(F_sem),
            "L": L_t,
            "Q": Q_t,
            "policy_val": float(obj),
            "eps_eta": float(eps_eta),
            "y_pred": y_pred,
        }


def simulate():
    rng = np.random.RandomState(1)
    agent = UPCAMin()

    # True environment dynamics (for generating o_t)
    B_true = np.array([[0.98, 0.0, 0.0],
                       [0.0, 0.99, 0.0],
                       [0.0, 0.0, 0.97]])
    U_true = agent.U.copy()
    s_true = np.array([0.0, 0.0, 0.0])

    # Candidate actions
    action_set = [
        np.array([ 0.5,  0.0]),
        np.array([-0.5,  0.0]),
        np.array([ 0.0,  0.5]),
        np.array([ 0.0, -0.5]),
        np.array([ 0.0,  0.0]),
    ]

    print("t |    F      L      Q   | a0   a1  | eps_eta  | note")
    print("-"*70)
    for t in range(1, 41):
        # environment step: create a gentle target drift to make decisions non-trivial
        drift = np.array([0.01, 0.0, -0.005])
        s_true = B_true @ s_true + U_true @ np.array([0.2, -0.1]) + drift + rng.randn(3)*0.01
        o_t = agent.A @ s_true + rng.randn(3)*0.05

        # Demo 1: Precision gate (increase threat precision at t=15)
        if t == 15:
            agent.Gamma[0] = 4.0  # raise precision on channel 0 (threat)
        if t == 25:
            agent.Gamma[0] = 1.0  # drop back

        # Demo 2: Ethics flip (at t=28 prefer negative y[0])
        note = ""
        if t == 28:
            agent.eta_mu = np.array([-1.0, 0.0])
            note = "<-- flip η to prefer y0 ~ -1"

        out = agent.step(o_t, action_set, context=None)
        print(f"{out['t']:2d} | {out['F_total']:6.3f}  {out['L']:6.3f}  {out['Q']:6.3f} | "
              f"{out['a'][0]:+4.1f} {out['a'][1]:+4.1f} | {out['eps_eta']:7.3f} | {note}")

    print("\nSummary:")
    print(f"  Final F = {out['F_total']:.3f}")
    print(f"  L(t) min/max = {np.min(agent.L_hist):.3f}/{np.max(agent.L_hist):.3f}")
    print(f"  Q(t) min/max = {np.min(agent.Q_hist):.3f}/{np.max(agent.Q_hist):.3f}")
    print("  (Observe: L dips when Γ_threat spikes; policy shifts after η flip.)")

if __name__ == '__main__':
    simulate()
