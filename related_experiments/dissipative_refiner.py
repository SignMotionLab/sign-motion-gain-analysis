"""
Dissipativity-Based Long-Horizon Stability Guarantees for Sign Language Motion Generation
Core Method Implementation: Dynamical Refiners, Lyapunov Stability Certificate & Kinematic Metrics
"""

import numpy as np
import scipy.linalg


class LyapunovStabilityCertifier:
    """
    Computes theoretical stability bounds and Lyapunov energy guarantees
    based on Theorem 1 in wending.tex:
        ||z(t)||_2^2 <= (p_+ / p_-) * exp(-t / (2 * p_+)) * ||z(0)||_2^2 + (4 * p_+ * ||PB||_2^2 / p_-) * delta^2
    """
    def __init__(self, omega=0.5, zeta=0.7, mass=1.0):
        self.omega = float(omega)
        self.zeta = float(zeta)
        self.mass = float(mass)
        self.k = self.mass * (self.omega ** 2)
        self.d = self.mass * (2.0 * self.zeta * self.omega)
        
        # Scalar 2D block state-space system for each 1D coordinate error z_i = [e_i, \dot{e}_i]^T
        # \dot{z} = A z + B d
        self.A_block = np.array([
            [0.0, 1.0],
            [-self.k / self.mass, -self.d / self.mass]
        ], dtype=np.float64)
        
        self.B_block = np.array([
            [0.0],
            [1.0 / self.mass]
        ], dtype=np.float64)
        
        # Solve continuous Lyapunov equation A^T P + P A = -Q with Q = I
        self.Q_block = np.eye(2, dtype=np.float64)
        self.P_block = scipy.linalg.solve_continuous_lyapunov(self.A_block.T, -self.Q_block)
        
        # Eigenvalues
        eigvals = np.linalg.eigvalsh(self.P_block)
        self.p_minus = float(np.min(eigvals))
        self.p_plus = float(np.max(eigvals))
        
        # Norm of P * B
        pb = self.P_block @ self.B_block
        self.norm_PB = float(np.linalg.norm(pb, ord=2))
        
        # Asymptotic steady-state error factor: 4 * p_+ * ||PB||^2 / p_-
        self.steady_state_bound_factor = (4.0 * self.p_plus * (self.norm_PB ** 2)) / self.p_minus

    def compute_theoretical_bound(self, t_array, z0_norm, delta):
        """
        Calculates theoretical upper bound trajectory for error state ||z(t)||_2^2.
        
        Parameters:
            t_array: array-like of time steps (seconds)
            z0_norm: scalar initial error norm ||z(0)||_2
            delta: maximum disturbance magnitude ||d(t)||_2 <= delta
        """
        t = np.asarray(t_array, dtype=np.float64)
        exp_decay = np.exp(-t / (2.0 * self.p_plus))
        init_term = (self.p_plus / self.p_minus) * exp_decay * (z0_norm ** 2)
        disturb_term = self.steady_state_bound_factor * (delta ** 2)
        return init_term + disturb_term

    def compute_lyapunov_energy(self, error_pos, error_vel):
        """
        Computes empirical quadratic Lyapunov energy V(z) = sum_j (z_j^T P_block z_j)
        for multidimensional tracking error across all frames.
        
        Parameters:
            error_pos: shape (T, D) position error
            error_vel: shape (T, D) velocity error
            
        Returns:
            V: shape (T,) Lyapunov energy at each frame
            z_norm_sq: shape (T,) squared state norm ||z(t)||_2^2
        """
        T, D = error_pos.shape
        V = np.zeros(T, dtype=np.float64)
        z_norm_sq = np.zeros(T, dtype=np.float64)
        
        for t in range(T):
            ep = error_pos[t]
            ev = error_vel[t]
            z_t = np.stack([ep, ev], axis=1) # (D, 2)
            # Quadratic form z_j P z_j
            v_t = np.sum((z_t @ self.P_block) * z_t)
            V[t] = v_t
            z_norm_sq[t] = np.sum(ep ** 2) + np.sum(ev ** 2)
            
        return V, z_norm_sq


class DampedSecondOrderRefiner:
    r"""
    Damped Second-Order Dynamic Refiner (Method 4: Proposed Dissipative Refiner)
    Solves continuous-time tracking dynamics:
        M \ddot{q} + D \dot{q} + K(q - q_r) = d(t)
    with substepped numerical integration.
    """
    def __init__(self, omega=0.5, zeta=0.7, dt=0.1, substeps=10):
        self.omega = float(omega)
        self.zeta = float(zeta)
        self.damp = float(2.0 * self.zeta * self.omega)
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.sub_dt = self.dt / self.substeps

    def refine(self, ref_traj, disturbance_delta=0.0, rng=None):
        """
        Tracks reference trajectory ref_traj with damped second order dynamics.
        
        Parameters:
            ref_traj: shape (T, D) semantic reference poses
            disturbance_delta: perturbation magnitude delta
            rng: np.random.Generator instance
            
        Returns:
            q_out: shape (T, D) refined pose trajectory
            v_out: shape (T, D) velocity trajectory
            disturbances: shape (T, D) applied disturbances
        """
        T, D = ref_traj.shape
        if T < 2:
            return ref_traj.copy(), np.zeros_like(ref_traj), np.zeros_like(ref_traj)

        q = ref_traj[0].astype(np.float64).copy()
        v = np.zeros_like(q)
        
        q_out = [q.copy()]
        v_out = [v.copy()]
        disturb_list = [np.zeros_like(q)]
        
        for t_idx in range(1, T):
            target = ref_traj[t_idx].astype(np.float64)
            # Sample bounded disturbance ||d|| <= delta
            if disturbance_delta > 0:
                if rng is None:
                    rng = np.random.default_rng(2026)
                raw_d = rng.standard_normal(D)
                norm_d = np.linalg.norm(raw_d) + 1e-12
                # Uniform random radius inside delta-ball
                r = rng.uniform(0, disturbance_delta)
                d_vec = (raw_d / norm_d) * r
            else:
                d_vec = np.zeros(D, dtype=np.float64)
                
            disturb_list.append(d_vec.copy())
            
            # Substepping numerical integration
            for _ in range(self.substeps):
                acc = -self.damp * v - (self.omega ** 2) * (q - target) + d_vec
                v += self.sub_dt * acc
                q += self.sub_dt * v
                
            q_out.append(q.copy())
            v_out.append(v.copy())
            
        return (
            np.asarray(q_out, dtype=np.float32),
            np.asarray(v_out, dtype=np.float32),
            np.asarray(disturb_list, dtype=np.float32)
        )


class UndampedODERefiner:
    """
    Undamped ODE Refiner (Method 3: Ablation with zeta = 0, D = 0)
    Generates persistent oscillatory tracking without dissipative stabilization.
    """
    def __init__(self, omega=0.5, dt=0.1, substeps=10):
        self.omega = float(omega)
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.sub_dt = self.dt / self.substeps

    def refine(self, ref_traj, disturbance_delta=0.0, rng=None):
        T, D = ref_traj.shape
        if T < 2:
            return ref_traj.copy(), np.zeros_like(ref_traj), np.zeros_like(ref_traj)

        q = ref_traj[0].astype(np.float64).copy()
        v = np.zeros_like(q)
        
        q_out = [q.copy()]
        v_out = [v.copy()]
        disturb_list = [np.zeros_like(q)]
        
        for t_idx in range(1, T):
            target = ref_traj[t_idx].astype(np.float64)
            if disturbance_delta > 0:
                if rng is None:
                    rng = np.random.default_rng(2026)
                raw_d = rng.standard_normal(D)
                norm_d = np.linalg.norm(raw_d) + 1e-12
                r = rng.uniform(0, disturbance_delta)
                d_vec = (raw_d / norm_d) * r
            else:
                d_vec = np.zeros(D, dtype=np.float64)
                
            disturb_list.append(d_vec.copy())
            
            for _ in range(self.substeps):
                acc = -(self.omega ** 2) * (q - target) + d_vec
                v += self.sub_dt * acc
                q += self.sub_dt * v
                
            q_out.append(q.copy())
            v_out.append(v.copy())
            
        return (
            np.asarray(q_out, dtype=np.float32),
            np.asarray(v_out, dtype=np.float32),
            np.asarray(disturb_list, dtype=np.float32)
        )


class FramewiseSmoothnessFilter:
    """
    Framewise Smoothness Filter (Method 2: Moving Average / Smoothing Penalty Baseline)
    """
    def __init__(self, window_size=5):
        self.window_size = int(window_size)

    def refine(self, ref_traj, disturbance_delta=0.0, rng=None):
        T, D = ref_traj.shape
        if disturbance_delta > 0:
            if rng is None:
                rng = np.random.default_rng(2026)
            noise = rng.uniform(-disturbance_delta, disturbance_delta, size=ref_traj.shape).astype(np.float32)
            noisy_traj = ref_traj + noise
        else:
            noisy_traj = ref_traj.copy()

        pad_w = self.window_size // 2
        padded = np.pad(noisy_traj, ((pad_w, pad_w), (0, 0)), mode='edge')
        kernel = np.ones(self.window_size, dtype=np.float32) / self.window_size
        
        smoothed = np.zeros_like(noisy_traj)
        for d in range(D):
            smoothed[:, d] = np.convolve(padded[:, d], kernel, mode='valid')[:T]
            
        # Velocity approximation
        vel = np.zeros_like(smoothed)
        if T > 1:
            vel[1:] = np.diff(smoothed, axis=0)
            
        return smoothed, vel, np.zeros_like(smoothed)


class BaseGeneratorRefiner:
    """
    Raw Base Generator (Method 1: Unrefined Raw Prototype Concatenation)
    """
    def refine(self, ref_traj, disturbance_delta=0.0, rng=None):
        T, D = ref_traj.shape
        if disturbance_delta > 0:
            if rng is None:
                rng = np.random.default_rng(2026)
            noise = rng.uniform(-disturbance_delta, disturbance_delta, size=ref_traj.shape).astype(np.float32)
            out = ref_traj + noise
        else:
            out = ref_traj.copy()
            
        vel = np.zeros_like(out)
        if T > 1:
            vel[1:] = np.diff(out, axis=0)
        return out, vel, np.zeros_like(out)


# ==========================================
# Kinematic and Physical Consistency Metrics
# ==========================================

def compute_kinematic_rms(traj, order=1):
    """
    Computes RMS of trajectory temporal differences:
    order=1: Velocity RMS
    order=2: Acceleration RMS
    order=3: Jerk RMS (Smoothness / Impact metric)
    """
    x = traj.astype(np.float64)
    for _ in range(order):
        if len(x) < 2:
            return 0.0
        x = np.diff(x, axis=0)
    return float(np.sqrt(np.mean(x * x)))


def compute_bone_length_variance(traj, joint_pairs):
    """
    Computes bone length variance (rigidity violation metric).
    traj: shape (T, num_joints * 2) or (T, num_joints, 2)
    joint_pairs: list of (parent_idx, child_idx)
    """
    T = len(traj)
    if T < 2:
        return 0.0
    
    if traj.ndim == 2:
        kps = traj.reshape(T, -1, 2)
    else:
        kps = traj
        
    variances = []
    for (i, j) in joint_pairs:
        if i >= kps.shape[1] or j >= kps.shape[1]:
            continue
        lengths = np.linalg.norm(kps[:, i, :] - kps[:, j, :], axis=-1)
        variances.append(float(np.var(lengths)))
        
    return float(np.mean(variances)) if variances else 0.0
