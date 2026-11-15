import numpy as np


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


class _TrajectoryTracker:
    def __init__(self, dt=0.1, relativeState=False, k_long=1.0, v_ref_steps=10,
                 v_bounds=(-1.5, 1.5), w_bounds=(-1.5, 1.5), dv_max=1.0, dw_max=1.0):
        self.dt = dt
        self.relativeState = relativeState
        self.k_long = k_long
        self.v_ref_steps = v_ref_steps
        self.v_min, self.v_max = v_bounds
        self.w_min, self.w_max = w_bounds
        self.dv_max, self.dw_max = dv_max, dw_max
        self.last_u = np.zeros(2)
        self.solve_time = 0.0

    def reset(self):
        self.last_u = np.zeros(2)

    def get_solve_time(self):
        return self.solve_time

    def _to_ego_frame(self, current_state, traj):
        traj = np.asarray(traj, dtype=float)
        if traj.ndim == 1:
            traj = traj[None, :]
        if traj.shape[1] == 4:
            traj = np.stack([traj[:, 0], traj[:, 1], np.arctan2(traj[:, 2], traj[:, 3])], axis=1)
        if self.relativeState:
            return traj
        x, y, th = current_state
        c, s = np.cos(th), np.sin(th)
        dx, dy = traj[:, 0] - x, traj[:, 1] - y
        return np.stack([c * dx + s * dy, -s * dx + c * dy, _wrap(traj[:, 2] - th)], axis=1)

    def _speed(self, P):
        n = min(self.v_ref_steps, len(P) - 1)
        if n >= 1:
            seg = np.hypot(np.diff(P[:n + 1, 0]), np.diff(P[:n + 1, 1]))
            v_ref = seg.sum() / (n * self.dt)
        else:
            v_ref = 0.0
        # Distance to the pose the ego should occupy now, along the ego heading
        e_long = P[0, 0]
        return v_ref + self.k_long * e_long

    def _limit(self, v, w):
        v = np.clip(v, self.v_min, self.v_max)
        w = np.clip(w, self.w_min, self.w_max)
        v = np.clip(v, self.last_u[0] - self.dv_max, self.last_u[0] + self.dv_max)
        w = np.clip(w, self.last_u[1] - self.dw_max, self.last_u[1] + self.dw_max)
        self.last_u = np.array([v, w])
        return self.last_u

    @staticmethod
    def path_heading(P, min_seg=0.02):
        th = P[:, 2].copy()
        if len(P) < 2:
            return th
        d = np.gradient(P[:, :2], axis=0)
        ok = np.hypot(d[:, 0], d[:, 1]) >= min_seg / 2
        th[ok] = np.arctan2(d[ok, 1], d[ok, 0])
        return th

    @staticmethod
    def make_control_callable(u):
        return lambda obj, t, s: u

    def demand(self, current_state, traj, noise=None):
        import time
        t0 = time.perf_counter()
        P = self._to_ego_frame(current_state, traj)
        v = self._speed(P)
        w = self._steer(P, v)
        u = self._limit(v, w)
        self.solve_time = time.perf_counter() - t0
        return self.make_control_callable(u.copy())

    def _steer(self, P, v):
        raise NotImplementedError


class PurePursuit(_TrajectoryTracker):

    def __init__(self, target='leader', speed_mode=None, gap=None, k_gap=0.5, ki_gap=0.05,
                 ld_min=0.5, k_ld=1.0, k_turn=1.0, v_stop=0.05, **kw):
        super().__init__(**kw)
        assert target in ('leader', 'lookahead')
        self.target = target
        self.speed_mode = speed_mode or ('gap' if target == 'leader' else 'schedule')
        assert self.speed_mode in ('gap', 'schedule')
        self.gap, self.k_gap, self.ki_gap = gap, k_gap, ki_gap
        self.ld_min, self.k_ld, self.k_turn, self.v_stop = ld_min, k_ld, k_turn, v_stop
        self._gap_ref, self._gap_int = gap, 0.0

    def reset(self):
        super().reset()
        self._gap_ref, self._gap_int = self.gap, 0.0

    def _speed(self, P):
        if self.speed_mode == 'schedule':
            return super()._speed(P)
        d = float(np.hypot(P[-1, 0], P[-1, 1]))
        if self._gap_ref is None:
            self._gap_ref = d
        e = d - self._gap_ref
        v = self.k_gap * e + self.ki_gap * self._gap_int
        if self.v_min < v < self.v_max:
            self._gap_int += e * self.dt
        return v

    def _steer(self, P, v):
        if self.target == 'leader':
            g = P[-1]
        else:
            d = np.hypot(P[:, 0], P[:, 1])
            start = int(np.argmin(d))
            ld = max(self.ld_min, self.k_ld * abs(v))
            far = np.nonzero(d[start:] >= ld)[0]
            g = P[start + far[0]] if far.size else P[-1]
        xg, yg = g[0], g[1]
        dist2 = xg * xg + yg * yg
        if dist2 < 1e-9:
            return 0.0
        if abs(v) < self.v_stop or xg < 0:
            return self.k_turn * np.arctan2(yg, xg)
        return v * 2.0 * yg / dist2


class Stanley(_TrajectoryTracker):
    def __init__(self, k=1.0, k_soft=0.5, wheelbase=0.5, max_steer=1.0, k_turn=1.0, v_stop=0.05, **kw):
        super().__init__(**kw)
        self.k, self.k_soft, self.wheelbase = k, k_soft, wheelbase
        self.max_steer, self.k_turn, self.v_stop = max_steer, k_turn, v_stop

    def _steer(self, P, v):
        fx, fy = self.wheelbase, 0.0
        j = int(np.argmin(np.hypot(P[:, 0] - fx, P[:, 1] - fy)))
        xr, yr = P[j, 0], P[j, 1]
        thr = self.path_heading(P)[j]
        e = -np.sin(thr) * (fx - xr) + np.cos(thr) * (fy - yr)
        psi = _wrap(thr)
        delta = psi + np.arctan2(-self.k * e, self.k_soft + abs(v))
        delta = float(np.clip(delta, -self.max_steer, self.max_steer))
        if abs(v) < self.v_stop:
            return self.k_turn * delta
        return v * np.tan(delta) / self.wheelbase
