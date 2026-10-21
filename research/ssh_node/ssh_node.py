#!/usr/bin/env python3
"""SSHNode -- one neuron whose hidden state is an N-site SSH chain.

A drop-in for `core/runtime/runtime.py`'s `ResonatorState`: same public
surface (`freq_hz`, `damping`, `coupling`, `threshold`, `energy_ema`) and
the same `step(drive, dt) -> bool` contract (returns True the instant the
edge-crossing of `sqrt(energy_ema) >= threshold` happens). The difference
is internal: instead of a single damped driven oscillator, the hidden
state is a chain of N such oscillators with a **staggered** internal
coupling (Su-Schrieffer-Heeger dimerisation: bonds alternate t1, t2, t1,
...), and the output is read from ONE site -- the topological edge site
(site 0) -- with **no training**. The readout filter is fixed entirely by
the coupling geometry.

Per-site dynamics mirror `ResonatorState.step()` verbatim (symplectic
Euler, the -2 damping omega v term, the gated x^2 energy EMA), extended
with the coupling term:

    accel_i = -omega_i^2 x_i - 2 damping omega_i v_i + (K @ x)_i + b_i * drive

K is spectrally normalised so rho(K) < RHO * min(omega_i^2) (RHO defaults
to 0.7). Without this the chain diverges -- the effective stiffness
diag(omega^2) - K stops being positive definite and the integrator blows
up to NaN on the first strong drive (the lesson from
research/ssh_resonator_bridge.py's first run).

Topological phase:  |t2| > |t1|  (g > 0)   -> localised edge mode at site 0
Trivial phase:      |t1| > |t2|  (g < 0)   -> no edge mode
Uniform (control):  t1 == t2     (g = 0)   -> gapless chain, no localisation

Defect mechanisms (apply AFTER construction, never to the readout site):
  decouple_sites(...)  -- chiral-preserving: zero the site's row/col of K,
                          its input coupling b_i, and freeze x_i = v_i = 0.
                          A dead node that is cleanly removed from the chain.
  detune_sites(...)    -- chiral-breaking: shift a site's own omega_i.
                          On-site disorder; breaks the sublattice symmetry.

The two are kept separate because every prior substrate in this portfolio
(topological-phononics, the photonic array, the Spikeling LIF chain) showed
the edge-vs-bulk *distinction* transferring while the specific
"chiral-preserving damage is the protected case" *mechanism* did not.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

RHO = 0.7  # keep rho(K) below RHO * min(omega^2) for integrator stability


def ssh_coupling(n_sites: int, g: float) -> np.ndarray:
    """Symmetric N x N nearest-neighbour SSH coupling, bonds alternating
    t1 = 1 - g (intra-cell) and t2 = 1 + g (inter-cell). NOT yet
    spectrally normalised -- SSHNode does that against its own omega."""
    if n_sites % 2 != 0:
        raise ValueError("SSH chain needs an even number of sites (N/2 unit cells)")
    t1, t2 = 1.0 - g, 1.0 + g
    K = np.zeros((n_sites, n_sites))
    for i in range(n_sites - 1):
        K[i, i + 1] = K[i + 1, i] = t1 if i % 2 == 0 else t2
    return K


@dataclass
class SSHNode:
    name: str
    freq_hz: float
    damping: float
    coupling: float                     # scales `drive` into the chain (per-site, see drive_mode)
    n_sites: int = 8
    dimerization: float = 0.6           # g: >0 topological, <0 trivial, 0 uniform
    drive_mode: str = "all"             # "all" -> b_i = 1 for every site (matches step_resonators'
                                        # single shared drive); "edge" -> b_0 = 1 only (a physical
                                        # SSH chain driven at its boundary)
    readout_site: int = 0               # the topological edge site

    # --- surface kept identical to ResonatorState so callers don't branch ---
    threshold: float = 0.0008
    gate_threshold: float = 0.00024
    energy_time_constant: float = 0.0025
    energy_ema: float = 0.0

    # internal state (initialised in __post_init__)
    x: np.ndarray = field(default=None, repr=False)
    v: np.ndarray = field(default=None, repr=False)
    _omega2: np.ndarray = field(default=None, repr=False)
    _omega: np.ndarray = field(default=None, repr=False)
    _K: np.ndarray = field(default=None, repr=False)
    _b: np.ndarray = field(default=None, repr=False)
    _frozen: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        n = self.n_sites
        self.x = np.zeros(n)
        self.v = np.zeros(n)
        omega0 = 2.0 * math.pi * self.freq_hz
        self._omega = np.full(n, omega0, dtype=float)
        self._omega2 = self._omega ** 2
        self._frozen = np.zeros(n, dtype=bool)

        b = np.zeros(n)
        if self.drive_mode == "all":
            b[:] = self.coupling
        elif self.drive_mode == "edge":
            # inject only at the readout edge (a physical SSH chain is
            # boundary-driven) but boost so the TOTAL injected drive equals
            # the "all" mode's -- keeps input power comparable across arms.
            b[self.readout_site] = self.coupling * n
        else:
            raise ValueError(f"drive_mode must be 'all' or 'edge', got {self.drive_mode!r}")
        self._b = b

        self._set_K(ssh_coupling(n, self.dimerization))

    # ------------------------------------------------------------------ K
    def _set_K(self, K_raw: np.ndarray) -> None:
        """Spectrally normalise so the chain is stable, then store."""
        spec = float(np.max(np.abs(np.linalg.eigvalsh(K_raw)))) if np.any(K_raw) else 0.0
        scale = (RHO * float(np.min(self._omega2)) / (spec + 1e-12)) if spec > 0 else 1.0
        self._K = K_raw * scale

    # -------------------------------------------------------------- defects
    def decouple_sites(self, sites: Iterable[int]) -> SSHNode:
        """Chiral-preserving: cleanly remove sites from the chain."""
        raw = ssh_coupling(self.n_sites, self.dimerization)
        for s in sites:
            if s == self.readout_site:
                raise ValueError("refusing to damage the readout site")
            raw[s, :] = 0.0
            raw[:, s] = 0.0
            self._b[s] = 0.0
            self._frozen[s] = True
            self.x[s] = 0.0
            self.v[s] = 0.0
        self._set_K(raw)
        return self

    def detune_sites(self, deltas: Mapping[int, float]) -> SSHNode:
        """Chiral-breaking: omega_i -> omega_i * (1 + delta) for the given sites."""
        for s, d in deltas.items():
            if s == self.readout_site:
                raise ValueError("refusing to damage the readout site")
            self._omega[s] = 2.0 * math.pi * self.freq_hz * (1.0 + d)
        self._omega2 = self._omega ** 2
        # a detune changes min(omega^2); renormalise K against the new floor
        self._set_K(ssh_coupling(self.n_sites, self.dimerization) * self._decoupled_mask())
        return self

    def _decoupled_mask(self) -> np.ndarray:
        m = np.ones((self.n_sites, self.n_sites))
        dead = np.where(self._frozen)[0]
        for s in dead:
            m[s, :] = 0.0
            m[:, s] = 0.0
        return m

    def _k_scale(self) -> float:
        raw = ssh_coupling(self.n_sites, self.dimerization) * self._decoupled_mask()
        spec = float(np.max(np.abs(np.linalg.eigvalsh(raw)))) if np.any(raw) else 0.0
        return (RHO * float(np.min(self._omega2)) / (spec + 1e-12)) if spec > 0 else 1.0

    # ---------------------------------------------------------------- step
    def step(self, drive: float, dt: float) -> bool:
        """One symplectic-Euler step of the whole chain; edge-triggered
        fire on `sqrt(energy_ema) >= threshold`, exactly like
        ResonatorState.step()."""
        accel = (-self._omega2 * self.x
                 - 2.0 * self.damping * self._omega * self.v
                 + self._K @ self.x
                 + self._b * drive)
        self.v += accel * dt
        self.x += self.v * dt
        if self._frozen.any():                       # keep dead sites at rest
            self.x[self._frozen] = 0.0
            self.v[self._frozen] = 0.0

        xe = float(self.x[self.readout_site])
        alpha = min(1.0, dt / self.energy_time_constant)
        was_above = math.sqrt(self.energy_ema) >= self.threshold
        if abs(xe) >= self.gate_threshold:
            self.energy_ema += alpha * (xe * xe - self.energy_ema)
        else:
            self.energy_ema -= alpha * self.energy_ema
        now_above = math.sqrt(self.energy_ema) >= self.threshold
        return now_above and not was_above

    # ------------------------------------------------------------- helpers
    def reset(self) -> None:
        self.x[:] = 0.0
        self.v[:] = 0.0
        self.energy_ema = 0.0

    @property
    def omega(self) -> float:
        return float(self._omega[self.readout_site])

    @property
    def edge_localization(self) -> float:
        """|psi_0|^2 of the SSH mid-gap edge state: the eigenvector of the
        COUPLING matrix K (restricted to live sites) whose eigenvalue is
        closest to 0. With uniform on-site omega^2 the dynamical matrix
        diag(omega^2) - K shares K's eigenvectors, so this K=0 mode is the
        one a drive at frequency omega excites. Topological (t2>t1) ->
        localised at site 0 -> value near 1; trivial/uniform -> delocalised
        -> value near 1/N. Diagnostics only, not part of step()."""
        live = ~self._frozen
        Kl = self._K[np.ix_(live, live)]
        w, V = np.linalg.eigh(Kl)
        psi = V[:, int(np.argmin(np.abs(w)))]
        return float(psi[0] ** 2 / (psi @ psi))
