#!/usr/bin/env python3
"""hopf_theta.py - the Jacobi theta line bundle on a torus: a degree-one line bundle like the Hopf one, but on T^2 instead of S^2.

The torus is C / (Z + tau Z).  The odd Jacobi theta function

    theta1(z | tau) = 2 sum_{n >= 0} (-1)^n q^{(n + 1/2)^2} sin((2n + 1) pi z),    q = e^{i pi tau}

is not periodic but quasi-periodic, theta1(z + 1) = -theta1(z) and theta1(z + tau) = -e^{-i pi tau - 2 pi i z} theta1(z).  So it is not a
function on the torus but a section of a line bundle whose transition functions are those two factors: the theta (Landau-level) bundle,
whose Hermitian norm is ||s||^2 = |theta1(z)|^2 e^{-2 pi y^2 / Im tau}, y = Im z.  Its Chern number is 1, computed here by routes that share
no formula:

  (a) zeros: the argument principle (1/2 pi i) contour integral of theta1'/theta1 round a fundamental parallelogram counts one zero;
  (b) transition functions: the phase of theta1(z + tau)/theta1(z), unwrapped along the edge z -> z + 1, winds once (theta1 evaluated, the
      factor formula never used);
  (c) Chern-Weil: the curvature of the metric e^{-2 pi y^2 / Im tau}, integrated by finite differences over the torus, is 1 (the curvature is
      the constant 4 pi / Im tau, so this is an analytic identity; the grid only sums it);
  (d) Poincare-Lelong: the flux of grad log ||s||^2 through a small circle round the zero is 4 pi (1 - pi r^2 / Im tau), i.e. one unit of
      delta function at the zero balanced by the uniform curvature;
  (e) Hopf pullback: the tautological line bundle of hopf_line_bundle.py pulled back to T^2 by the degree-one map
      k -> (sin k1, sin k2, m + cos k1 + cos k2) (normalised), 0 < m < 2, has lattice (Fukui-Hatsugai-Suzuki) Chern number of magnitude 1.
      Line bundles on T^2 are classified by c1, so it is isomorphic to the theta bundle or to its dual. With m = 3 the map has degree 0
      and the Chern number is 0 (control).

Two more controls: with the opposite-sign weight e^{+2 pi y^2 / Im tau} the norm is not doubly periodic, and the even theta3 has its zero
at (1 + tau)/2 rather than at 0.  Also checked: the flat connection on the trivial bundle with holonomies (e(a), e(b)) round the two
cycles has a holonomy group with exactly lcm(den a, den b) elements. Its curvature is 0, so its holonomy group can be a finite subgroup
of the structure group U(1) (see HOPF_WITHOUT_PRINCIPAL_BUNDLES.md section 6).

    python3 hopf_theta.py --tests
"""
import math
import sys
from fractions import Fraction

import numpy as np

import hopf_line_bundle as L

TAU = 0.31 + 1.17j


def theta1(z, tau=TAU, n=30):
    q = np.exp(1j * np.pi * tau)
    k = np.arange(n)
    z = np.asarray(z, complex)[..., None]
    return 2 * np.sum((-1.0) ** k * q ** ((k + 0.5) ** 2) * np.sin((2 * k + 1) * np.pi * z), axis=-1)


def dtheta1(z, tau=TAU, n=30):
    q = np.exp(1j * np.pi * tau)
    k = np.arange(n)
    z = np.asarray(z, complex)[..., None]
    return 2 * np.sum((-1.0) ** k * q ** ((k + 0.5) ** 2) * (2 * k + 1) * np.pi * np.cos((2 * k + 1) * np.pi * z), axis=-1)


def theta3(z, tau=TAU, n=30):
    q = np.exp(1j * np.pi * tau)
    k = np.arange(1, n)
    z = np.asarray(z, complex)[..., None]
    return 1 + 2 * np.sum(q ** (k ** 2) * np.cos(2 * k * np.pi * z), axis=-1)


def log_norm2(z, tau=TAU, sign=-1):
    """log ||s||^2 = log |theta1|^2 + sign * 2 pi y^2 / Im tau (sign = -1 is the metric that makes it a function on the torus)."""
    return np.log(np.abs(theta1(z, tau)) ** 2) + sign * 2 * np.pi * np.imag(z) ** 2 / tau.imag


def zeros_by_argument_principle(f, df, tau=TAU, z0=-0.37 - 0.41 * TAU, n=4000):
    """(1/2 pi i) contour integral of f'/f round the parallelogram z0, z0+1, z0+1+tau, z0+tau (midpoint rule on each edge)."""
    corners = [z0, z0 + 1, z0 + 1 + tau, z0 + tau, z0]
    total = 0
    t = (np.arange(n) + 0.5) / n
    for a, b in zip(corners, corners[1:]):
        z = a + (b - a) * t
        total += np.sum(df(z, tau) / f(z, tau)) * (b - a) / n
    return total / (2j * np.pi)


def transition_winding(tau=TAU, z0=0.13 + 0.29j, n=2000):
    """Winding number of the phase of theta1(z + tau)/theta1(z) as z runs over the edge z0 -> z0 + 1 (minus sign: c1 = -winding)."""
    z = z0 + np.linspace(0, 1, n + 1)
    ph = np.unwrap(np.angle(theta1(z + tau, tau) / theta1(z, tau)))
    return -(ph[-1] - ph[0]) / (2 * np.pi)


def chern_weil(tau=TAU, n=200):
    """c1 = (1/4 pi) * integral over the torus of -Laplacian(log h), h = e^{-2 pi y^2 / Im tau}, by central differences on an n x n grid."""
    u = (np.arange(n) + 0.5) / n
    U, V = np.meshgrid(u, u, indexing="ij")
    Z = U + V * tau
    logh = lambda z: -2 * np.pi * np.imag(z) ** 2 / tau.imag
    lap = (logh(Z + 1e-3) + logh(Z - 1e-3) + logh(Z + 1e-3j) + logh(Z - 1e-3j) - 4 * logh(Z)) / 1e-6
    area = tau.imag                                      # |1 x tau|
    return float(np.mean(-lap) * area / (4 * np.pi))


def flux_round_zero(r, tau=TAU, n=2000, h=1e-6):
    """Flux of grad log ||s||^2 out of the circle |z| = r round the zero z = 0 of theta1 (Poincare-Lelong)."""
    t = 2 * np.pi * np.arange(n) / n
    z = r * np.exp(1j * t)
    dn = (log_norm2(z * (1 + h / r)) - log_norm2(z * (1 - h / r))) / (2 * h)
    return float(np.sum(dn) * 2 * np.pi * r / n)


def norm_periodicity(tau=TAU, sign=-1, m=50, seed=3):
    """max over random z of |log||s(z + 1)||^2 - log||s(z)||^2| and the same for z + tau."""
    rng = np.random.default_rng(seed)
    z = rng.random(m) + rng.random(m) * tau
    return float(max(np.max(np.abs(log_norm2(z + 1, tau, sign) - log_norm2(z, tau, sign))),
                     np.max(np.abs(log_norm2(z + tau, tau, sign) - log_norm2(z, tau, sign)))))


def hopf_pullback_chern(m, n=40):
    """FHS lattice Chern number of the tautological line bundle (hopf_line_bundle.state_of) pulled back to T^2 by the map
    k -> d(k)/|d(k)|, d = (sin k1, sin k2, m + cos k1 + cos k2)."""
    ks = 2 * np.pi * np.arange(n) / n
    psi = np.empty((n, n, 2), complex)
    for i, a in enumerate(ks):
        for j, b in enumerate(ks):
            d = np.array([math.sin(a), math.sin(b), m + math.cos(a) + math.cos(b)])
            psi[i, j] = L.state_of(d / np.linalg.norm(d))
    link = lambda x, y: np.vdot(x, y) / abs(np.vdot(x, y))
    F = 0.0
    for i in range(n):
        for j in range(n):
            a, b, c, d = psi[i, j], psi[(i + 1) % n, j], psi[(i + 1) % n, (j + 1) % n], psi[i, (j + 1) % n]
            F += np.angle(link(a, b) * link(b, c) * link(c, d) * link(d, a))
    return F / (2 * np.pi)


def flat_holonomy_group(a, b):
    """The holonomy group of the flat connection on the trivial line bundle over T^2 with holonomies e(a), e(b) round the cycles:
    generated by those two phases, enumerated as a set of fractions mod 1."""
    a, b = Fraction(a) % 1, Fraction(b) % 1
    group, frontier = {Fraction(0)}, [Fraction(0)]
    while frontier:
        g = frontier.pop()
        for h in ((g + a) % 1, (g + b) % 1):
            if h not in group:
                group.add(h)
                frontier.append(h)
    return group


def run_tests(verbose=True):
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print("  %-4s %s %s" % ("ok" if cond else "FAIL", name, extra))

    rng = np.random.default_rng(7)
    z = rng.random(20) + rng.random(20) * TAU
    e1 = np.max(np.abs(theta1(z + 1) / theta1(z) + 1))
    e2 = np.max(np.abs(theta1(z + TAU) / theta1(z) + np.exp(-1j * np.pi * TAU - 2j * np.pi * z)))
    check("quasi-periodicity theta1(z+1) = -theta1, theta1(z+tau) = -e(-tau/2 - z) theta1 (relative)", max(e1, e2) < 1e-8,
          "(%.1e)" % max(e1, e2))
    N = zeros_by_argument_principle(theta1, dtheta1)
    check("(a) argument principle: one zero per cell", abs(N - 1) < 1e-8, "(%.10f%+.1ei)" % (N.real, N.imag))
    w = transition_winding()
    check("(b) transition function theta1(z+tau)/theta1(z) winds once", abs(w - 1) < 1e-9, "(%.10f)" % w)
    c = chern_weil()
    check("(c) Chern-Weil: curvature of e^{-2 pi y^2/Im tau} integrates to 1", abs(c - 1) < 1e-6, "(%.8f)" % c)
    r = 0.15
    fl, pred = flux_round_zero(r), 4 * np.pi * (1 - np.pi * r ** 2 / TAU.imag)
    check("(d) Poincare-Lelong: flux of grad log||s||^2 round the zero = 4 pi (1 - pi r^2/Im tau)", abs(fl - pred) < 1e-5,
          "(%.6f vs %.6f)" % (fl, pred))
    p = norm_periodicity()
    check("the norm |theta1|^2 e^{-2 pi y^2/Im tau} is doubly periodic", p < 1e-10, "(%.1e)" % p)
    pc = norm_periodicity(sign=+1)
    check("control: with e^{+2 pi y^2/Im tau} it is not", pc > 1.0, "(%.2f)" % pc)
    t3 = abs(theta3(0.5 + 0.5 * TAU))
    check("control: the even theta3 vanishes at (1+tau)/2, not at 0", t3 < 1e-12 and abs(theta3(0)) > 0.5, "(%.1e)" % t3)
    c1, c3 = hopf_pullback_chern(1.0), hopf_pullback_chern(3.0)
    check("(e) Hopf tautological bundle pulled back by a degree-one map T^2 -> S^2: |C| = 1", abs(abs(c1) - 1) < 1e-9, "(%.6f)" % c1)
    check("control: the degree-zero map (m = 3) gives C = 0", abs(c3) < 1e-9, "(%.6f)" % c3)
    sizes = [len(flat_holonomy_group(Fraction(1, 12), 0)), len(flat_holonomy_group(Fraction(1, 4), Fraction(1, 6))),
             len(flat_holonomy_group(Fraction(1, 3), Fraction(2, 3)))]
    check("flat connections: holonomy groups of order 12, 12, 3 (a finite subgroup of U(1))", sizes == [12, 12, 3], str(sizes))
    return ok


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
