"""Vitesse limite en virage stationnaire, avec et sans appui.

Sans appui : v = sqrt(mu g R)
Avec appui : m v^2 / R <= mu (m g + 1/2 rho Cl A v^2)
  => v = sqrt( mu m g / (m/R - 1/2 mu rho Cl A) )

Valide seulement si le dénominateur est > 0. Sinon l'appui croît plus
vite que la force latérale requise : il n'y a plus de limite d'adhérence
dans ce modèle (on renvoie inf).
"""
import math

from racelab.models.tires import Tire
from racelab.models.vehicle import Vehicle
from racelab.constants import G


def v_max_no_aero(tire: Tire, radius: float) -> float:
    """Vitesse limite sans appui [m/s]."""
    return math.sqrt(tire.mu * G * radius)


def v_max_aero(vehicle: Vehicle, tire: Tire, radius: float) -> float:
    """Vitesse limite avec appui [m/s]."""
    m, mu = vehicle.mass, tire.mu
    k = 0.5 * vehicle.rho * vehicle.cl * vehicle.frontal_area
    denom = m / radius - mu * k
    if denom <= 0:
        return math.inf
    return math.sqrt(mu * m * G / denom)


def critical_radius(vehicle: Vehicle, tire: Tire) -> float:
    """Rayon où le dénominateur s'annule : R = m / (mu * k)."""
    k = 0.5 * vehicle.rho * vehicle.cl * vehicle.frontal_area
    return vehicle.mass / (tire.mu * k)

def main() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    vehicle, tire = Vehicle(), Tire()
    rc = critical_radius(vehicle, tire)

    # On s'arrête à 95 % du rayon critique : au-delà, la vitesse diverge.
    radii = np.linspace(20, 0.95 * rc, 200)
    v_no = [v_max_no_aero(tire, r) * 3.6 for r in radii]
    v_aero = [v_max_aero(vehicle, tire, r) * 3.6 for r in radii]

    plt.plot(radii, v_no, label="Sans appui")
    plt.plot(radii, v_aero, label="Avec appui")
    plt.xlabel("Rayon du virage R [m]")
    plt.ylabel("Vitesse limite [km/h]")
    plt.title("Vitesse limite en virage")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/figures/cornering.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()