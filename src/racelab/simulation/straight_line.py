"""Accélération longitudinale en fonction de la vitesse (ligne droite)."""
import matplotlib.pyplot as plt
import numpy as np

from racelab.models.tires import Tire
from racelab.models.vehicle import Vehicle
from racelab.physics.dynamics import (
    drive_force,
    longitudinal_acceleration,
    max_speed_power_limited,
)


def main() -> None:
    vehicle, tire = Vehicle(), Tire()
    v_top = max_speed_power_limited(vehicle)
    speeds = np.linspace(1, v_top, 300)

    ax = [longitudinal_acceleration(vehicle, tire, v) for v in speeds]
    # Régime : la force motrice est-elle limitée par la puissance ou l'adhérence ?
    power_limited = [
        drive_force(vehicle, tire, v) >= vehicle.power / v - 1e-9 for v in speeds
    ]
    transition = next(
        (v for v, p in zip(speeds, power_limited) if p), None
    )

    plt.plot(speeds * 3.6, ax)
    if transition is not None:
        plt.axvline(transition * 3.6, linestyle="--", color="gray",
                    label=f"Passage adhérence → puissance ({transition*3.6:.0f} km/h)")
    plt.xlabel("Vitesse [km/h]")
    plt.ylabel("Accélération ax [m/s²]")
    plt.title("Accélération en ligne droite")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/figures/acceleration.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()