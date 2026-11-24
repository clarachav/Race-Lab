"""Démo : profil de vitesse sur une mini-piste (droite, virage, droite)."""
import matplotlib.pyplot as plt

from racelab.models.tires import Tire
from racelab.models.track import Segment, build_track
from racelab.models.vehicle import Vehicle
from racelab.simulation.cornering import v_max_aero
from racelab.simulation.lap_time import simulate_lap


def main() -> None:
    vehicle, tire = Vehicle(), Tire()
    track = build_track([
        Segment(800),
        Segment(100, radius=50),
        Segment(600),
    ])
    result = simulate_lap(vehicle, tire, track)

    corner_limit = v_max_aero(vehicle, tire, 50) * 3.6
    print(f"Temps : {result.lap_time:.2f} s")
    print(f"Vitesse max atteinte : {result.speed.max() * 3.6:.0f} km/h")
    print(f"Vitesse limite du virage : {corner_limit:.0f} km/h")

    plt.plot(track.s, result.speed * 3.6, label="Vitesse simulée")
    plt.axhline(corner_limit, linestyle="--", color="gray",
                label="Limite d'adhérence du virage")
    plt.axvspan(800, 900, alpha=0.15, label="Virage (R = 50 m)")
    plt.xlabel("Distance parcourue [m]")
    plt.ylabel("Vitesse [km/h]")
    plt.title(f"Profil de vitesse : {result.lap_time:.1f} s")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/figures/lap_profile.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()