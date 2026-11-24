"""Dynamique longitudinale en ligne droite.

Force motrice = min(P / v, mu * Fz_arrière)   (voiture à roues arrière motrices)
Newton : m * ax = F_motrice - F_traînée

Hypothèses (à documenter) : l'appui est réparti 50/50 entre les essieux,
et on néglige le transfert de charge dû à l'accélération elle-même.
"""
from racelab.models.aerodynamics import downforce, drag
from racelab.models.tires import Tire
from racelab.models.vehicle import Vehicle
from racelab.physics.load_transfer import static_axle_loads
from racelab.constants import G


def max_speed_power_limited(vehicle: Vehicle) -> float:
    """Vitesse max en ligne droite [m/s] : P = F_D * v  =>  v = (P / k_d)^(1/3)."""
    k_drag = 0.5 * vehicle.rho * vehicle.cd * vehicle.frontal_area
    return (vehicle.power / k_drag) ** (1 / 3)


def drive_force(vehicle: Vehicle, tire: Tire, v: float) -> float:
    """Force motrice disponible [N] à la vitesse v [m/s]."""
    _, rear_static = static_axle_loads(vehicle)
    fz_rear = rear_static + 0.5 * downforce(vehicle, v)
    traction_limit = tire.max_force(fz_rear)
    power_limit = vehicle.power / v if v > 0 else float("inf")
    return min(power_limit, traction_limit)


def longitudinal_acceleration(vehicle: Vehicle, tire: Tire, v: float) -> float:
    """Accélération ax [m/s^2] à la vitesse v [m/s]."""
    return (drive_force(vehicle, tire, v) - drag(vehicle, v)) / vehicle.mass

def braking_deceleration(vehicle: Vehicle, tire: Tire, v: float) -> float:
    """Décélération maximale au freinage [m/s^2] (valeur positive).

    Les 4 roues freinent à la limite d'adhérence, la traînée aide :
    m * a = mu * (m g + F_appui) + F_traînée
    Hypothèse : on néglige le transfert de charge au freinage.
    """
    fz_total = vehicle.mass * G + downforce(vehicle, v)
    return (tire.max_force(fz_total) + drag(vehicle, v)) / vehicle.mass