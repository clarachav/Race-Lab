"""Aérodynamique : F = 1/2 * rho * C * A * v^2."""
from racelab.models.vehicle import Vehicle


def _dynamic_force(vehicle: Vehicle, coeff: float, v: float) -> float:
    return 0.5 * vehicle.rho * coeff * vehicle.frontal_area * v**2


def drag(vehicle: Vehicle, v: float) -> float:
    """Traînée F_D [N], v en m/s."""
    return _dynamic_force(vehicle, vehicle.cd, v)


def downforce(vehicle: Vehicle, v: float) -> float:
    """Appui F_DF [N], v en m/s."""
    return _dynamic_force(vehicle, vehicle.cl, v)