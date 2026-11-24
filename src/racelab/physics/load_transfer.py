"""Charges sur les essieux et transfert de charge longitudinal.

Convention : ax > 0 = accélération vers l'avant, ax < 0 = freinage.
Transfert : dFz = m * ax * h / L  (vers l'arrière quand on accélère).
"""
from racelab.constants import G
from racelab.models.vehicle import Vehicle


def static_axle_loads(vehicle: Vehicle) -> tuple[float, float]:
    """Charges statiques (avant, arrière) [N], voiture à l'arrêt."""
    weight = vehicle.mass * G
    front = weight * vehicle.l_r / vehicle.wheelbase
    rear = weight * vehicle.l_f / vehicle.wheelbase
    return front, rear


def longitudinal_transfer(vehicle: Vehicle, ax: float) -> float:
    """Transfert de charge dFz [N], positif vers l'arrière."""
    return vehicle.mass * ax * vehicle.cg_height / vehicle.wheelbase


def axle_loads(vehicle: Vehicle, ax: float) -> tuple[float, float]:
    """Charges (avant, arrière) [N] sous l'accélération longitudinale ax."""
    front, rear = static_axle_loads(vehicle)
    transfer = longitudinal_transfer(vehicle, ax)
    return front - transfer, rear + transfer