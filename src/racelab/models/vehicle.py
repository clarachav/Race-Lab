"""Modèle de véhicule. Toutes les unités sont SI."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Vehicle:
    mass: float = 800.0          # m  [kg], avec pilote
    wheelbase: float = 3.6       # L  [m]
    l_f: float = 1.6             # CG -> essieu avant [m]
    l_r: float = 2.0             # CG -> essieu arrière [m]
    cg_height: float = 0.30      # h  [m]
    rho: float = 1.225           # densité de l'air [kg/m^3]
    frontal_area: float = 1.5    # A  [m^2]
    power: float = 750_000.0     # P  [W], puissance moteur (≈ 1000 ch, illustratif)
    cd: float = 0.9              # coefficient de traînée
    cl: float = 3.5              # coefficient d'appui

    def __post_init__(self) -> None:
        for name in ("mass", "wheelbase", "l_f", "l_r", "cg_height",
                     "rho", "frontal_area", "power"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} doit être strictement positif")
        if self.cd < 0 or self.cl < 0:
            raise ValueError("cd et cl doivent être >= 0")
        if abs(self.l_f + self.l_r - self.wheelbase) > 1e-9:
            raise ValueError("l_f + l_r doit être égal à l'empattement")
        