"""Modèle pneumatique simplifié : Coulomb + cercle de friction."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Tire:
    mu: float = 1.5  # coefficient d'adhérence (valeur illustrative)

    def __post_init__(self) -> None:
        if self.mu <= 0:
            raise ValueError("mu doit être strictement positif")

    def max_force(self, fz: float) -> float:
        """Force tangentielle max F = mu * Fz [N]."""
        return self.mu * fz

    def available_lateral(self, fx: float, fz: float) -> float:
        """Force latérale restante quand le pneu fournit déjà Fx.

        Cercle de friction : Fy = sqrt((mu Fz)^2 - Fx^2).
        """
        fmax = self.max_force(fz)
        if abs(fx) > fmax:
            return 0.0
        return math.sqrt(fmax**2 - fx**2)