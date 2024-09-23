from dataclasses import dataclass
from .base import BaseShapeConfig

@dataclass(kw_only=True)
class CylinderDexDeformShapeConfig(BaseShapeConfig, name='cylinder_dexdeform'):
    center: tuple[float, float, float] = (0.5, 0.38, 0.6)
    size: tuple[float, float, float] = (0.003, 0.1, 0.0)
    num_particles: int = 10000
    vol: float = (1. / 64 / 2) ** 2
    mode: str = 'random_symmetric'
