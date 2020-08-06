from typing import Tuple
import numpy as np

__all__ = ['MaterialIsotropico']


class Material:
    """Classe abstrata que define as propriedades de um material genérico."""

    def __init__(self, ec: float, et: float, nu: float):
        """Construtor.

        Args:
            ec: Módulo de elasticidade à compressão.
            et: Módulo de elasticidade à tração.
            nu: Coeficiente de poisson.
        """
        self.ec = ec
        self.et = et
        self.nu = nu

    def matriz_constitutiva(self) -> np.ndarray:
        """Matriz constitutiva elástica"""
        pass


class MaterialIsotropico(Material):
    def __init__(self, e0: float, nu: float):
        """Construtor.

        Args:
            e0: Módulo de elasticidade.
            nu: Coeficiente de Poisson.
        """
        super().__init__(e0, e0, nu)

    def matriz_constitutiva(self) -> np.ndarray:
        """Retorna a matriz constitutiva elástica do material."""
        e0 = self.ec
        nu = self.nu
        return (e0 / (1 - nu ** 2)) * np.array([[1, nu, 0],
                                                [nu, 1, 0],
                                                [0, 0, (1 - nu) / 2]])


class ConcretoOrtotropico(Material):
    """Classe que descreve o concreto como um material ortotrópico segundo a teoria de Darwin e Pecknold."""

    def __init__(self, ec: float, et: float, nu: float):
        super().__init__(ec, et, nu)

    def matriz_constitutiva(self):
        super().matriz_constitutiva()
