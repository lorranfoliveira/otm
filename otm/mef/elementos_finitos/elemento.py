import numpy as np
from otm.mef.materiais import Material


class Elemento:
    """Classe que implementa as propriedades em comum dos elementos finitos"""

    def __init__(self, nos: np.ndarray, material: Material):
        self.nos = nos
        self.material = material

    def numero_nos(self) -> int:
        """Retorna o número de nós que fazem a composição do elemento"""
        return self.nos.shape[0]
