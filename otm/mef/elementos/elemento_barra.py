from otm.mef.elementos.base_elemento_poligonal import BaseElementoPoligonal
from otm.mef.elementos.elemento import Elemento
import numpy as np
from otm.mef.materiais import Material
from math import dist
from otm.mef.excecoes import ErroMEF


class ElementoBarra(Elemento):
    def __init__(self, nos: np.ndarray, material: Material, area_secao: float = 1):
        super().__init__(nos, material)

        # Verfica a quantidade de elementos finitos
        if (n := self.numero_nos()) != 2:
            raise ErroMEF(f'Uma barra deve conter 2 nós! Nós identificados: {n}')

        self.area_secao = area_secao

    def comprimento(self) -> float:
        """Retorna o comprimento da barra"""
        return dist(*self.nos)

    def matriz_rigidez(self):
        pass

    def angulo_inclinacao(self):
        pass

    def matriz_rotacao(self):
        pass
