import numpy as np
import shapely.geometry as sh_geo
from otm.mef.elementos_finitos.elemento import Elemento
from typing import List
from otm.mef.materiais import Material

__all__ = ['BaseElementoPoligonal']


class BaseElementoPoligonal(Elemento):
    """Classe abstrata que implementa as propriedades comuns dos elementos finitos poligonais reais e
    isoparamétricos"""
    banco_funcoes_forma = {}
    banco_diff_funcoes_forma = {}
    banco_pontos_pesos_gauss = {}

    # Dados para a integração numérica
    pesos_gauss = np.array(3 * [1 / 3])
    pontos_gauss = np.array([[1 / 6, 1 / 6],
                             [2 / 3, 1 / 6],
                             [1 / 6, 2 / 3]])

    def __init__(self, nos: np.ndarray, material: Material):
        super().__init__(nos, material)

    @property
    def num_nos(self) -> int:
        """Retorna o número de lados do polígono"""
        return len(self.nos)

    def poligono(self) -> sh_geo.Polygon:
        """Retorna o elemento como um polígono do Shapely."""
        return sh_geo.Polygon(self.nos)

    def centroide(self) -> tuple:
        """Retorna o centroide do polígono"""
        p = self.poligono().centroid
        return p.x, p.y

    def diametro_equivalente(self) -> float:
        """Retorna o diâmetro equivalente do elemento em função de sua área."""
        return 2 * np.sqrt(self.area() / np.pi)

    def area(self) -> float:
        """Retorna a área do elemento"""
        return self.poligono().area

    def triangular_poligono(self) -> List[np.ndarray]:
        """Discretiza o elemento em triângulos.

        Returns:
            Retorna uma lista com as coordenadas de cada triângulo.
        """
        # O número de triângulos é igual ao de lados do polígono
        triangulos = []
        # Replicação do primeiro nó para valer a lógica abaixo
        nos = np.concatenate((self.nos, np.array([self.nos[0]])))
        c = self.centroide()

        for v in range(self.num_nos):
            triangulos.append(np.array([c, nos[v], nos[v + 1]]))
        return triangulos
