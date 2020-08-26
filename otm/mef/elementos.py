import numpy as np
from otm.mef.materiais import Material
import shapely.geometry as sh_geo
from typing import List
import symengine
from loguru import logger

__all__ = ['Elemento', 'BaseElementoPoligonal', 'ElementoPoligonalIsoparametrico', 'ElementoPoligonal']


class Elemento:
    """Classe que implementa as propriedades em comum dos elementos finitos"""

    def __init__(self, nos: np.ndarray, material: Material):
        self.nos = nos
        self.material = material

    def numero_nos(self) -> int:
        """Retorna o número de nós que fazem a composição do elemento"""
        return self.nos.shape[0]


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


class ElementoPoligonalIsoparametrico(BaseElementoPoligonal):
    """Classe que implementa as propriedades de um polígono regular de 'n' lados inscrito em uma
    circunferência de raio 1 e origem em (0, 0)"""

    def __init__(self, num_lados: int):
        """Construtor.

        Args:
            num_lados: Número de lados do elemento finito poligonal.
        """
        self._num_lados = num_lados

        super().__init__(self.coordenadas_nos_elemento_isoparametrico(), None)

        self._x = symengine.var('x')
        self._y = symengine.var('y')

    def coordenadas_nos_elemento_isoparametrico(self) -> np.ndarray:
        """Retorna as coordenadas dos vértices do elemento isoparamétrico.

        Raises:
            ValueError:
                Se o número de nós for menor que 3.
            ValueError:
                Se o tipo de saída de dado não for 'sympy', 'No' ou 'numpy'
        """
        n_lados = self.num_nos
        # Coordenadas dos pontos do elemento de referência
        return np.array([[np.cos(2 * np.pi * i / n_lados), np.sin(2 * np.pi * i / n_lados)]
                         for i in range(1, n_lados + 1)])

    @property
    def num_nos(self) -> int:
        return self._num_lados

    def funcoes_forma(self):
        """Retorna as funções isoparamétricas de um elemento poligonal com n lados, com n >= 3"""
        """Retorna uma lista com as funções de forma calculadas pelo sympy."""

        def func_symengine():
            logger.debug(f'Montando funções de forma para n={self.num_nos}')

            x, y = self._x, self._y
            indices = list(range(self._num_lados)) + list(range(self._num_lados))

            def a(i):
                x0, y0 = self.nos[indices[i - 1]]  # i - 1
                x1, y1 = self.nos[indices[i]]  # i
                return (y1 - y0) / (x0 * y1 - x1 * y0)

            def b(i):
                x0, y0 = self.nos[indices[i - 1]]  # i - 1
                x1, y1 = self.nos[indices[i]]  # i
                return (x0 - x1) / (x0 * y1 - x1 * y0)

            def kappa(i):
                x0, y0 = self.nos[indices[i - 1]]  # i - 1
                x1, y1 = self.nos[indices[i]]  # i

                if i == 0:
                    return 1
                else:
                    return kappa(i - 1) * ((a(i + 1) * (x0 - x1) + b(i + 1) * (y0 - y1)) / (
                        a(i - 1) * (x1 - x0) + b(i - 1) * (y1 - y0)))

            def reta(i):
                return 1 - a(i) * x - b(i) * y

            def produto_retas(i):
                prod = 1
                for j in range(self.num_nos):
                    if j != indices[i] and j != indices[i + 1]:
                        prod *= reta(j)
                return kappa(i) * prod

            funcoes = []
            for j in range(n := self.num_nos):
                func = (produto_retas(j) / sum(produto_retas(k) for k in range(n)))
                funcoes.append(func)

            return funcoes

        if (nlados := self.num_nos) not in self.banco_funcoes_forma:
            self.banco_funcoes_forma[nlados] = func_symengine()

        return self.banco_funcoes_forma[nlados]

    def diff_funcoes_forma(self):
        """Retorna uma matriz com as derivadas das funções de forma em relação a x (linha 0) e a y (linha 1)"""

        def diff_func():
            func_forma = self.funcoes_forma()
            return symengine.Matrix([[f.diff(v) for f in func_forma] for v in [self._x, self._y]])

        if (nlados := self.num_nos) not in self.banco_diff_funcoes_forma:
            func = diff_func()
            self.banco_diff_funcoes_forma[nlados] = symengine.lambdify(list(func.free_symbols), func)

        return self.banco_diff_funcoes_forma[nlados]

    @staticmethod
    def funcoes_forma_triangulo(x, y):
        return np.array([1 - x - y, x, y])

    @staticmethod
    def diff_funcoes_forma_triangulo():
        return np.array([[-1, 1, 0],
                         [-1, 0, 1]])

    def pontos_de_integracao(self):
        """Calcula os pontos de integração e os pesos para cada um a partir da discretização do
        elemento em triângulos
        """
        if not (self.num_nos in self.banco_pontos_pesos_gauss):
            n = self.num_nos
            pts_gauss = self.pontos_gauss
            w = self.pesos_gauss

            # A quantidade de triângulos é numericamente igual à de lados do polígono (n)
            triangulos = self.triangular_poligono()

            # Cada triângulo possui 3 pontos de integração
            pontos = np.zeros((n * len(w), 2))
            pesos = np.zeros(n * len(w))

            for i in range(n):
                for no in range(len(w)):
                    func_forma = self.funcoes_forma_triangulo(*pts_gauss[no])
                    diff_func_forma = self.diff_funcoes_forma_triangulo()
                    # Matriz jacobiana
                    j0 = triangulos[i].T @ diff_func_forma.T
                    # Conversão de coordenadas do triângulo isoparamétricas para coordenadas
                    # cartesianas do polígono isoparamétrico
                    k = i * len(w) + no
                    pontos[k, :] = func_forma.T @ triangulos[i]
                    # Valor da função no ponto de gauss considerando o peso
                    pesos[k] = 1 / 2 * np.linalg.det(j0) * w[no]

                    self.banco_pontos_pesos_gauss[self.num_nos] = [pontos, pesos]

        return self.banco_pontos_pesos_gauss[self.num_nos]


class ElementoPoligonal(BaseElementoPoligonal):
    """Classe que define as propriedades dos elementos finitos poligonais"""

    def __init__(self, nos: np.ndarray, material: Material, espessura: float = 1, id_nos=None):
        super().__init__(nos, material)
        self.espessura = espessura
        self.id_nos = id_nos

    @staticmethod
    def id_no_para_grau_liberdade(id_no) -> list:
        """Calcula os graus de liberdade em função da numeração de um nó"""
        return [2 * id_no, 2 * id_no + 1]

    def centroide(self) -> sh_geo.Point:
        """Retorna o centroide do polígono"""
        return self.poligono().centroid

    def graus_liberdade(self) -> np.ndarray:
        """Retorna os graus de liberdade do elemento considerando todos os graus de liberdade impedidos."""
        gl = []
        for no in self.id_nos:
            gls_i = self.id_no_para_grau_liberdade(no)
            gl += gls_i
        return np.array(gl, dtype=int)

    def elemento_referencia(self) -> ElementoPoligonalIsoparametrico:
        """Retorna o elemento isoparamétrico de referência."""
        return ElementoPoligonalIsoparametrico(self.num_nos)

    def diff_funcoes_forma_referencia(self):
        if (n := self.num_nos) not in self.banco_diff_funcoes_forma:
            el_ref = self.elemento_referencia()
            return el_ref.diff_funcoes_forma()
        else:
            return self.banco_diff_funcoes_forma[n]

    def pontos_integracao_referencia(self):
        if (n := self.num_nos) not in self.banco_pontos_pesos_gauss:
            el_ref = self.elemento_referencia()
            return el_ref.pontos_de_integracao()
        else:
            return self.banco_pontos_pesos_gauss[n]

    def matriz_jacobiana(self, x, y):
        """Calcula a matriz jacobiana do elemento para pontos x, y numéricos."""
        df = np.array(self.diff_funcoes_forma_referencia()(x, y)).reshape(2, self.num_nos)
        return df @ self.nos

    def jacobiano(self, x, y) -> float:
        """Determinante da matriz jacobiana em um ponto."""
        return np.linalg.det(self.matriz_jacobiana(x, y))

    def matriz_b(self, x, y):
        n = self.num_nos

        df = np.array(self.diff_funcoes_forma_referencia()(x, y)).reshape(2, n)

        h = np.linalg.inv(self.matriz_jacobiana(x, y))
        nlh = h.shape[0]
        nch = h.shape[1]

        b1 = np.array([[1, 0, 0, 0],
                       [0, 0, 0, 1],
                       [0, 1, 1, 0]])

        b2 = np.zeros((2 * nlh, 2 * nch))
        b2[np.ix_(range(nlh), range(nch))] = h
        b2[np.ix_(range(nlh, b2.shape[0]), range(nch, b2.shape[1]))] = h

        # A matriz b3 é montada transposta para facilitar o processo
        b3 = np.zeros((2 * n, 4))
        df_t = df.T
        c = 0
        for i in df_t:
            b3[c, [0, 1]] = i
            c += 1
            b3[c, [2, 3]] = i
            c += 1

        return b1 @ b2 @ b3.T

    def matriz_rigidez(self) -> np.ndarray:
        """Retorna a matriz de rigidez do elemento"""
        t = self.espessura
        bfunc = self.matriz_b
        d = self.material.matriz_constitutiva()

        # Pontos e pesos de gauss
        pontos_gauss, pesos_gauss = self.pontos_integracao_referencia()

        # Matriz de zeros
        k = np.zeros((self.num_nos * 2, self.num_nos * 2))

        # Integração numérica
        for pt, w in zip(pontos_gauss, pesos_gauss):
            b = bfunc(*pt)
            k += b.T @ d @ b * w * t * self.jacobiano(*pt)
        return k

# class ElementoBarra(Elemento):
#     def __init__(self, nos: np.ndarray, material: Material, area_secao: float = 1):
#         super().__init__(nos, material)
#
#         # Verfica a quantidade de elementos finitos
#         if (n := self.numero_nos()) != 2:
#             raise ErroMEF(f'Uma barra deve conter 2 nós! Nós identificados: {n}')
#
#         self.area_secao = area_secao
#
#     def comprimento(self) -> float:
#         """Retorna o comprimento da barra"""
#         return dist(*self.nos)
#
#     def matriz_rigidez(self):
#         pass
#
#     def angulo_inclinacao(self):
#         pass
#
#     def matriz_rotacao(self):
#         pass
