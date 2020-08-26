import numpy as np
import symengine
from loguru import logger
from otm.mef.elementos.base_elemento_poligonal import BaseElementoPoligonal

__all__ = ['ElementoPoligonalIsoparametrico']


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

