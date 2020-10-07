import numpy as np
from otm.mef.materiais import Material, Concreto, Aco
import shapely.geometry as sh_geo
from typing import List, Optional, Union
import symengine
from loguru import logger

__all__ = ['Elemento', 'BaseElementoPoligonal', 'ElementoPoligonalIsoparametrico', 'ElementoPoligonal', 'ElementoBarra']


class Elemento:
    """Classe abstrata que implementa as propriedades em comum dos elementos finitos."""

    def __init__(self, nos: np.ndarray, material: Optional[Union[Concreto, Aco]]):
        """Construtor.

        Args:
            nos: Matriz que contém os nós que formam o elemento.
            material: Material que compõe o elemento.
        """
        self.nos = nos
        self.material = material

    @property
    def num_nos(self) -> int:
        """Retorna o número de nós que compõem o elemento."""
        return self.nos.shape[0]

    def matriz_rigidez(self) -> np.ndarray:
        pass

    @staticmethod
    def id_no_para_grau_liberdade(id_no) -> list:
        """Calcula os graus de liberdade do elemento em função da numeração de um nó."""
        return [2 * id_no, 2 * id_no + 1]

    def graus_liberdade(self):
        pass


class BaseElementoPoligonal(Elemento):
    """Classe abstrata que implementa as propriedades comuns dos elementos finitos poligonais físicos e
    isoparamétricos."""
    # Os bancos armazenam dados que foram gerados para serem acessados por outras partes do código em um
    # tempo de processamento mais curto. A chave dos dicionários consistem no número de lados que
    # o elemento poligonal possui e o valor nas funções/valores referentes àquele polígono.
    BANCO_FUNCOES_FORMA = {}
    BANCO_DIFF_FUNCOES_FORMA = {}
    BANCO_PONTOS_PESOS_GAUSS = {}

    # Dados para a integração numérica de um domínio triangular isoparamétrico pela quadratura gaussiana.
    PESOS_INTEGRACAO_TRI_GAUSS = np.full(3, 1 / 3)
    PONTOS_INTEGRACAO_TRI_GAUSS = np.array([[1 / 6, 1 / 6],
                                            [2 / 3, 1 / 6],
                                            [1 / 6, 2 / 3]])

    def __init__(self, nos: np.ndarray, material: Optional[Material]):
        super().__init__(nos, material)

    def poligono(self) -> sh_geo.Polygon:
        """Retorna o polígono do Shapely que representa o elemento."""
        return sh_geo.Polygon(self.nos)

    def centroide(self) -> np.ndarray:
        """Retorna o centroide do polígono"""
        p = self.poligono().centroid
        return np.array([p.x, p.y])

    def area(self) -> float:
        """Retorna a área do elemento"""
        return self.poligono().area

    def triangular_poligono(self) -> List[np.ndarray]:
        """Discretiza o elemento poligonal em triângulos. Considera-se o centroide do elemento como
        ponto em comum a todos os triângulos.

        Returns:
            Retorna uma lista com as coordenadas de cada triângulo.
        """
        # O número de triângulos é igual ao número de lados do polígono.
        triangulos = []
        # Replicação do primeiro nó, adicionando-se uma nova linha na matriz, para valer a lógica abaixo.
        nos = np.concatenate((self.nos, np.array([self.nos[0]])))
        c = self.centroide()

        for v in range(self.num_nos):
            triangulos.append(np.array([c, nos[v], nos[v + 1]]))
        return triangulos


class ElementoPoligonalIsoparametrico(BaseElementoPoligonal):
    """Classe que implementa as propriedades de um polígono regular de 'n' lados inscrito em uma
    circunferência de raio 1 e origem em (0, 0)."""

    def __init__(self, num_lados: int):
        """Construtor.

        Args:
            num_lados: Número de lados do elemento finito poligonal.
        """
        self._num_lados = num_lados

        super().__init__(self.coordenadas_vertices(), None)

        # Variáveis simbólicas necessárias para o cálculo das funções de forma.
        self._x = symengine.var('x')
        self._y = symengine.var('y')

    def coordenadas_vertices(self) -> np.ndarray:
        """Retorna as coordenadas dos nós do elemento isoparamétrico."""
        n_lados = self._num_lados
        # Coordenadas dos pontos do elemento de referência
        return np.array([[np.cos(2 * np.pi * i / n_lados), np.sin(2 * np.pi * i / n_lados)]
                         for i in range(1, n_lados + 1)])

    def funcoes_forma(self) -> list:
        """Retorna as funções isoparamétricas de um elemento poligonal com n lados, com n >= 3.
        A chave do dicionário é composta pelo número de lados do polígono e seu valor consiste
        na função de forma como uma expressão do symengine."""

        def func_symengine():
            """Função interna que calcula as funções de forma do elemento."""
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

        if (nlados := self.num_nos) not in self.BANCO_FUNCOES_FORMA:
            self.BANCO_FUNCOES_FORMA[nlados] = func_symengine()

        return self.BANCO_FUNCOES_FORMA[nlados]

    def diff_funcoes_forma(self):
        """Retorna uma matriz com as derivadas das funções de forma em relação a x (linha 0) e a y (linha 1).
        As funções retornadas são funções `lambda`."""

        def diff_func():
            func_forma = self.funcoes_forma()
            return symengine.Matrix([[f.diff(v) for f in func_forma] for v in [self._x, self._y]])

        if (nlados := self.num_nos) not in self.BANCO_DIFF_FUNCOES_FORMA:
            func = diff_func()
            self.BANCO_DIFF_FUNCOES_FORMA[nlados] = symengine.lambdify(list(func.free_symbols), func)

        return self.BANCO_DIFF_FUNCOES_FORMA[nlados]

    @staticmethod
    def funcoes_forma_triangulo(x, y) -> np.ndarray:
        """Funções de forma referentes a um elemento triangular isoparamétrico."""
        return np.array([1 - x - y, x, y])

    @staticmethod
    def diff_funcoes_forma_triangulo() -> np.ndarray:
        """# Derivadas das funções de forma de um triângulo isoparamétrico em relação a `x` (linha 0),
        em relação a `y` (linha 1)."""
        return np.array([[-1, 1, 0],
                         [-1, 0, 1]])

    def pontos_de_integracao(self) -> List[np.ndarray]:
        """Calcula os pontos de integração e os pesos para cada um a partir da discretização do
        elemento em triângulos.

        Returns:
            Retorna uma lista com dois índices. O índice 0 é uma matriz com os pontos de integração.
            O índice 1 é um vetor com os pesos.
        """
        if not (self.num_nos in self.BANCO_PONTOS_PESOS_GAUSS):
            n = self.num_nos
            pts_gauss = self.PONTOS_INTEGRACAO_TRI_GAUSS
            w = self.PESOS_INTEGRACAO_TRI_GAUSS

            # A quantidade de triângulos é numericamente igual à de lados do polígono (n)
            triangulos = self.triangular_poligono()

            # Cada triângulo possui 3 pontos de integração
            pontos = np.zeros((n * len(w), 2))
            pesos = np.zeros(n * len(w))

            for i in range(n):
                for no in range(c := len(w)):
                    func_forma = self.funcoes_forma_triangulo(*pts_gauss[no])
                    diff_func_forma = self.diff_funcoes_forma_triangulo()
                    # Matriz jacobiana
                    j0 = triangulos[i].T @ diff_func_forma.T
                    # Conversão de coordenadas do triângulo isoparamétricas para coordenadas
                    # cartesianas do polígono isoparamétrico
                    k = i * c + no
                    pontos[k, :] = func_forma.T @ triangulos[i]
                    # Valor da função no ponto de gauss considerando o peso
                    pesos[k] = 1 / 2 * np.linalg.det(j0) * w[no]

                    self.BANCO_PONTOS_PESOS_GAUSS[self.num_nos] = [pontos, pesos]

        return self.BANCO_PONTOS_PESOS_GAUSS[self.num_nos]


class ElementoPoligonal(BaseElementoPoligonal):
    """Classe que define as propriedades dos elementos finitos poligonais."""

    def __init__(self, nos: np.ndarray, material: Material, espessura: float = 1, id_nos=None):
        super().__init__(nos, material)
        self.espessura = espessura
        self.id_nos = id_nos

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
        """Retorna uma matriz de derivadas das funções de forma do elemento de referência. As funções
        retornadas são funções `lambda`. Caso o elemento de referência já tenha sido utilizado, os
        valores correspondentes serão apenas buscados em um banco de dados."""
        if (n := self.num_nos) not in self.BANCO_DIFF_FUNCOES_FORMA:
            el_ref = self.elemento_referencia()
            return el_ref.diff_funcoes_forma()
        else:
            return self.BANCO_DIFF_FUNCOES_FORMA[n]

    def pontos_integracao_referencia(self):
        """Retorna os pontos de integração do elemento de referência. Caso o elemento de referência já
        tenha sido utilizado, os valores correspondentes serão apenas buscados em um banco de dados."""
        if (n := self.num_nos) not in self.BANCO_PONTOS_PESOS_GAUSS:
            el_ref = self.elemento_referencia()
            return el_ref.pontos_de_integracao()
        else:
            return self.BANCO_PONTOS_PESOS_GAUSS[n]

    def matriz_jacobiana(self, x, y):
        """Retorna a matriz jacobiana do elemento para as coordenadas numéricas (x, y)."""
        df = np.array(self.diff_funcoes_forma_referencia()(x, y)).reshape(2, self.num_nos)
        return df @ self.nos

    def jacobiano(self, x, y) -> float:
        """Retorna o determinante da matriz jacobiana em um ponto de coordenadas (x, y)."""
        return np.linalg.det(self.matriz_jacobiana(x, y))

    def matriz_b(self, x, y) -> np.ndarray:
        """Retorna a matriz de compatibilidade cinemática nodal para as coordenadas (x, y)."""
        n = self.num_nos

        df = np.array(self.diff_funcoes_forma_referencia()(x, y)).reshape(2, n)

        h = np.linalg.inv(self.matriz_jacobiana(x, y))
        num_lin_h, num_cols_h = h.shape

        b1 = np.array([[1, 0, 0, 0],
                       [0, 0, 0, 1],
                       [0, 1, 1, 0]])

        b2 = np.zeros((2 * num_lin_h, 2 * num_cols_h))
        b2[np.ix_(range(num_lin_h), range(num_cols_h))] = h
        b2[np.ix_(range(num_lin_h, b2.shape[0]), range(num_cols_h, b2.shape[1]))] = h

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
        """Retorna a matriz de rigidez do elemento."""
        t = self.espessura
        b_func = self.matriz_b

        d = self.material.matriz_constitutiva_isotropico()

        # Pontos e pesos de gauss
        pontos_gauss, pesos_gauss = self.pontos_integracao_referencia()

        # Matriz de zeros
        k = np.zeros((self.num_nos * 2, self.num_nos * 2))

        # Integração numérica
        for pt, w in zip(pontos_gauss, pesos_gauss):
            b = b_func(*pt)
            k += b.T @ d @ b * w * t * self.jacobiano(*pt)
        return k

    def matriz_b_origem(self) -> np.ndarray:
        """Retorna a matriz B calculada na origem (0, 0) do elemento isoparamétrico."""
        return self.matriz_b(0, 0)

    def matrizes_b_pontos_integracao(self) -> List[np.ndarray]:
        """Retorna as matrizes b calculadas em cada um dos pontos de integração do elemento.
        Retorna também os pesos de integração multiplicados pela espessura e pelo jacobiano.
        [b_i, (w_i * t * jacobiano_i)]
        """
        t = self.espessura
        b_func = self.matriz_b
        pontos_gauss, pesos_gauss = self.pontos_integracao_referencia()
        dados = []

        # Integração numérica
        for pt, w in zip(pontos_gauss, pesos_gauss):
            b = b_func(*pt)
            dados.append([b, w * t * self.jacobiano(*pt)])
        return dados


class ElementoBarra(Elemento):
    """Elemento finito de barra com área da seção transversal unitária."""

    def __init__(self, nos: np.ndarray, material: Aco, id_nos=None):
        super().__init__(nos, material)
        self.id_nos = id_nos

    def graus_liberdade(self) -> np.ndarray:
        """Retorna os graus de liberdade do elemento considerando todos os graus de liberdade impedidos."""
        gl = []
        for no in self.id_nos:
            gls_i = self.id_no_para_grau_liberdade(no)
            gl += gls_i
        return np.array(gl, dtype=int)

    def comprimento(self) -> float:
        """Retorna o comprimento da barra."""
        return np.linalg.norm(self.nos[1] - self.nos[0])

    def matriz_rigidez_local(self) -> np.ndarray:
        """Matrizes de rigidez dos elementos considerando a área da seção e o módulo de elasticidade unitários."""
        c = self.comprimento()
        ke = np.zeros((4, 4))
        ke[0, 0] = ke[2, 2] = 1 / c
        ke[2, 0] = ke[0, 2] = -1 / c

        return ke

    def matriz_rotacao(self) -> np.ndarray:
        """Retorna a matriz de rotação do elemento."""
        t = self.angulo_inclinacao()
        return np.array([[np.cos(t), np.sin(t), 0, 0],
                         [-np.sin(t), np.cos(t), 0, 0],
                         [0, 0, np.cos(t), np.sin(t)],
                         [0, 0, -np.sin(t), np.cos(t)]])

    def angulo_inclinacao(self) -> np.ndarray:
        """Retorna o ângulo de rotação do elemento."""
        dx = self.nos[1, 0] - self.nos[0, 0]
        dy = self.nos[1, 1] - self.nos[0, 1]

        if dx == 0:
            if dy < 0:
                termo = -np.inf
            else:
                termo = np.inf
        else:
            termo = dy / dx

        return np.arctan(termo)

    def matriz_rigidez(self) -> np.ndarray:
        r = self.matriz_rotacao()
        ke = self.matriz_rigidez_local()
        return r @ ke @ r
