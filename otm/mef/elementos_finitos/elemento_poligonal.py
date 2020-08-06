from otm.mef.elementos_finitos.elemento_poligonal_isoparametrico import ElementoPoligonalIsoparametrico
from otm.mef.elementos_finitos.base_elemento_poligonal import BaseElementoPoligonal
import numpy as np
import shapely.geometry as geo
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from otm.mef.materiais import Material
import symengine

__all__ = ['ElementoPoligonal']


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

    def centroide(self) -> geo.Point:
        """Retorna o centroide do polígono"""
        return self.poligono().centroid

    def graus_liberdade(self) -> np.ndarray:
        """Retorna os graus de liberdade do elemento considerando todos os graus de liberdade impedidos."""
        gl = []
        for no in self.id_nos:
            gls_i = self.id_no_para_grau_liberdade(no)
            gl += gls_i
        return np.array(gl)

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

    def plotar_deslocamento(self, f, restricoes):
        """Plota as figuras da estrutura deformada"""
        # Polígono original
        poli_orig = self.nos
        poli_orig = np.concatenate((poli_orig, np.array([poli_orig[0]])))
        code = [Path.MOVETO]
        for i in range(self.num_nos):
            code.append(Path.LINETO)

        # Polígono deformado
        n = 1
        k = self.matriz_rigidez()
        k = np.delete(k, restricoes, 0)
        k = np.delete(k, restricoes, 1)

        u = np.zeros(2 * self.num_nos)
        u[np.setdiff1d(range(2 * self.num_nos), restricoes)] = np.linalg.solve(k, np.delete(f, restricoes))
        u = u.reshape(int(u.shape[0] / 2), 2)

        poli_def = poli_orig + n * np.concatenate((u, np.array([u[0]])))

        # Diâmetro equivalente e cálculo da escala das figuras
        dim_eq = self.diametro_equivalente()
        esc = 0.2 * dim_eq

        fig, ax = plt.subplots()
        ax.add_patch(patches.PathPatch(Path(poli_orig, code), linewidth=1, facecolor='orange', alpha=0.4))
        ax.add_patch(patches.PathPatch(Path(poli_def, code), linewidth=1, facecolor='green', alpha=0.4))
        ax.add_patch(patches.Arrow(poli_def[1, 0], poli_def[1, 1], 0, esc, facecolor='black', edgecolor='black',
                                   width=0.1 * esc, linewidth=1))
        ax.margins(0.05)
        ax.axis('equal')
        plt.axis('off')
        plt.grid(b=None)

        plt.show()
