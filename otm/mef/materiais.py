import numpy as np

__all__ = ['Material', 'Concreto']

E_CONCRETO_COMPRIMIDO = 2.49
E_CONCRETO_TRACIONADO = 2
NU_CONCRETO_COMPRIMIDO = 0.2
E_ACO_COMPRIMIDO = 0
E_ACO_TRACIONADO = 20


def matriz_constitutiva_isotropico(e: float, nu: float):
    """Retorna a matriz constitutiva de um material isotrópico.

    Args:
        e: Módulo de elasticidade.
        nu: Coeficiente de Poisson.
    """
    return (e / (1 - nu ** 2)) * np.array([[1, nu, 0],
                                           [nu, 1, 0],
                                           [0, 0, (1 - nu) / 2]])


def matriz_constitutiva_ortotropico() -> np.ndarray:
    """Os índices das variáveis se referem às direções principais 1 e 2."""
    e_1 = E_CONCRETO_COMPRIMIDO
    e_2 = E_CONCRETO_TRACIONADO
    nu_1 = NU_CONCRETO_COMPRIMIDO

    nu_2 = nu_1 * e_2 / e_1
    nu_ef = np.sqrt(nu_1 * nu_2)
    e_12 = nu_ef * np.sqrt(e_1 * e_2)

    return np.array([[e_1, e_12, 0],
                     [e_12, e_2, 0],
                     [0, 0, (1 / 4) * (e_1 + e_2 - 2 * e_12)]])


# Matriz constitutiva do concreto ortotrópico nas direções principais.
D_CONCRETO_ORTOTROPICO = matriz_constitutiva_ortotropico()


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


class Concreto(Material):
    """Classe que descreve o concreto como um material ortotrópico segundo a teoria de Darwin e Pecknold."""

    def __init__(self, ec: float, et: float, nu: float):
        super().__init__(ec, et, nu)

    def matriz_rotacao(self, ex: float, ey: float, exy: float):
        """Retorna a matriz de rotação da matriz constitutiva elástica ortotrópica.

        Args:
            ex: Tensão normal na direção x.
            ey: Tensão normal na direção y.
            exy: Tensão cisalhante
        """
        t = self.angulo_rotacao(ex, ey, exy)
        return np.array([[np.cos(t) ** 2, np.sin(t) ** 2, 2 * np.cos(t) * np.sin(t)],
                         [np.sin(t) ** 2, np.cos(t) ** 2, -2 * np.cos(t) * np.sin(t)],
                         [-np.cos(t) * np.sin(t), np.cos(t) * np.sin(t), np.cos(t) ** 2 - np.sin(t) ** 2]])

    @staticmethod
    def angulo_rotacao(ex, ey, exy) -> float:
        """Retorna o ângulo de inclinação das sx e sy em relação ao eixo das tensões principais."""
        return np.arctan(exy / (ex - ey)) / 2

    def matriz_constitutiva(self) -> np.ndarray:
        """Retorna a matriz constitutiva elástica nas direções dos eixos x e y."""
        e_1 = self.ec
        e_2 = self.et
        nu_1 = self.nu

        nu_2 = nu_1 * e_2 / e_1
        nu_ef = np.sqrt(nu_1 * nu_2)
        e_12 = nu_ef * np.sqrt(e_1 * e_2)

        return (1 / (1 - nu_ef ** 2)) * np.array([[e_1, e_12, 0],
                                                  [e_12, e_2, 0],
                                                  [0, 0, 0.25 * (e_1 + e_2 - 2 * e_12)]])

    def matriz_constitutiva_ortotropica(self, s1, s2) -> np.ndarray:
        """Retorna a matriz constitutiva elástica nas direções dos eixos x e y.

        Args:
            s1: Tensão principal 1.
            s2: Tensão principal 2.
        """
        # Tensões principais no elemento.
        nu_t = self.nu * self.et / self.ec

        if s1 < 0:
            e_1 = self.ec
            nu_1 = self.nu
        else:
            e_1 = self.et
            nu_1 = nu_t

        if s2 < 0:
            e_2 = self.ec
            nu_2 = self.nu
        else:
            e_2 = self.et
            nu_2 = nu_t

        nu_ef = np.sqrt(nu_1 * nu_2)
        e_12 = nu_ef * np.sqrt(e_1 * e_2)

        return (1 / (1 - nu_ef ** 2)) * np.array([[e_1, e_12, 0],
                                                  [e_12, e_2, 0],
                                                  [0, 0, 0.25 * (e_1 + e_2 - 2 * e_12)]])

    def matriz_constitutiva_ortotropica_rotacionada(self, tensoes, deformacoes) -> np.ndarray:
        """Retorna a matriz constitutiva elástica rotacionada para os eixos principais."""
        sx, sy, txy = tensoes
        ex, ey, gxy = deformacoes
        r = self.matriz_rotacao(ex, ey, gxy)
        d = self.matriz_constitutiva_ortotropica(*self.tensoes_principais_elemento(sx, sy, txy))

        return r.T @ d @ r

    def matriz_constitutiva_isotropico(self) -> np.ndarray:
        """Retorna a matriz constitutiva elástica para um material isotrópico. Utiliza-se o módulo de
        elasticidade do concreto à compressão."""
        e0 = self.ec
        nu = self.nu
        return (e0 / (1 - nu ** 2)) * np.array([[1, nu, 0],
                                                [nu, 1, 0],
                                                [0, 0, (1 - nu) / 2]])

    @staticmethod
    def tensoes_principais_elemento(sx, sy, txy) -> np.ndarray:
        """Retorna as tensões principais de um elemento."""
        p1 = (sx + sy) / 2
        p2 = np.sqrt(((sx + sy) / 2) ** 2 + txy ** 2)
        s1 = p1 + p2
        s2 = p1 - p2

        return np.array([s1, s2])

    @staticmethod
    def maior_tensao_no_elemento(sx, sy, txy) -> float:
        # Tensões principais
        s1, s2 = Concreto.tensoes_principais_elemento(sx, sy, txy)
        return s1 if abs(s1) >= abs(s2) else s2
