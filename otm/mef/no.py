from shapely.geometry import Point
import numpy as np
from typing import List


class No(Point):
    """Define as propriedades de um nó geométrico."""
    NUM_GRAUS_LIB_NO = 2

    def __init__(self, x: float, y: float, idt: int = None, cargas: List[float] = None, apoios: List[int] = None):
        """Construtor.

        Args:
            x: Coordenada x.
            y: Coordenada y.
            idt: Identificação
            cargas: Vetor de cargas [fx, fy].
            apoios: Apoios do nó [ux, uy]. Se for 0 é livre, se for 1 é impedido.
        """
        super().__init__(x, y)
        self.idt = idt
        self.cargas = cargas
        self.apoios = apoios

    def __repr__(self):
        return f'{type(self).__name__}(idt={self.idt}, x={self.x}, y={self.y}, cargas={self.cargas}, ' \
               f'apoios={self.apoios})'

    @property
    def cargas(self) -> np.ndarray:
        if self._cargas is None:
            return np.zeros(2)
        else:
            return self._cargas

    @cargas.setter
    def cargas(self, value):
        if value is None:
            self._cargas = None
        else:
            n = 0
            try:
                n = len(value)
            except TypeError:
                TypeError(f'As cargas do nó {self.idt} devem estar em um vetor iterável!')

            if not n == 2:
                raise ValueError(f'O vetor de cargas do nó {self.idt} deve possuir apenas 2 elementos!')
            else:
                self._cargas = np.array(value)

    @property
    def apoios(self) -> np.ndarray:
        if self._apoios is None:
            return np.zeros(2, dtype=int)
        else:
            return self._apoios

    @apoios.setter
    def apoios(self, value):
        if value is None:
            self._apoios = None
        else:
            n = 0
            try:
                n = len(value)
            except TypeError:
                TypeError(f'Os apoios do nó {self.idt} devem estar em um vetor iterável!')

            if not n == 2:
                raise ValueError(f'O vetor de apoios do nó {self.idt} deve possuir apenas 2 elementos!')
            if not all(i == 0 or i == 1 for i in value):
                raise ValueError(f'Os elementos do vetor de apoios do nó {self.idt} devem assumir valores '
                                 f'0 ou 1 apenas!')
            else:
                self._apoios = np.array(value, dtype=int)

    def graus_lib(self) -> np.ndarray:
        """Retorna um vetor com os graus de liberdade do nó."""
        return np.array([self.idt * self.NUM_GRAUS_LIB_NO - 1, self.idt * self.NUM_GRAUS_LIB_NO])

    def graus_lib_livres(self) -> np.ndarray:
        gl = self.graus_lib()
        gllivres = np.zeros(len(np.nonzero(self.apoios[0])))
        c = 0
        for i, ap in enumerate(self.apoios):
            if ap == 0:
                gllivres[c] = gl[i]
                c += 1

        return gllivres

    def num_graus_lib(self) -> int:
        return len(self.graus_lib())

    def apoiado(self) -> bool:
        return any(i == 1 for i in self.apoios)

    def carregado(self) -> bool:
        return any(i != 0 for i in self.cargas)
