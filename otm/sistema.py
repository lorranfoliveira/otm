from otm.mef.estrutura import Estrutura


class Otimizador:
    """Classe responsável pelo processo de otimização da estrutura"""

    def __init__(self, estrutura: Estrutura, r_min: float, volume0: float = 0.5, p: float = 3):
        """Construtor

        Args:
            estrutura: Estrutura a ser otimizada
            p: Coeficiente de penalidade
            r_min: Raio de busca do filtro
            volume0: Percentual de volume total em relação ao volume máximo de material da estrutura
        """
        pass
