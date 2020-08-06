import unittest
import pathlib
import otm
from typing import List


class TestEstrutura(unittest.TestCase):
    path = pathlib.Path(__file__).parent.parent.joinpath('exemplos_testes')
    # Dados das estruturas a serem testadas
    path_estruturas = {0: path.joinpath('modelo_referencia_1'),
                       1: path.joinpath('modelo_referencia_2')}

    forcas = {0: {6: (1, 0), 7: (1, 0)},
              1: {765: [0, -2e6]}}

    apoios = {0: {0: (1, 0), 1: (1, 1)},
              1: {6: (1, 1), 34: (1, 1), 35: (1, 1), 36: (1, 1), 129: (1, 1), 132: (1, 1), 356: (1, 1), 502: (1, 1),
                  509: (1, 1), 663: (1, 1), 776: (1, 1), 777: (1, 1)}}

    espessuras = {0: 0.1,
                  1: 0.16}

    # {id: [mod_elasticidade, poisson, espessura]}
    dados_materiais = {0: [25e5, 0.3],
                       1: [16.22e9, 0.3]}

    def materiais(self) -> List[otm.MaterialIsotropico]:
        """Retorna os materiais definidos pelos dados em 'dados_materiais'"""
        mats = []
        for i in self.dados_materiais:
            mats.append(otm.MaterialIsotropico(*self.dados_materiais[i]))
        return mats

    def estruturas(self) -> List[otm.Estrutura]:
        """Retorna as estruturas definidas pelos dados anteriores"""
        materiais = self.materiais()
        estruts = []

        for i in self.path_estruturas:
            est = otm.Estrutura(self.path_estruturas[i], materiais[i], self.espessuras[i], self.forcas[i],
                                self.apoios[i])
            estruts.append(est)
        return estruts

    def test_leitura_dados_arquivo(self):
        pass