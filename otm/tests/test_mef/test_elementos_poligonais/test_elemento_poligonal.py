import unittest
from otm import ElementoPoligonal, MaterialIsotropico
import otm.tests.test_mef.test_elementos_poligonais.dados_testes_elementos as dados
import numpy as np


class TestElementoPoligonal(unittest.TestCase):
    def test_matriz_rigidez(self):
        material = MaterialIsotropico(1, 0.2)
        for e in dados.elementos:
            el = ElementoPoligonal(dados.elementos[e], material)
            self.assertTrue(np.allclose(el.matriz_rigidez(), dados.ke[e]))
