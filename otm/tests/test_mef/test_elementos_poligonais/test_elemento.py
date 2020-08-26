import unittest
import numpy as np
from otm.mef.elementos.base_elemento_poligonal import BaseElementoPoligonal
from otm import ElementoPoligonalIsoparametrico
from shapely.geometry import Polygon


class TestElemento(unittest.TestCase):

    def test_matriz_constitutiva(self):
        dados = {(1, 0.3): np.array([[1.0989010989010988, 0.3296703296703296, 0.0],
                                     [0.3296703296703296, 1.0989010989010988, 0.0],
                                     [0.0, 0.0, 0.3846153846153845]]),
                 (200e9, 0.2): np.array([[208333333333.33334, 41666666666.66667, 0.0],
                                         [41666666666.66667, 208333333333.33334, 0.0],
                                         [0.0, 0.0, 83333333333.33334]]),
                 (23e9, 0.25): np.array([[24533333333.333332, 6133333333.333333, 0.0],
                                         [6133333333.333333, 24533333333.333332, 0.0],
                                         [0.0, 0.0, 9200000000.0]])}

        for d in dados:
            self.assertTrue(np.allclose(BaseElementoPoligonal.matriz_constitutiva(*d), dados[d]))

    def test_num_lados(self):
        for n in range(3, 10):
            el_iso = ElementoPoligonalIsoparametrico(n)
            nos = el_iso.coordenadas_nos_elemento_isoparametrico()
            self.assertEqual(BaseElementoPoligonal(nos).num_nos, n)

    def test_poligono(self):
        for n in range(3, 10):
            el_iso = ElementoPoligonalIsoparametrico(n)
            nos = el_iso.coordenadas_nos_elemento_isoparametrico()
            el = BaseElementoPoligonal(nos)
            nos_calc = np.array(el.poligono().boundary.coords)
            self.assertTrue(np.allclose(nos_calc, np.concatenate((nos, np.array([nos[0]])))))

    def test_centroide(self):
        for n in range(3, 10):
            el_iso = ElementoPoligonalIsoparametrico(n)
            nos = el_iso.coordenadas_nos_elemento_isoparametrico()
            el = BaseElementoPoligonal(nos)

            if n == 3:
                c = np.array([0.3333333333333333, 0.3333333333333333])
            else:
                c = np.zeros(2)

            self.assertTrue(np.allclose(el.centroide(), c))

    def test_area(self):
        for n in range(3, 10):
            el_iso = ElementoPoligonalIsoparametrico(n)
            nos = el_iso.coordenadas_nos_elemento_isoparametrico()
            el = BaseElementoPoligonal(nos)
            poli = Polygon(nos)

            self.assertAlmostEqual(el.area(), poli.area)

    def test_diametro_equivalente(self):
        dados = {3: 0.7978845608028654,
                 4: 2.256758334191025,
                 5: 1.7399157780084151,
                 6: 1.818783486985395}

        for d in dados:
            el_iso = ElementoPoligonalIsoparametrico(d)
            nos = el_iso.coordenadas_nos_elemento_isoparametrico()
            el = BaseElementoPoligonal(nos)

            self.assertAlmostEqual(el.diametro_equivalente(), dados[d])

    def test_triangular_poligono(self):
        resultado = np.array([[[0.41041252791061605, 2.065226971950941], [8.0, 10.0], [-1.0, 12.0]],
                              [[0.41041252791061605, 2.065226971950941], [-1.0, 12.0], [-7.4853, 7.6569]],
                              [[0.41041252791061605, 2.065226971950941], [-7.4853, 7.6569], [-11.0, 2.0]],
                              [[0.41041252791061605, 2.065226971950941], [-11.0, 2.0], [-4.6569, -6.4853]],
                              [[0.41041252791061605, 2.065226971950941], [-4.6569, -6.4853], [3.0, -8.0]],
                              [[0.41041252791061605, 2.065226971950941], [3.0, -8.0], [8.0, -3.0]],
                              [[0.41041252791061605, 2.065226971950941], [8.0, -3.0], [11.0, -4.0]],
                              [[0.41041252791061605, 2.065226971950941], [11.0, -4.0], [8.0, 10.0]]])

        nos = np.array([[8, 10],
                        [-1, 12],
                        [-7.4853, 7.6569],
                        [-11, 2],
                        [-4.6569, -6.4853],
                        [3, -8],
                        [8, -3],
                        [11, -4]])

        el = BaseElementoPoligonal(nos)
        self.assertTrue(np.allclose(np.array(el.triangular_poligono()), resultado))
