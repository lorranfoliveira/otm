import otm
import pathlib
from otm.otimizador.oc import OC
from otm.plotagem import Plot
import numpy as np
from otm.dados import Dados


def analisar_estrutura(dados):
    r1 = [0, 3, 49, 150, 151, 291, 292, 774, 775, 795, 1708, 1821, 1822, 1833, 1834, 1835, 2068, 2069, 3472, 3871, 3872,
          3890, 5588, 5595, 5599, 5600, 5601, 5602, 5609, 5629, 5631, 5802, 5803, 5814, 5815, 7162, 7376, 7377, 7378,
          7379, 7380, 8216, 8220, 8221, 8360, 8361, 8362, 8363, 9088, 9511, 9512]

    apoios = {i: (1, 1) for i in r1}
    forcas = {7924: (0, -1)}

    est = otm.Estrutura(dados, espessura=1, dict_cargas=forcas, dict_apoios=apoios)
    est.salvar_dados_estrutura()


concreto = otm.Concreto(2.4, 0.2, 0.2)
dados = Dados(pathlib.Path(__file__).parent.joinpath('Cantilever.zip'), concreto, 1)

# Gerar malha
# malha = otm.Malha(dados, 5000)
# reg, verts = malha.criar_malha(3, 100)
# otm.GeradorMalha.exibir_malha(arq, False, 0.25)

# analisar_estrutura(dados)
plot = Plot(dados)
# plot.plotar_estrutura_deformada(0.1)

rmin = 2
x_ini = 0.3
otimizador = OC(dados, rho_inicial=x_ini, p=5, rmin=rmin, tecnica_otimizacao=0)
otimizador.otimizar_estrutura(passo_p=0.5)
# plot.plotar_malha(True)
# plot.plotar_estrutura_otimizada(0)
