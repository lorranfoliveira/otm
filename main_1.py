import otm
import pathlib
from otm.otimizador.oc import OC
from otm.plotagem import Plot
import numpy as np
from otm.dados import Dados


def analisar_estrutura(dados):
    r1 = [4, 34, 35, 121, 122, 215, 441, 442, 832, 1771, 2075, 2076, 2081, 2089, 2090, 2091, 2092, 2093, 3797, 3836,
          3837, 3857, 3858, 3862, 3863, 5111, 5112, 5122, 5725, 6685, 6686, 6697, 6698, 6713, 6714, 6715, 6716, 6717,
          7110, 8244, 8957, 8961, 8962, 9384, 9388, 9389]

    apoios = {i: (1, 1) for i in r1}
    forcas = {5: (0, -1)}

    est = otm.Estrutura(dados, espessura=1, dict_cargas=forcas, dict_apoios=apoios)
    est.salvar_dados_estrutura()


concreto = otm.Concreto(2.4, 0.2, 0.2)
dados = Dados(pathlib.Path(__file__).parent.joinpath('Cantilever.zip'), concreto, 1)

# Gerar malha
# malha = otm.Malha(dados, 5000)
# reg, verts = malha.criar_malha(3, 100)
# otm.GeradorMalha.exibir_malha(arq, False, 0.25)

# analisar_estrutura(dados)
# plot = Plot(dados)
# plot.plotar_estrutura_deformada(0.1)

rmin = 2
x_ini = 0.5
otimizador = OC(dados, rho_inicial=x_ini, p=5, rmin=rmin, tecnica_otimizacao=0)
otimizador.otimizar_estrutura(passo_p=1)
# plot.plotar_malha(True)
# plot.plotar_estrutura_otimizada(0)
# plot.plotar_tensoes_estrutura()
