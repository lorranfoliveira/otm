import otm
import pathlib
from otm.otimizador.oc import OC
from otm.plotagem import Plot
import numpy as np
from otm.dados import Dados


def analisar_estrutura(dados):
    r1 = [4, 36, 37, 41, 44, 180, 181, 195, 196, 197, 822, 823, 827, 828, 846, 926, 2200, 2201, 2212, 2222, 2274, 2275,
          2301, 3817, 3820, 3935, 3936, 5699, 5700, 5741, 5742, 5854, 5855, 5856, 5857, 5900, 5901, 7292, 7293, 7297,
          7298, 7446, 7447, 8391, 8395, 8396, 9078, 9403, 9404]

    apoios = {i: (1, 1) for i in r1}
    forcas = {424: (0, -1)}

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

# rmin = 2
# x_ini = 0.5
# otimizador = OC(dados, rho_inicial=x_ini, p=5, rmin=rmin, tecnica_otimizacao=0)
# otimizador.otimizar_estrutura(passo_p=1)
# plot.plotar_malha(True)
plot.plotar_estrutura_otimizada(0)
# plot.plotar_tensoes_estrutura()
