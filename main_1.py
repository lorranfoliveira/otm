import otm
import pathlib
from otm.otimizador.oc import OC
from otm.plotagem import Plot
import numpy as np
from otm.dados import Dados


def analisar_estrutura(dados):
    r1 = [1, 7, 8, 178, 179, 274, 275, 278, 282, 283, 284, 285, 389, 578, 579, 580, 617, 618, 913, 914, 918, 919]

    apoios = {i: (1, 1) for i in r1}
    forcas = {1908: (0, -1)}

    est = otm.Estrutura(dados, espessura=1, dict_cargas=forcas, dict_apoios=apoios)
    est.salvar_dados_estrutura()


concreto = otm.Concreto(2.4, 20, 0.2)
dados = Dados(pathlib.Path(__file__).parent.joinpath('Cantilever.zip'), concreto, 1)

# Gerar malha
# malha = otm.Malha(dados, 1000)
# reg, verts = malha.criar_malha(3, 100)
# otm.GeradorMalha.exibir_malha(arq, False, 0.25)

# analisar_estrutura(dados)
plot = Plot(dados)
# plot.plotar_estrutura_deformada(0.1)

rmin = 2
x_ini = 0.5
otimizador = OC(dados, rho_inicial=x_ini, p=5, rmin=rmin, tecnica_otimizacao=0)
otimizador.otimizar_estrutura(passo_p=0.5)
# plot.plotar_malha(True)
plot.plotar_estrutura_otimizada(0)
plot.plotar_tensoes_estrutura()
