import otm
import pathlib
from otm.otimizador.oc import OC
from otm.plotagem import Plot
import numpy as np
from otm.dados import Dados


def analisar_estrutura(dados):
    r1 = [58, 63, 201, 204, 431, 508, 509, 511, 514, 684, 685, 724, 725, 861, 917, 918]
    apoios = {i: (1, 1) for i in r1}
    forcas = {225: (0, -1)}

    concreto = otm.MaterialIsotropico(1, 0.25)
    est = otm.Estrutura(dados, concreto, espessura=1, dict_cargas=forcas, dict_apoios=apoios)
    est.carregar_e_salvar_dados()


dados = Dados(pathlib.Path(__file__).parent.joinpath('Teste.zip'))

# Gerar malha
# malha = otm.Malha(dados, 500)
# reg, verts = malha.criar_malha(3, 100)
# otm.GeradorMalha.exibir_malha(arq, False, 0.25)

# analisar_estrutura(dados)
# rmin = 0.25
# x_ini = 0.5
#
# Sem esquema de projeção
# otimizador = OC(dados, x_inicial=x_ini, p=5, rmin=0, tecnica_otimizacao=0)
# otimizador = OC(dados, x_inicial=x_ini, p=3, rmin=rmin, tecnica_otimizacao=4)
# otimizador = OC(x_inicial=x_ini, p=3, rmin=rmin, tecnica_otimizacao=2)
#
# otimizador.otimizar_estrutura(passo=0.5)
plot = Plot(dados)
# plot.plotar_malha()
plot.plotar_estrutura_deformada(1e-3)
