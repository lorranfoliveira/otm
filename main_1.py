import otm
import pathlib
from otm.otimizador.oc import OC
from otm.plotagem import Plot
from otm.dados import Dados


def analisar_estrutura(dados):
    r1 = [53, 54, 56, 175, 176, 181, 182, 192, 193, 438, 439, 440, 464, 910, 911, 914, 925, 926, 930, 931, 994, 1858,
          1870, 1875, 1876, 2082, 2084, 3366, 3367, 3562, 5065, 5066, 5321, 5322, 5326, 5327, 5361, 5362, 6839, 8999,
          9010, 9011, 9022, 9466, 9470, 9471]

    apoios = {**{i: (1, 0) for i in r1}, **{1: (0, 1)}}
    forcas = {994: (0, -1)}

    est = otm.Estrutura(dados, espessura=1, dict_cargas=forcas, dict_apoios=apoios)
    est.salvar_dados_estrutura()


concreto = otm.Concreto(2.4, 0.2, 0.2)
aco = otm.Aco(0, 20)
dados = Dados(pathlib.Path(__file__).parent.joinpath('MBB.zip'), concreto, aco, 1)

# Gerar malha
# malha = otm.Malha(dados, 5000)
# reg, verts = malha.criar_malha(3, 300)
# otm.GeradorMalha.exibir_malha(arq, False, 0.25)

# analisar_estrutura(dados)

rmin = 1
x_ini = 0.4
otimizador = OC(dados, fracao_volume=x_ini, p=3, rmin=rmin, tecnica_otimizacao=0)
otimizador.otimizar_estrutura(passo_p=1)

plot = Plot(dados)
# plot.plotar_malha()
# plot.plotar_estrutura_deformada(0.5)
plot.plotar_estrutura_otimizada(tecnica_otimizacao=0, corte_barras=0.5)
# plot.plotar_tensoes_estrutura()
