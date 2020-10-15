import otm
import pathlib
from otm.otimizador.oc import OC
from otm.plotagem import Plot
from otm.dados import Dados


def analisar_estrutura(dados):
    r1 = [44, 45, 83, 99, 100, 122, 246, 459, 506, 507, 510, 619, 639, 640, 658, 1780, 1782, 1783, 1784, 1901, 1902,
          1925, 2015, 2016, 2017, 2036, 2219, 2262, 2269, 2285, 2289, 2308, 5063, 5138, 5139, 5140, 5249, 5252, 5255,
          5258, 5259, 5260, 5292, 5293, 5383, 5452, 5453, 5685, 5686, 5707, 5708, 5709, 5747, 5830, 5832, 5837, 5838,
          10212, 10213, 10270, 10271, 10320, 10321, 10571, 10613, 10622, 10623, 12245, 12246, 15900, 15901, 16048,
          16049, 16054, 16058, 16059, 16071, 20813, 20884, 20885, 26992, 26993]

    r2 = [7021, 5436, 1924, 590]

    r3 = r1 + r2

    apoios = {**{i: (1, 0) for i in r3}, **{13083: (0, 1)}}
    forcas = {7021: (0, -1e2)}

    est = otm.Estrutura(dados, espessura=10, dict_cargas=forcas, dict_apoios=apoios)
    est.salvar_dados_estrutura()


concreto = otm.Concreto(2400, 20, 0.2)
aco = otm.Aco(0, 20000)
dados = Dados(pathlib.Path(__file__).parent.joinpath('MBB.zip'), concreto, aco, 1)

# Gerar malha
# malha = otm.Malha(dados, 15000)
# reg, verts = malha.criar_malha(3, 300)
# otm.GeradorMalha.exibir_malha(arq, False, 0.25)

# analisar_estrutura(dados)

rmin = 10
x_ini = 0.4
otimizador = OC(dados, fracao_volume=x_ini, p=5, rmin=rmin, tecnica_otimizacao=0)
otimizador.otimizar_estrutura(passo_p=1)

plot = Plot(dados)
# plot.plotar_malha(True)
# plot.plotar_estrutura_deformada(10)
plot.plotar_estrutura_otimizada(tecnica_otimizacao=0, corte_barras=0)
# plot.plotar_tensoes_estrutura()

# print(min(dados.deslocamentos_estrutura_original))
