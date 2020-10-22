import otm
import pathlib
from otm.otimizador.oc import OC
from otm.plotagem import Plot
from otm.dados import Dados


def analisar_estrutura(dados):
    r1 = [432, 435, 1219, 1220, 1231, 1232, 1233, 1248, 1259, 1274, 1275, 1283, 1422, 1423, 2765, 2767, 2771, 2772,
          2794, 2795, 2806, 2807, 2825, 2848, 2849, 2853, 2854, 2992, 2993, 5287, 5294, 5296, 5297, 5299, 5300, 5400,
          5411, 5412, 5439, 5440, 5583, 5584, 8585, 8588, 8607, 8608, 8617, 8618, 8619, 8635, 8636, 8662, 8663, 8692,
          8693, 11716, 11717, 11724, 11725, 11726, 12049, 14942, 16943, 18314, 18315, 18406, 18407]

    r2 = [8411, 2859, 5410, 3019]

    r3 = r1 + r2

    apoios = {**{i: (1, 0) for i in r3}, **{29: (0, 1)}}
    forcas = {8411: (0, -100)}

    est = otm.Estrutura(dados, espessura=1, dict_cargas=forcas, dict_apoios=apoios)
    est.salvar_dados_estrutura()


concreto = otm.Concreto(2490, 200, 0.2)
aco = otm.Aco(0, 20000)
dados = Dados(pathlib.Path(__file__).parent.joinpath('MBB_kenia_04.zip'), concreto, aco, 1)

# Gerar malha
# malha = otm.Malha(dados, 10000)
# reg, verts = malha.criar_malha(3, 300)

# analisar_estrutura(dados)
#
rmin = 10
fv = 0.4
otimizador = OC(dados, fracao_volume=fv, p=5, rmin=rmin, tecnica_otimizacao=4)
otimizador.otimizar_estrutura(passo_p=1)
#
plot = Plot(dados)
# plot.plotar_malha(True)
# plot.plotar_estrutura_deformada(30)
plot.plotar_estrutura_otimizada(tecnica_otimizacao=0, corte_barras=0)
# plot.plotar_tensoes_estrutura()

# print(min(dados.deslocamentos_estrutura_original))
