import otm
import pathlib
from otm.otimizador.oc import OC
from otm.plotagem import Plot
from otm.dados import Dados


def analisar_estrutura(dados):
    r1 = [8, 60, 173, 179, 180, 268, 579, 581, 582, 589, 590, 618, 698, 699, 705, 706, 771, 772, 778, 779, 1461, 1465,
          1469, 1470, 1471, 1472, 1523, 1524, 1650, 1651, 2787, 2788, 2824, 2825, 2943, 2946, 4400, 4401, 4402, 4403,
          4414, 6017, 6018, 6019, 6020]

    r2 = [244, 4573, 1428, 4424, 209]

    r3 = r1 + r2

    apoios = {**{i: (1, 0) for i in r3}, **{2546: (0, 1)}}
    forcas = {244: (0, -1e3)}

    est = otm.Estrutura(dados, espessura=10, dict_cargas=forcas, dict_apoios=apoios)
    est.salvar_dados_estrutura()


concreto = otm.Concreto(2400, 20, 0.2)
aco = otm.Aco(0, 20000)
dados = Dados(pathlib.Path(__file__).parent.joinpath('MBB_mod.zip'), concreto, aco, 1)

# Gerar malha
# malha = otm.Malha(dados, 5000)
# reg, verts = malha.criar_malha(3, 100)
# otm.GeradorMalha.exibir_malha(arq, False, 0.25)

analisar_estrutura(dados)

rmin = 1
x_ini = 0.5
otimizador = OC(dados, fracao_volume=x_ini, p=5, rmin=rmin, tecnica_otimizacao=0)
otimizador.otimizar_estrutura(passo_p=0.5)

# plot = Plot(dados)
# plot.plotar_malha(True)
# plot.plotar_estrutura_deformada(0.2)
# plot.plotar_estrutura_otimizada(tecnica_otimizacao=0, corte_barras=0)
# plot.plotar_tensoes_estrutura()

# print(min(dados.deslocamentos_estrutura_original))
