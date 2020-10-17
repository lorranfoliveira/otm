import otm
import pathlib
from otm.otimizador.oc import OC
from otm.plotagem import Plot
from otm.dados import Dados


def analisar_estrutura(dados):
    r1 = [367, 1154, 1174, 1185, 2610, 2611, 2621, 2622, 3079, 3145, 3161, 4601, 4602, 4625, 4627, 5152, 5153, 5154,
          5158, 5159, 5162, 5163, 6601, 6605, 6606, 6609, 6614, 6615, 6961, 6962, 7016, 7017, 8174, 8175, 8179, 8180,
          8181, 8182, 8475, 8491, 9151, 9192, 9193, 9338, 9339, 9348, 9699, 9703, 9704, 9801, 9802]

    r2 = [8455, 5143, 2650, 372]

    r3 = r1 + r2

    apoios = {**{i: (1, 0) for i in r3}, **{1709: (0, 1)}}
    forcas = {8455: (0, -1e2)}

    est = otm.Estrutura(dados, espessura=10, dict_cargas=forcas, dict_apoios=apoios)
    est.salvar_dados_estrutura()


concreto = otm.Concreto(2400, 20, 0.2)
aco = otm.Aco(0, 20000)
dados = Dados(pathlib.Path(__file__).parent.joinpath('MBB_5000.zip'), concreto, aco, 1)

# Gerar malha
# malha = otm.Malha(dados, 5000)
# reg, verts = malha.criar_malha(3, 500)
# otm.GeradorMalha.exibir_malha(arq, False, 0.25)

# analisar_estrutura(dados)
#
rmin = 10
fv = 0.3
otimizador = OC(dados, fracao_volume=fv, p=3, rmin=rmin, tecnica_otimizacao=0)
otimizador.otimizar_estrutura(passo_p=1)
#
plot = Plot(dados)
# plot.plotar_malha(True)
# plot.plotar_estrutura_deformada(30)
plot.plotar_estrutura_otimizada(tecnica_otimizacao=0, corte_barras=0)
# plot.plotar_tensoes_estrutura()

# print(min(dados.deslocamentos_estrutura_original))
