import otm
import pathlib
from otm.otimizador.oc import OC
from otm.plotagem import Plot
from otm.dados import Dados


def analisar_estrutura(dados):
    apoios = {i: (1, 1) for i in
              [83, 84, 92, 218, 219, 220, 234, 238, 589, 590, 591, 595, 596, 599, 600, 1255, 1618, 1657, 1658, 1659,
               1674, 1675]}
    forcas = {1520: (0, -1)}

    est = otm.Estrutura(dados, espessura=1, dict_cargas=forcas, dict_apoios=apoios)
    est.salvar_dados_estrutura()


concreto = otm.Concreto(2.4, 0.2, 0.2)
aco = otm.Aco(0, 20)
dados = Dados(pathlib.Path(__file__).parent.joinpath('Cantilever.zip'), concreto, aco, 1)

# Gerar malha
# malha = otm.Malha(dados, 1000)
# reg, verts = malha.criar_malha(3, 300)
# otm.GeradorMalha.exibir_malha(arq, False, 0.25)

# analisar_estrutura(dados)
# plot = Plot(dados)
# plot.plotar_malha()
# plot.plotar_estrutura_deformada(0.1)

rmin = 1
x_ini = 0.5
otimizador = OC(dados, fracao_volume=x_ini, p=5, rmin=rmin, tecnica_otimizacao=0)
otimizador.otimizar_estrutura(passo_p=0.5)
# plot.plotar_estrutura_otimizada(0)
# plot.plotar_tensoes_estrutura()
