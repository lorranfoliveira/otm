import otm
import pathlib
from time import time
from datetime import timedelta
from otm.otimizador.oc import OC
from otm.plotagem import Plot
from otm.dados import Dados

# Dados de entrada
raio_ep = 4
fracao_volume = 0.35
delta_p = 0.5
p = 3


def analisar_estrutura(dados):
    # Nós engastados barras
    r1 = [2, 14, 49, 145, 146, 510, 657, 1610, 1611, 1612, 1613, 1636, 2014, 2016, 2021, 2028, 4085, 4086, 4094, 4372,
          4870, 4871, 4872, 5115, 8300, 8309, 8310, 8638, 8657, 8693, 9317, 9351, 12841, 12842, 12856, 12857, 12860,
          12861, 13125, 13530, 13531, 13581, 14224, 14225, 14238, 14239, 14268, 14272, 14273, 17029, 17030, 17044,
          17047, 17494, 17497, 17498, 17504, 17509, 17520, 17521, 17532, 17834, 17835, 17836, 17850, 17851, 17921,
          17925, 17926, 18378, 20373, 20380, 20381, 21012, 21013, 21014, 21020, 21021, 21049, 21059, 21060, 21061,
          21062, 21271, 21282, 21286, 21287, 21340, 21344, 21345, 23964, 23965, 24135, 24136, 24324, 24325]

    apoios = {i: (1, 1) for i in r1}
    forcas = {713: (0, -100)}
    est = otm.Estrutura(dados, espessura=1, dict_cargas=forcas, dict_apoios=apoios)
    est.salvar_dados_estrutura()


nome_arq = f'{pathlib.Path(__file__).stem}'
concreto = otm.Concreto(2490, 200, 0.2, 1)
aco = otm.Aco(0, 20000)
dados = Dados(pathlib.Path(__file__).parent.joinpath(f'{nome_arq}.zip'), concreto, aco)

# Gerar malha
malha = otm.Malha(dados, 10000)
malha.criar_malha(3, 300)
#
# analisar_estrutura(dados)
# otimizador = OC(dados, fracao_volume=fracao_volume, p=p, rmin=raio_ep, tecnica_otimizacao=1)
#
# t0 = time()
# otimizador.otimizar_estrutura(erro_max=0.1, passo_p=delta_p, parametro_fitro=1.4)
# with open(f'{dados.arquivo.stem}_tempo_execucao.txt', 'w') as arq:
#     arq.write(f'{timedelta(seconds=time() - t0)}')
#
plot = Plot(dados)
plot.plotar_malha()
# plot.plotar_estrutura_deformada(1e-2)
# plot.plotar_estrutura_otimizada(tecnica_otimizacao=1, rmin=0, tipo_cmap='binary')
# plot.plotar_animacao_otimizacao('jet')
