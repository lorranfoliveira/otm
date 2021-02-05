import otm
import pathlib
from time import time
from datetime import timedelta
from otm.otimizador.oc import OC
from otm.plotagem import Plot
from otm.dados import Dados
from dados_constantes_problemas import *

# Dados de entrada
vf = 0.4
rmin = 3.5
delta_p = -1
tecnica_otm = 2
tipo_concreto = 0


def analisar_estrutura(dados):
    # Nós engastados barras
    r1 = [350, 1392, 1470, 2456, 3300, 3301, 3329, 3330, 3572, 3637, 5144, 5146, 5147, 6951, 6971, 6981, 6985, 6986,
          7036, 7448, 7451, 7460, 7464, 7471, 7570, 7571, 7585, 7586, 7587, 7588, 7595, 7596, 9593, 9598, 9633, 9637,
          9638, 12194, 12221, 12228, 12229, 12230, 12503, 12521, 12525, 12557, 12558, 12929, 12930, 12934, 12942, 12943,
          12947, 12948, 15309, 15313, 15314, 15348, 18227, 18228, 18237, 18238, 18242, 18243, 18246, 18247, 18248,
          18507, 18508, 18513, 18514, 18946, 18947, 18949, 18954, 18955, 21751, 21752, 21794, 21795, 21796, 21799,
          24737, 24751, 24752, 28024, 30309, 33901, 33902, 37899, 37900, 37901, 37902]

    apoios = {**{i: (1, 0) for i in r1},
              **{4412: (0, 1)}}
    forcas = {9593: (0, -50)}
    est = otm.Estrutura(dados, espessura=1, dict_cargas=forcas, dict_apoios=apoios)
    est.salvar_dados_estrutura()


nome_arq = f'{pathlib.Path(__file__).stem}'
concreto = otm.Concreto(Ecc, Ecc if tipo_concreto == 0 else Ect, nu, tipo_concreto)
aco = otm.Aco(0, Ea)
dados = Dados(pathlib.Path(__file__).parent.joinpath(f'{nome_arq}.zip'), concreto, aco)

# Gerar malha
# malha = otm.Malha(dados, 20000)
# malha.criar_malha(3, 200)
#
# analisar_estrutura(dados)
# otimizador = OC(dados, fracao_volume=vf, p=p, rmin=rmin, tecnica_otimizacao=tecnica_otm)

# t0 = time()
# otimizador.otimizar_estrutura(erro_max=0.1, passo_p=delta_p, parametro_fitro=None)
# with open(f'{dados.arquivo.stem}_dados.txt', 'w') as arq:
#     titulos = ['Iterações',
#                'p',
#                'beta',
#                'Função objetivo',
#                'Volume total final',
#                'Percentual de densidades intermediárias',
#                'Erro deslocamentos',
#                'Erro densidades intermediárias',
#                'Percentual de volume dos elementos contínuos',
#                'Percentual de volume dos elementos de treliça']
#
#     arq.write(f'Tempo -> {timedelta(seconds=time() - t0)}\n')
#
#     res_fin = dados.resultados_gerais_iteracao_final()
#     for i in range(len(res_fin)):
#         arq.write(f'{titulos[i]}: {res_fin[i]}\n')

plot = Plot(dados)
# plot.plotar_malha()
# plot.plotar_estrutura_deformada(1e-2)
plot.plotar_estrutura_otimizada(tecnica_otimizacao=1, tipo_cmap='binary')
plot.plotar_tensoes()
