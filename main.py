import otm
import pathlib
from otm.otimizador.oc import OC
from otm.plotagem import Plot
from otm.dados import Dados

# Dados de entrada
numero_modelo = 33
raio_ep = 5
fracao_volume = 0.36
delta_p = -1
p = 3


def analisar_estrutura(dados):
    r1 = [0, 6, 11, 133, 167, 168, 169, 400, 505, 506, 1252, 1254, 1256, 1257, 1309, 1349, 1350, 1351, 1352, 2813, 2815,
          2818, 2926, 2927, 2936, 2952, 3005, 3006, 3008, 3011, 3012, 3019, 3020, 3021, 3161, 4986, 4987, 5143, 5161,
          5179, 5400, 5546, 5547, 5549, 7552, 7555, 7556, 7570, 7572, 7573, 7599, 7601, 7603, 7604, 7801, 7802, 9926,
          9927, 9928, 9935, 9937, 9938, 9954, 9955, 10087, 10212, 10213, 11795, 11796, 11828, 11879, 11880, 11881,
          11889, 11890, 11891, 11892, 11895, 11896, 11897, 11911, 13050, 13051, 13053, 13054, 13123, 13124]

    r2 = [31, 7581, 13052, 3]

    r3 = r1 + r2

    apoios = {**{i: (1, 0) for i in r3}, **{74: (0, 1)}}
    forcas = {31: (0, -100)}

    est = otm.Estrutura(dados, espessura=1, dict_cargas=forcas, dict_apoios=apoios)
    est.salvar_dados_estrutura()


nome_arq = f'MBB_{numero_modelo}'
concreto = otm.Concreto(2490, 200, 0.2)
aco = otm.Aco(0, 20000)
dados = Dados(pathlib.Path(__file__).parent.joinpath(f'{nome_arq}.zip'), concreto, aco, 1)

# Gerar malha
# malha = otm.Malha(dados, 14620, 'retangular')
# reg, verts = malha.criar_malha(3, 300)

# analisar_estrutura(dados)
#
otimizador = OC(dados, fracao_volume=fracao_volume, p=p, rmin=raio_ep, tecnica_otimizacao=1)
otimizador.otimizar_estrutura(passo_p=delta_p)

plot = Plot(dados)
# plot.plotar_malha()
# plot.plotar_estrutura_deformada(10)
plot.plotar_estrutura_otimizada(tecnica_otimizacao=1, corte_barras=0, rmin=raio_ep)

