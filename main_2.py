import otm
import pathlib
from otm.otimizador.oc import OC
from loguru import logger


def analisar_estrutura(arquivo):
    apoios = {i: (1, 0) for i in
              [7, 11, 13, 25, 27, 29, 106, 283, 499, 513, 514, 578, 581, 587, 588, 589, 590, 608, 609, 614, 615, 620,
               621, 747, 748, 752, 753, 756, 771, 772, 773, 780, 799, 806, 818, 950, 1040, 1041, 1066, 1097, 1200, 1201,
               1209, 1359, 1360, 1396, 1399, 1423, 1426, 1791, 1792, 1809, 1826, 1850, 1851, 2016, 2168, 2172, 2173,
               2212, 2214, 2215, 2257, 2258, 2261, 2265, 2266, 2663, 2664, 2711, 2712, 2729, 2730, 3136, 3147, 3156,
               3157, 3392, 3393, 3398, 3401, 3804, 3805, 3806, 3807, 3866, 4268, 4269, 4270, 4474, 4478, 4479, 4480,
               4603, 4604, 4605, 4739, 5273, 5277, 5296, 5569, 5570, 5660, 5674, 5675, 5679, 5680, 5765, 5767, 5768,
               5772, 5786, 5793, 6411, 6412, 6853, 6855, 6993, 6996, 7002, 7003, 7012, 7013, 8033, 8038, 8039, 8081,
               8082, 8083, 8084, 8102, 8103, 8901, 9484, 9485, 9486, 9489, 9892, 9893]}

    r1 = [1, 7, 28, 125, 126, 305, 315, 316, 339, 340, 717, 1315, 1316, 1460, 1461, 2111, 2112]
    for i in r1:
        apoios[i] = (1, 1)

    forcas = {46: (0, -0.5),
              4325: (0, -0.5)}

    concreto = otm.MaterialIsotropico(1, 0.25)
    est = otm.Estrutura(arquivo, concreto, espessura=1, dict_cargas=forcas, dict_apoios=apoios)
    est.carregar_e_salvar_dados()


arq = pathlib.Path(__file__).parent.joinpath('T.zip')
arq_dxf = pathlib.Path(__file__).parent.joinpath(f'{arq.name.replace("zip", "dxf")}')

# logger.add(f'{arq.name.replace(".zip", "")}.log')

# Gerar malha
# ite = otm.GeradorMalha(f'{arq_dxf}', 10000)
# reg, verts = ite.criar_malha(3, 100)
# otm.GeradorMalha.exibir_malha(arq, True, 0.25)

# analisar_estrutura(arq)
# print(min(otm.Estrutura.deslocamentos_arquivo(arq)))
# otm.Estrutura.plotar_estrutura_deformada(arq, 0)
rmin = 0
x_ini = 0.5

# Sem esquema de projeção
otimizador = OC(arq, x_inicial=x_ini, p=5, rmin=rmin, tecnica_otimizacao=0, esquema_projecao=1)
# otimizador = OC(arq, x_inicial=x_ini, p=5, rmin=rmin, tecnica_otimizacao=1, esquema_projecao=1)
# otimizador = OC(arq, x_inicial=x_ini, p=3, rmin=rmin, tecnica_otimizacao=2, esquema_projecao=1)
otimizador.otimizar()
otimizador.otimizar_estrutura(passo=0.5)
otimizador.plotar_estrutura_otimizada()
# otimizador.plotar_estrutura_otimizada('binary')
