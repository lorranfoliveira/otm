import otm
import pathlib
from otm.otimizador.oc import OC
from otm.plotagem import Plot
from otm.dados import Dados

# Dados de entrada
raio_ep = 7
fracao_volume = 0.4
delta_p = -1
p = 3


def analisar_estrutura(dados):
    # Nós engastados barras
    r1 = [972, 993, 994, 2259, 2262, 2421, 2422, 2681, 2703, 3429, 3444, 3448, 3449, 3464, 5959, 5960, 5961, 5962, 6083,
          6084, 6085, 6345, 6349, 6350, 6359, 6375, 6833, 6841, 6947, 7005, 8197, 8215, 8227, 8228, 8236, 8291, 9242,
          9348, 9350, 9352, 9362, 9363, 11538, 11539, 11540, 11663, 11670, 12034, 12035, 12053, 12057, 12058, 12059,
          12060, 12061, 12067, 12667, 12671, 12678, 12891, 12894, 12934, 12996, 13020, 14327, 14328, 14330, 14331,
          14377, 15527, 15532, 16224, 16227, 17543, 17970, 17972, 17980, 17982, 17983, 17984, 17988, 17989, 17999,
          18000, 18693, 18717, 19134, 19141, 19145, 19174, 19274, 19296, 19303, 20427, 20428, 20451, 20581, 20587,
          20588, 20644, 21262, 21305, 21777, 21779, 21802, 23204, 23216, 23217, 23233, 23237, 23238, 23239, 24197,
          24205, 24206, 24210, 25116, 25934, 26007, 26008, 26744, 26747, 26748, 26771, 27506, 27527, 27789, 27790,
          28667, 28668, 29585, 29608, 29609, 29941, 29954, 29955, 29958, 29959, 29960, 29983, 30462, 30463, 30464,
          30465, 30497, 30953, 30956, 31135, 31142, 31143, 31148, 31943, 31944, 32672, 32709, 32710, 32759, 33994,
          34515, 34518, 34519, 35098, 35114, 35115, 35154, 35172, 35800, 35801, 35805, 35870, 35871, 35872, 35879,
          36135, 36346, 36464, 36468, 36470, 36471, 36472, 36473, 36486, 36488, 36489, 36490, 36532, 36533, 36605,
          36894, 36988, 37008, 37254, 37258, 37259, 37266, 37267, 37268, 37406, 37787, 37792, 37797, 37799, 37821,
          37822, 37826, 37827, 37828, 37849, 37858, 38130, 38136, 38266, 38272, 38273, 38519, 38520, 38526, 38529,
          38582, 38583, 38600, 38601, 38608, 38609, 38636, 38637, 38812, 38816, 38963, 38967, 38968, 39113, 39114,
          39115, 39168, 39196, 39199, 39367, 39536, 39537, 39592, 39685, 39686, 39701, 39705, 39788, 39814, 39815,
          39846]

    apoios = {i: (1, 1) for i in r1}
    c = 25
    forcas = {5793: (0, -c),
              23907: (c, 0),
              13621: (0, c),
              10574: (-c, 0)}

    est = otm.Estrutura(dados, espessura=1, dict_cargas=forcas, dict_apoios=apoios)
    est.salvar_dados_estrutura()


nome_arq = f'circulo_1'
concreto = otm.Concreto(2490, 200, 0.2)
aco = otm.Aco(0, 20000)
dados = Dados(pathlib.Path(__file__).parent.joinpath(f'{nome_arq}.zip'), concreto, aco, 1)

# Gerar malha
# malha = otm.Malha(dados, 20000)
# malha.criar_malha(3, 300, 0.1, 2, 5, 2)

# analisar_estrutura(dados)

otimizador = OC(dados, fracao_volume=fracao_volume, p=p, rmin=raio_ep, tecnica_otimizacao=1)
otimizador.otimizar_estrutura(passo_p=delta_p, parametro_fitro=10)

plot = Plot(dados)
# plot.plotar_malha()
# plot.plotar_estrutura_deformada(1)
plot.plotar_estrutura_otimizada(tecnica_otimizacao=1, rmin=raio_ep, tipo_cmap='jet')
# plot.plotar_animacao_otimizacao('jet')
