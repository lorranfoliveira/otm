import otm
import pathlib
from otm.otimizador.oc import OC
from otm.plotagem import Plot
from otm.dados import Dados

# Dados de entrada
raio_ep = 5
fracao_volume = 0.35
delta_p = -1
p = 3


def analisar_estrutura(dados):
    # Nós engastados barras
    r1 = [3678, 7905, 27202, 8233, 27098, 3661]
    r2 = [201, 2935, 30615, 30392, 30190, 11655]

    # Nós engastados contorno
    r3 = [6, 352, 356, 358, 359, 1289, 1321, 3665, 3680, 3681, 3712, 3879, 3880, 3884, 3885, 7803, 7804, 7819, 7822,
          7823, 7824, 7850, 7851, 7852, 7853, 7911, 7912, 7957, 7958, 7962, 7963, 8230, 8237, 8238, 8239, 13803, 13804,
          13859, 20520, 20521, 20522, 20523, 20528, 20537, 20548, 20559, 20560, 20577, 27102, 27103, 27106, 27192,
          27204, 27721, 27722, 32709, 32710, 32758, 32759, 32834, 32835, 32845, 32846]

    r4 = [26, 211, 213, 980, 2909, 2910, 2917, 2918, 2925, 2926, 2951, 3007, 6274, 6287, 6288, 6289, 6305, 11498, 11499,
          11525, 11653, 11654, 11658, 11659, 17776, 17777, 17915, 17935, 17937, 17938, 17939, 17960, 17961, 18080,
          24363, 24364, 24373, 24374, 24377, 24378, 24421, 24422, 24697, 30194, 30195, 30401, 30402, 30617, 30618,
          34986, 35001, 35006, 38037, 38038, 38051, 38052, 38056, 38057, 38105, 38106]

    r5 = r1 + r2 + r3 + r4

    apoios = {i: (1, 1) for i in r5}
    forcas = {18352: (0, -100)}

    est = otm.Estrutura(dados, espessura=1, dict_cargas=forcas, dict_apoios=apoios)
    est.salvar_dados_estrutura()


nome_arq = f'consolo'
concreto = otm.Concreto(2490, 200, 0.2)
aco = otm.Aco(0, 20000)
dados = Dados(pathlib.Path(__file__).parent.joinpath(f'{nome_arq}.zip'), concreto, aco, 1)

# Gerar malha
malha = otm.Malha(dados, 21000)
reg, verts = malha.criar_malha(3, 300)
#
# analisar_estrutura(dados)
#
# otimizador = OC(dados, fracao_volume=fracao_volume, p=p, rmin=raio_ep, tecnica_otimizacao=1)
# otimizador.otimizar_estrutura(passo_p=delta_p)

plot = Plot(dados)
plot.plotar_malha()
# plot.plotar_estrutura_deformada(10)
# plot.plotar_estrutura_otimizada(tecnica_otimizacao=1, corte_barras=0, rmin=raio_ep, tipo_cmap='jet')
