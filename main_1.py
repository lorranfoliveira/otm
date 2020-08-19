import otm
import pathlib
from loguru import logger
from otm.otimizador.oc import OC


def analisar_estrutura(arquivo):
    r1 = [105, 107, 151, 395, 397, 399, 400, 401, 402, 403, 404, 405, 406, 431, 486, 712, 714, 729, 733, 734, 767, 768,
          994, 995, 1014, 1016, 1017, 1092, 1093, 1110, 1590, 1591, 1647, 2069, 2070, 2112, 2113, 2147, 2148, 2191,
          2204, 2898, 2902, 2903, 2953, 2954, 3518, 3519, 3532, 4942, 4943, 8330, 8333, 8337, 8338]

    apoios = {i: (1, 0) for i in r1}
    apoios[17] = (0, 1)

    forcas = {8330: (0, -1)}

    concreto = otm.MaterialIsotropico(1, 0.25)
    est = otm.Estrutura(arquivo, concreto, espessura=1, dict_cargas=forcas, dict_apoios=apoios)
    est.carregar_e_salvar_dados()


arq = pathlib.Path(__file__).parent.joinpath('Cantilever2.zip')
arq_dxf = pathlib.Path(__file__).parent.joinpath(f'{arq.name.replace("zip", "dxf")}')

# logger.add(f'{arq.name.replace(".zip", "")}.log')

# Gerar malha
# ite = otm.GeradorMalha(f'{arq_dxf}', 10000)
# reg, verts = ite.criar_malha(3, 100)
# otm.GeradorMalha.exibir_malha(arq, True, 0.25)

# analisar_estrutura(arq)
# print(min(otm.Estrutura.deslocamentos_arquivo(arq)))
# otm.Estrutura.plotar_estrutura_deformada(arq, 1e-2)
rmin = 0.
x_ini = 0.5

# Sem esquema de projeção
otimizador = OC(arq, x_inicial=x_ini, p=5, rmin=rmin, tecnica_otimizacao=0, esquema_projecao=0)
# otimizador = OC(arq, x_inicial=x_ini, p=5, rmin=rmin, tecnica_otimizacao=1, esquema_projecao=0)
# otimizador = OC(arq, x_inicial=x_ini, p=3, rmin=rmin, tecnica_otimizacao=2, esquema_projecao=0)

otimizador.otimizar_estrutura(passo=0.5)
otimizador.plotar_estrutura_otimizada()
# otimizador.plotar_estrutura_otimizada('binary')
