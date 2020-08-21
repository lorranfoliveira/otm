import otm
import pathlib
from loguru import logger
from otm.otimizador.oc import OC
from otm.plotagem import Plot
from otm.dados import Dados


def analisar_estrutura(arquivo):
    r1 = [117, 210, 302, 307, 369, 487, 492, 493, 587, 589, 590, 664, 762, 763, 818, 819, 862, 863]

    apoios = {i: (1, 1) for i in r1}
    forcas = {934: (0, -1)}

    concreto = otm.MaterialIsotropico(1, 0.25)
    est = otm.Estrutura(arquivo, concreto, espessura=1, dict_cargas=forcas, dict_apoios=apoios)
    est.carregar_e_salvar_dados()


arq = pathlib.Path(__file__).parent.joinpath('O.zip')
arq_dxf = pathlib.Path(__file__).parent.joinpath(f'{arq.name.replace("zip", "dxf")}')

# logger.add(f'{arq.name.replace(".zip", "")}.log')

# Gerar malha
# ite = otm.GeradorMalha(f'{arq_dxf}', 500)
# reg, verts = ite.criar_malha(3, 100)
# otm.GeradorMalha.exibir_malha(arq, False, 0.25)

# analisar_estrutura(arq)
# print(min(otm.Estrutura.deslocamentos_arquivo(arq)))
# otm.Estrutura.plotar_estrutura_deformada(arq, 1e-2)
# rmin = 0.25
# x_ini = 0.5
#
# Sem esquema de projeção
# otimizador = OC(arq, x_inicial=x_ini, p=5, rmin=0, tecnica_otimizacao=0, esquema_projecao=0)
# otimizador = OC(arq, x_inicial=x_ini, p=5, rmin=rmin, tecnica_otimizacao=1, esquema_projecao=0)
# otimizador = OC(arq, x_inicial=x_ini, p=3, rmin=rmin, tecnica_otimizacao=2, esquema_projecao=0)
#
# otimizador.otimizar_estrutura(passo=0.5)
dados = Dados(arq)
plot = Plot(dados)
plot.plotar_malha()
