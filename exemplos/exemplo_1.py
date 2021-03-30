import otm
import pathlib
from otm.plotagem import Plot
from otm.dados import Dados
from otm.otimizador.oc import OC

"""Descrição do problema:

Consolo curto de pilar biengastado com carga vertical para baixo em sua extremidade.
"""

# ======================== DADOS DE ENTRADA ========================
# ----- MATERIAIS -----

# Módulo de elasticidade do concreto à compressão.
Ecc = 2490
# Módulo de elasticidade do concreto à tração.
Ect = 200
# Modelo de concreto a ser utilizado:
# Tipo 0: Modelo isotrópico do concreto.
# Tipo 1: Modelo ortottrópico do concreto.
tipo_concreto = 0
# Módulo de elasticidade do aço à tração.
Eat = 2e4
# Módulo de elasticidade do aço à compressão.
Eac = 0
# Coeficiente de Poisson.
nu = 0.2

# ----- MALHA -----
# Número de elementos finitos bidimensionais.
num_els_poli = 2000
# Nível de conectividade da treliça hiperconectada. Se não for pelo método híbrido, utilizar None.
nivel_conect = None
# Espaçamento entre os nós da grade da treliça hiperconectada. Se não for pelo método híbrido, utilizar None.
espacamento_grade = None
# Distância entre a face do domínio e aa primeira reta da grade. Se não for pelo método híbrido, utilizar None.
d = None
# ----- OTIMIZAÇÃO -----
# Espessura dos elementos finitos bidimensionais.
esp = 1
# Raio de projeção.
raio_ep = 4
# Coeficiente de penalização máximo.
p = 3
# Fração de volume inicial.
vf = 0.35
# Variação do coeficiente de penalização no método da continuação.
delta_p = 0.5
# Técnicas de otimização.
# 0 -> Sem filtro.
# 1 -> Com esquema de projeção linear direto.
# 2 -> Com esquema de projeção Heaviside direto.
# 3 -> Com esquema de projeção linear inverso.
# 4 -> Com esquema de projeção Heaviside inverso.
tecnica_otm = 1
# Parâmetro do filtro de barras finas.
param_filtro = 1.4


# ======================== ANÁLISE ESTRUTURAL ========================

def analisar_estrutura(dados, espessura):
    # Nós engastados barras
    r1 = [2, 14, 49, 145, 146, 510, 657, 1610, 1611, 1612, 1613, 1636, 2014, 2016, 2021, 2028, 4085, 4086, 4094, 4372,
          4870, 4871, 4872, 5115, 8300, 8309, 8310, 8638, 8657, 8693, 9317, 9351, 12841, 12842, 12856, 12857, 12860,
          12861, 13125, 13530, 13531, 13581, 14224, 14225, 14238, 14239, 14268, 14272, 14273, 17029, 17030, 17044,
          17047, 17494, 17497, 17498, 17504, 17509, 17520, 17521, 17532, 17834, 17835, 17836, 17850, 17851, 17921,
          17925, 17926, 18378, 20373, 20380, 20381, 21012, 21013, 21014, 21020, 21021, 21049, 21059, 21060, 21061,
          21062, 21271, 21282, 21286, 21287, 21340, 21344, 21345, 23964, 23965, 24135, 24136, 24324, 24325]

    apoios = {i: (1, 1) for i in r1}
    forcas = {713: (0, -100)}
    est = otm.Estrutura(dados, espessura=espessura, dict_cargas=forcas, dict_apoios=apoios)
    est.salvar_dados_estrutura()


concreto = otm.Concreto(Ecc, Ect, nu, tipo_concreto)
aco = otm.Aco(Eac, Eat)
nome_arq = f'{pathlib.Path(__file__).stem}'
dados = Dados(pathlib.Path(__file__).parent.joinpath(f'{nome_arq}.zip'), concreto, aco)

# ======================== GERAÇÃO DA MALHA ========================
malha = otm.Malha(dados, num_els_poli)
malha.criar_malha(3, 150, nivel_conect, espacamento_grade, d)

# ======================== OTIMIZAÇÃO ========================
# analisar_estrutura(dados, esp)
# otimizador = OC(dados, fracao_volume=vf, p=p, rmin=raio_ep, tecnica_otimizacao=tecnica_otm)
# otimizador.otimizar_estrutura(erro_max=0.1, passo_p=delta_p, parametro_fitro=1.4)

# ======================== PLOTAGEM ========================
plot = Plot(dados)
plot.plotar_malha()
# plot.plotar_estrutura_deformada(1e-2)
# Tipo de colormap pode ser 'binary' e 'jet'.
# plot.plotar_estrutura_otimizada(tecnica_otimizacao=tecnica_otm, tipo_cmap='binary')
