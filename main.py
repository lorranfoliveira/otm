import otm
import pathlib
from otm.otimizador.oc import OC
from otm.plotagem import Plot
from otm.dados import Dados

# Dados de entrada
raio_ep = 7
fracao_volume = 0.5
delta_p = 0.5
p = 5


def analisar_estrutura(dados):
    # Nós engastados barras
    r1 = [24, 153, 154, 213, 518, 548, 603, 604, 606, 607, 616, 617, 1183, 1185, 1186, 1189, 1235, 2474, 2475, 2480,
          2498, 4071, 4075, 4076, 4077, 4313, 5905, 5906, 5907, 5931, 5932, 6063, 6065, 6066, 7319, 7320]

    apoios = {**{i: (1, 0) for i in r1},
              **{989: (0, 1)}}
    forcas = {1235: (0, -10)}

    est = otm.Estrutura(dados, espessura=1, dict_cargas=forcas, dict_apoios=apoios)
    est.salvar_dados_estrutura()


nome_arq = f'mbb'
concreto = otm.Concreto(2490, 200, 0.2)
aco = otm.Aco(0, 20000)
dados = Dados(pathlib.Path(__file__).parent.joinpath(f'{nome_arq}.zip'), concreto, aco, 0)

# Gerar malha
# malha = otm.Malha(dados, 4800)
# malha.criar_malha(3, 300, 0.1, 2, 2, 10)

# analisar_estrutura(dados)

# otimizador = OC(dados, fracao_volume=fracao_volume, p=p, rmin=raio_ep, tecnica_otimizacao=0)
# otimizador.otimizar_estrutura(passo_p=delta_p, aplicar_filtro=False)

plot = Plot(dados)
plot.plotar_malha()
# plot.plotar_estrutura_deformada(1e-3)
# plot.plotar_estrutura_otimizada(tecnica_otimizacao=1, rmin=raio_ep, tipo_cmap='jet')
