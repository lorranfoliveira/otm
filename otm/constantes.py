# Palavras chave dos arquivos de entrada e de saída de dados
_prefixos_chave = ['MALHA', 'ANALISE', 'OTIMIZACAO', 'RESULTADOS']
#  Malha.
ARQUIVOS_DADOS_ZIP = {0: f'{_prefixos_chave[0]}_elementos.npz',
                      1: f'{_prefixos_chave[0]}_nos.npy',
                      10: f'{_prefixos_chave[0]}_poligono_dominio_estendido.wkb',
                      16: f'{_prefixos_chave[0]}_nos_rastreados.txt',
                      # Análise.
                      4: f'{_prefixos_chave[1]}_forcas.npy',
                      5: f'{_prefixos_chave[1]}_graus_liberdade_por_elemento.npz',
                      6: f'{_prefixos_chave[1]}_apoios.npy',
                      7: f'{_prefixos_chave[1]}_matrizes_rigidez_elementos.npz',
                      8: f'{_prefixos_chave[1]}_volumes_elementos.npy',
                      9: f'{_prefixos_chave[1]}_graus_liberdade_estrutura.npy',
                      11: f'{_prefixos_chave[1]}_vetor_reverse_cuthill_mckee.npy',
                      12: f'{_prefixos_chave[1]}_dados_entrada_txt.txt',
                      17: f'{_prefixos_chave[1]}_deslocamentos_estrutura_original.npy',
                      18: f'{_prefixos_chave[1]}_matrizes_cinematicas_nodais_origem.npz',
                      19: f'{_prefixos_chave[1]}_matrizes_cinematicas_por_ponto_integracao',
                      20: f'{_prefixos_chave[1]}_matrizes_rigidez_barras.npz',
                      21: f'{_prefixos_chave[1]}_comprimentos_barras.npy',
                      # Otimização.
                      13: f'{_prefixos_chave[2]}_pesos_nos.npz',
                      # Resultados.
                      14: f'{_prefixos_chave[3]}_resultados_rho.npy',
                      15: f'{_prefixos_chave[3]}_resultados_gerais.npy',
                      22: f'{_prefixos_chave[3]}_tensoes_elementos.npy'}

# Inserir a matriz B no arquivo para a utilização do concreto como material ortotrópico

ARQ_SAIDA_DADOS = {0: 'OTIMIZACAO_variaveis_projeto_por_iteracao.npz',
                   1: 'OTIMIZACAO_flexibilidade_por_iteracao.npz'}
