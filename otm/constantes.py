# Palavras chave dos arquivos de entrada e de saída de dados
_prefixos_chave = ['MALHA', 'ANALISE', 'OTIMIZACAO', 'RESULTADOS']

ARQUIVOS_DADOS_ZIP = {0: f'{_prefixos_chave[0]}_elementos.npz',
                      1: f'{_prefixos_chave[0]}_nos.npy',
                      2: f'{_prefixos_chave[0]}_diametro_medio_elementos.npy',
                      3: f'{_prefixos_chave[0]}_limites_dominio.npy',
                      4: f'{_prefixos_chave[1]}_forcas.npy',
                      5: f'{_prefixos_chave[1]}_graus_liberdade_por_elemento.npz',
                      6: f'{_prefixos_chave[1]}_apoios.npy',
                      7: f'{_prefixos_chave[1]}_matrizes_rigidez_elementos.npz',
                      8: f'{_prefixos_chave[1]}_volumes_elementos.npy',
                      9: f'{_prefixos_chave[1]}_graus_liberdade_estrutura.npy',
                      10: f'{_prefixos_chave[0]}_poligono_dominio_estendido.wkb',
                      11: f'{_prefixos_chave[1]}_vetor_reverse_cuthill_mckee.npy',
                      12: f'{_prefixos_chave[1]}_dados_entrada_txt.txt',
                      13: f'{_prefixos_chave[2]}_pesos_nos.npz',
                      14: f'{_prefixos_chave[3]}_resultados_rho.npy',
                      15: f'{_prefixos_chave[3]}_resultados_gerais.npy'}
# Inserir a matriz B no arquivo para a utilização do concreto como material ortotrópico

ARQ_SAIDA_DADOS = {0: 'OTIMIZACAO_variaveis_projeto_por_iteracao.npz',
                   1: 'OTIMIZACAO_flexibilidade_por_iteracao.npz'}
