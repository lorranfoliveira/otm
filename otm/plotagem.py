from loguru import logger
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.collections import PatchCollection
from matplotlib import path
from matplotlib import cm
from otm.otimizador.oc import OC
from otm.dados import Dados

__all__ = ['Plot']


class Plot:
    """Classe que implementa todos os gráficos referentes aos resultados"""

    def __init__(self, dados: Dados):
        self.dados = dados

    def plotar_grafico_generico(self, id_y: int = 3, titulo: str = ''):
        """Plota um gráfico de um dado genérico em `y` em função das iterações `x`.

        Args:
            id_y: Identificação da coluna do dado (resultados gerais) a ser representado na abcissa do gráfico.
                # 0 -> Id da iteração para valores constantes de `p` e `beta`, ou `c` (do código abaixo).
                # 1 -> `p`.
                # 2 -> `beta`.
                # 3 -> Valor da função objetivo.
                # 4 -> Percentual de volume da estrutura após a otimização em relação ao volume inicial.
                # 5 -> Percentual de densidades intermediárias.
                # 6 -> Erro relacionado aos deslocamentos.
                # 7 -> Erro relacionado ao percentual de densidades intermediárias.
            titulo: Título do gráfico.
        """
        y = self.dados.resultados_gerais[:, id_y]
        plt.title(titulo)
        plt.plot(y)
        plt.show()

    def plotar_malha(self, exibir_numeracao_nos=False, tamanho_numeracao=1):
        """Exibe a malha de elementos fintos do problema.

        Args:
            exibir_numeracao_nos: `True` para exibir a numeração dos nós.
            tamanho_numeracao: Regula o tamanho da numeração dos nós da malha.
        """
        logger.info('Criando o desenho da malha final')

        # Plotagem.
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.family'] = 'Calibri'
        fig, ax = plt.subplots()
        win = plt.get_current_fig_manager()
        win.window.state('zoomed')
        ax.axis('equal')

        xmin, ymin, xmax, ymax = self.dados.poligono_dominio_estendido.bounds
        dx = xmax - xmin
        dy = ymax - ymin
        plt.xlim(xmin - 0.1 * dx, xmax + 0.1 * dx)
        plt.ylim(ymin - 0.1 * dy, ymax + 0.1 * dy)

        elementos_poli = []
        # elementos_barra = []
        for el in self.dados.elementos:
            if len(el) == 2:
                pass
                # verts = [nos[el[0]], nos[el[1]]]
                # codes = [Path.MOVETO, Path.LINETO]
                # elementos_barra.append(Path(verts, codes))
            elif len(el) > 2:
                elementos_poli.append(patches.Polygon(self.dados.nos[el], linewidth=0.5, facecolor='None',
                                                      edgecolor='black'))

        ax.add_collection(PatchCollection(elementos_poli, match_original=True))
        # ax.add_collection(PathCollection(elementos_barra, linewidths=0.7, edgecolors='purple'))

        # Enumerar os pontos
        if exibir_numeracao_nos:
            for i, v in enumerate(self.dados.nos):
                ax.text(v[0], v[1], f'{i}', ha="center", va="center", size=tamanho_numeracao, color='b')

        # logger.info(f'Salvando a malha final em "{arquivo_entrada_dados.replace("zip", "svg")}"')
        plt.axis('off')
        plt.grid(b=None)
        plt.show()

    def plotar_estrutura_otimizada(self, tecnica_otimizacao: int, rmin: float = 0,
                                   tipo_cmap: str = 'jet'):
        """Exibe a malha final gerada. cmad jet ou binary"""
        logger.info('Criando o desenho da malha final')

        # Resultados finais
        rho_final = self.dados.rhos_iteracao_final()
        results_gerais_finais = self.dados.rhos_iteracao_final()

        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.family'] = 'Calibri'

        fig, ax = plt.subplots()
        win = plt.get_current_fig_manager()
        win.window.state('zoomed')
        ax.axis('equal')

        xmin, ymin, xmax, ymax = self.dados.poligono_dominio_estendido.bounds
        dx = xmax - xmin
        dy = ymax - ymin
        plt.xlim(xmin - 0.1 * dx, xmax + 0.1 * dx)
        plt.ylim(ymin - 0.1 * dy, ymax + 0.1 * dy)

        elementos_poli = []
        for j, el in enumerate(self.dados.elementos):
            if tipo_cmap == 'jet':
                elementos_poli.append(patches.Polygon(self.dados.nos[el], linewidth=0, fill=True,
                                                      facecolor=cm.jet(rho_final[j])))
            else:
                elementos_poli.append(patches.Polygon(self.dados.nos[el], linewidth=0, fill=True,
                                                      facecolor=cm.binary(rho_final[j])))

        # Adicionar marcador do diâmetro mínimo dos elementos
        path_diam_verts = [[xmax - rmin * 2 - 0.01 * dx, ymax - 0.01 * dx],
                           [xmax - 0.01 * dx, ymax - 0.01 * dx]]
        path_diam_codes = [path.Path.MOVETO, path.Path.LINETO]
        path_diam = path.Path(path_diam_verts, path_diam_codes)
        ax.add_patch(patches.PathPatch(path_diam, linewidth=2, color='magenta'))

        ax.add_collection(PatchCollection(elementos_poli, match_original=True))
        # ax.add_collection(PathCollection(elementos_barra, linewidths=0.7, edgecolors='purple'))
        plt.axis('off')
        plt.grid(b=None)

        # Título
        # Fixos
        di = f'Di: {results_gerais_finais[5]:.2f}%'
        els = f'NumElems: {len(self.dados.elementos)}'
        vf = f'vol: {results_gerais_finais[4]:.4f}%'
        # Variáveis
        rmin = ''
        tecnica_otm = 'Técnica: '

        if tecnica_otimizacao == 0:
            tecnica_otm += 'Sem filtro'
        else:
            rmin = f'rmin: {rmin}'

            if tecnica_otimizacao in OC.TECNICA_OTM_EP_LINEAR:
                tecnica_otm += 'Linear '
            elif tecnica_otimizacao in OC.TECNICA_OTM_EP_HEAVISIDE:
                tecnica_otm += 'Heaviside '
            else:
                raise ValueError(f'A técnica de otimização "{tecnica_otimizacao}" não é válida!')

            if tecnica_otimizacao in OC.TECNICA_OTM_EP_DIRETO:
                tecnica_otm += 'direto'
            elif tecnica_otimizacao in OC.TECNICA_OTM_EP_INVERSO:
                tecnica_otm += 'inverso'
            else:
                raise ValueError(f'A técnica de otimização "{tecnica_otimizacao}" não é válida!')

        plt.title(f'{tecnica_otm}     {els}     {vf}     {di}    {rmin}')
        plt.show()
