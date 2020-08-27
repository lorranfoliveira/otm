from loguru import logger
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.collections import PatchCollection, PathCollection
from matplotlib import path
from otm.mef.estrutura import Estrutura
from otm.mef.elementos import ElementoPoligonalIsoparametrico
from matplotlib import cm
import shapely.geometry as geo
from otm.otimizador.oc import OC
from otm.dados import Dados
from sympy.utilities.lambdify import lambdify
import numpy as np

__all__ = ['Plot']


class Plot:
    """Classe que implementa todos os gráficos referentes aos resultados"""

    def __init__(self, dados: Dados):
        self.dados = dados

    @staticmethod
    def plotar_funcao_forma_elemento(num_lados: int, id_no: int = 0):
        """Plota o gráfico da função de forma para um elemento isoparamétrico com `n` lados.

        Args:
            num_lados: Número de lados que possui o elemento isoparamétrico.
            id_no: Identificação do nó de referência onde a função de forma valerá 1.
        """
        elem = ElementoPoligonalIsoparametrico(num_lados)
        func_symb = elem.funcoes_forma()[id_no]
        func_lamb = lambdify(func_symb.free_symbols, func_symb, 'numpy')

        n = 300
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        xx, yy = np.meshgrid(x, y)
        zz = func_lamb(xx, yy)

        # Exclusão da parte inválida do gráfico
        poli = geo.Polygon(elem.nos)

        for i in range(zz.shape[0]):
            for j in range(zz.shape[1]):
                p = geo.Point(xx[i, j], yy[i, j])
                if poli.disjoint(p):
                    zz[i, j] = np.nan

        # Plotagem
        fig, ax = plt.subplots(1, 1)
        cp = ax.contourf(xx, yy, zz, cmap='viridis')
        fig.colorbar(cp)
        ax.axis('equal')

        plt.axis('off')
        plt.grid(b=None)

        plt.show()

    def posicao_nos_deslocados(self, escala: int = 1) -> np.ndarray:
        """Retorna uma matriz com as coordenadas dos nós deslocados.

        Args:
            escala: Escala aplicada nos deslocamentos para o auxílio na plotagem.
        """
        return self.dados.nos + escala * self.dados.deslocamentos_por_no()

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

    def plotar_estrutura_deformada(self, escala=1):
        """Exibe a malha final gerada"""

        logger.debug('Plotando a estrutura deformada')

        # Leitura dos dados importantes
        nos = self.dados.nos
        poli = self.dados.poligono_dominio_estendido
        vetor_elementos = self.dados.elementos

        fig, ax = plt.subplots()
        win = plt.get_current_fig_manager()
        win.window.state('zoomed')
        ax.axis('equal')

        xmin, ymin, xmax, ymax = poli.bounds
        dx = xmax - xmin
        dy = ymax - ymin
        plt.xlim(xmin - 0.1 * dx, xmax + 0.1 * dx)
        plt.ylim(ymin - 0.1 * dy, ymax + 0.1 * dy)

        nos_def = self.posicao_nos_deslocados(escala)

        elementos_poli_original = []
        elementos_poli_deformado = []
        # elementos_barra = []
        for el in vetor_elementos:
            if len(el) == 2:
                pass
                # verts_original = [nos[el[0]], nos[el[1]]]
                # verts_deformado = [nos_def[el[0]], nos_def[el[1]]]
                # codes = [Path.MOVETO, Path.LINETO]
                # elementos_barra.append(Path(verts_original, codes))
            elif len(el) > 2:
                elementos_poli_original.append(patches.Polygon(self.dados.nos[el], linewidth=0.7,
                                                               edgecolor=(0, 0, 0, 0.5), facecolor='None',
                                                               linestyle='--'))
                elementos_poli_deformado.append(patches.Polygon(nos_def[el], linewidth=0.7, edgecolor='black',
                                                                facecolor=(76 / 255, 191 / 255, 63 / 255, 0.4)))

        # Desenhar as cargas
        esc = min(dx, dy)
        dict_forcas = Estrutura.converter_vetor_forcas_em_dict(self.dados)
        dict_apoios = Estrutura.converter_vetor_apoios_em_dict(self.dados)

        for no in dict_forcas:
            for i, cg in enumerate(dict_forcas[no]):
                if cg != 0:
                    delta_x, delta_y = (0.1 * esc, 0) if i == 0 else (0, 0.1 * esc)
                    delta_x = -delta_x if i == 0 and cg < 0 else delta_x
                    delta_y = -delta_y if i == 1 and cg < 0 else delta_y

                    ax.add_patch(patches.Arrow(nos_def[no, 0], nos_def[no, 1], delta_x, delta_y, facecolor='blue',
                                               edgecolor='blue', width=0.01 * esc, linewidth=1))

        # Desenhar os apoios
        path_apoios = []
        for no in dict_apoios:
            for i, ap in enumerate(dict_apoios[no]):
                if ap != 0:
                    p0 = nos[no]
                    if i == 0:
                        p1 = np.array([p0[0] - 0.025 * esc, p0[1]])
                    else:
                        p1 = np.array([p0[0], p0[1] - 0.025 * esc])

                    path_apoios.append(path.Path([p0, p1],
                                                 [path.Path.MOVETO, path.Path.LINETO]))

        ax.add_collection(PathCollection(path_apoios, linewidths=2, edgecolors='red'))
        ax.add_collection(PatchCollection(elementos_poli_original, match_original=True))
        ax.add_collection(PatchCollection(elementos_poli_deformado, match_original=True))
        plt.axis('off')
        plt.grid(b=None)
        plt.title(f'Estrutura original deformada       escala: {escala}')
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
