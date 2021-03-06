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
from matplotlib.animation import FuncAnimation, writers
import matplotlib as mpl

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
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.family'] = 'Calibri'

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

        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.family'] = 'Calibri'

        # Plotagem.
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
        elementos_barra = []
        for el in self.dados.elementos:
            if len(el) == 2:
                verts = [self.dados.nos[el[0]], self.dados.nos[el[1]]]
                codes = [path.Path.MOVETO, path.Path.LINETO]
                elementos_barra.append(patches.PathPatch(path.Path(verts, codes), linewidth=0.7, edgecolor='red'))
            elif len(el) > 2:
                elementos_poli.append(patches.Polygon(self.dados.nos[el], linewidth=0.5, facecolor='None',
                                                      edgecolor='black'))

        ax.add_collection(PatchCollection(elementos_poli, match_original=True))
        ax.add_collection(PatchCollection(elementos_barra, match_original=True))

        # Enumerar os pontos
        if exibir_numeracao_nos:
            for i, v in enumerate(self.dados.nos):
                ax.text(v[0], v[1], f'{i}', ha="center", va="center", size=tamanho_numeracao, color='b')

        # logger.info(f'Salvando a malha final em "{arquivo_entrada_dados.replace("zip", "svg")}"')
        plt.axis('off')
        plt.grid(b=None)
        plt.show()
        # plt.savefig('malha.pdf')

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
        elementos_barra = []
        for el in vetor_elementos:
            if len(el) == 2:
                verts = [nos_def[el[0]], nos_def[el[1]]]
                codes = [path.Path.MOVETO, path.Path.LINETO]
                elementos_barra.append(patches.PathPatch(path.Path(verts, codes),
                                                         linewidth=1, edgecolor='purple'))
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
        ax.add_collection(PatchCollection(elementos_barra, match_original=True))
        plt.axis('off')
        plt.grid(b=None)
        plt.title(f'Estrutura original deformada       escala: {escala}')
        plt.show()

    def plotar_estrutura_otimizada(self, tecnica_otimizacao: int, rmin: float = 0,
                                   tipo_cmap: str = 'binary', visualizar_areas_barras=False):
        """Exibe a malha final gerada. cmad jet ou binary"""
        logger.info('Criando o desenho da malha final')

        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.family'] = 'Calibri'

        # Resultados finais
        rho_final = self.dados.rhos_iteracao_final()
        results_gerais_finais = self.dados.resultados_gerais_iteracao_final()

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
        elementos_barra = []

        x_bar_max = 0
        if self.dados.tem_barras():
            x_bar_max = max(rho_final[self.dados.num_elementos_poli::])

        for j, el in enumerate(self.dados.elementos):
            if j < self.dados.num_elementos_poli:
                if tipo_cmap == 'jet':
                    elementos_poli.append(patches.Polygon(self.dados.nos[el], linewidth=0, fill=True,
                                                          facecolor=cm.jet(rho_final[j])))
                else:
                    elementos_poli.append(patches.Polygon(self.dados.nos[el], linewidth=0, fill=True,
                                                          facecolor=cm.binary(rho_final[j])))
            else:
                verts = [self.dados.nos[el[0]], self.dados.nos[el[1]]]
                codes = [path.Path.MOVETO, path.Path.LINETO]

                rho = 15 * rho_final[j] / x_bar_max
                # rho = 6 if rho_final[j] > 0 else 0

                if rho > 0:
                    if tipo_cmap == 'jet':
                        elementos_barra.append(patches.PathPatch(path.Path(verts, codes),
                                                                 linewidth=rho, edgecolor='black'))
                    else:
                        elementos_barra.append(patches.PathPatch(path.Path(verts, codes),
                                                                 linewidth=rho, edgecolor='red'))
        # Enumerar os pontos
        if self.dados.tem_barras():
            if visualizar_areas_barras:
                for i in range(self.dados.num_elementos_poli, self.dados.num_elementos):
                    if rho_final[i] > 0:
                        # Centro da barra
                        nos_barra_i = self.dados.nos[self.dados.elementos[i]]
                        c = (nos_barra_i[0] + nos_barra_i[1]) / 2

                        cor = 'white' if tipo_cmap == 'jet' else 'blue'
                        ax.text(c[0], c[1], f'{rho_final[i] / x_bar_max:.2E}', ha="center", va="center",
                                size=0.05 * min(dx, dy), color=cor)

        # Desenhar o domínio do desenho
        # contorno = self.dados.poligono_dominio_estendido.boundary.coords[:]
        # linhas_cont = []
        # for lin in contorno:
        #     linhas_cont.append(patches.PathPatch(path.Path(lin, [path.Path.MOVETO, path.Path.LINETO]),
        #                                          linewidth=1, edgecolor='black'))

        # Adicionar marcador do diâmetro mínimo dos elementos
        path_diam_verts = [[xmax - rmin * 2 - 0.01 * dx, ymax - 0.01 * dx],
                           [xmax - 0.01 * dx, ymax - 0.01 * dx]]
        path_diam_codes = [path.Path.MOVETO, path.Path.LINETO]
        path_diam = path.Path(path_diam_verts, path_diam_codes)
        ax.add_patch(patches.PathPatch(path_diam, linewidth=2, color='magenta'))

        ax.add_collection(PatchCollection(elementos_poli, match_original=True))

        if self.dados.tem_barras():
            ax.add_collection(PatchCollection(elementos_barra, match_original=True))
        # ax.add_collection(PatchCollection(linhas_cont, match_original=True))
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

    def plotar_animacao_otimizacao(self, tipo_cmap: str = 'binary', salvar=True):
        """Exibe a malha final gerada. cmad jet ou binary"""
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.family'] = 'Calibri'
        mpl.rcParams['animation.ffmpeg_path'] = r'C:\Users\Lorran\Documents\Mestrado\otm\ffmpeg\bin\ffmpeg.exe'
        video_anim = r'C:\Users\Lorran\Documents\Mestrado\otm\animacao.mp4'

        # Resultados finais
        rhos = self.dados.resultados_rho

        fig, ax = plt.subplots()
        win = plt.get_current_fig_manager()
        win.window.state('zoomed')

        def frame(i):
            nonlocal rhos, ax

            ax.clear()
            ax.axis('equal')

            rhos_i = rhos[i]

            elementos_poli = []
            elementos_barra = []

            x_bar_max = 0
            if self.dados.tem_barras():
                x_bar_max = max(rhos_i[self.dados.num_elementos_poli::])

            for j, el in enumerate(self.dados.elementos):
                if j < self.dados.num_elementos_poli:
                    if tipo_cmap == 'jet':
                        elementos_poli.append(patches.Polygon(self.dados.nos[el], linewidth=0, fill=True,
                                                              facecolor=cm.jet(rhos_i[j])))
                    else:
                        elementos_poli.append(patches.Polygon(self.dados.nos[el], linewidth=0, fill=True,
                                                              facecolor=cm.binary(rhos_i[j])))
                else:
                    verts = [self.dados.nos[el[0]], self.dados.nos[el[1]]]
                    codes = [path.Path.MOVETO, path.Path.LINETO]

                    rho = 5 * rhos_i[j] / x_bar_max
                    # rho = 3 if rho_final[j] > 0 else 0

                    if rho > 0:
                        if tipo_cmap == 'jet':
                            elementos_barra.append(patches.PathPatch(path.Path(verts, codes),
                                                                     linewidth=rho, edgecolor='black'))
                        else:
                            elementos_barra.append(patches.PathPatch(path.Path(verts, codes),
                                                                     linewidth=rho, edgecolor='red'))

            ax.add_collection(PatchCollection(elementos_poli, match_original=True))

            if self.dados.tem_barras():
                ax.add_collection(PatchCollection(elementos_barra, match_original=True))

            return ax

        animation = FuncAnimation(fig, func=frame, frames=len(rhos), interval=10)

        xmin, ymin, xmax, ymax = self.dados.poligono_dominio_estendido.bounds
        dx = xmax - xmin
        dy = ymax - ymin
        plt.xlim(xmin - 0.1 * dx, xmax + 0.1 * dx)
        plt.ylim(ymin - 0.1 * dy, ymax + 0.1 * dy)
        # plt.axis('off')
        # plt.grid(b=None)

        if salvar:
            writer = writers['ffmpeg'](fps=10)
            animation.save(video_anim, writer=writer)
        else:
            plt.show()

    def plotar_tensoes(self):
        """Exibe a malha final gerada. cmad jet ou binary"""
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.family'] = 'Calibri'

        # Resultados finais
        tensoes = self.dados.ler_arquivo_entrada_dados_numpy(22)
        tensoes_norm = np.full(len(tensoes), 0.5)
        rho_final = self.dados.rhos_iteracao_final()

        for i, rho_i in enumerate(rho_final):
            if i < self.dados.num_elementos_poli:
                if abs(rho_i) > 1e-9:
                    if tensoes[i] >= 0:
                        tensoes_norm[i] = 1
                    else:
                        tensoes_norm[i] = 0
            else:
                if tensoes[i] >= 0:
                    tensoes_norm[i] = 1
                else:
                    tensoes_norm[i] = 0

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
        elementos_barra = []
        x_bar_max = max(rho_final[self.dados.num_elementos_poli::]) if self.dados.tem_barras() else 0

        for j, el in enumerate(self.dados.elementos):
            if j < self.dados.num_elementos_poli:
                elementos_poli.append(patches.Polygon(self.dados.nos[el], linewidth=0, fill=True,
                                                      facecolor=cm.seismic(tensoes_norm[j])))
            else:
                verts = [self.dados.nos[el[0]], self.dados.nos[el[1]]]
                codes = [path.Path.MOVETO, path.Path.LINETO]

                rho = 15 * rho_final[j] / x_bar_max
                # rho = 3 if rho_final[j] > 0 else 0

                if rho > 0:
                    elementos_barra.append(patches.PathPatch(path.Path(verts, codes),
                                                             linewidth=rho, edgecolor=cm.seismic(tensoes_norm[j])))

        ax.add_collection(PatchCollection(elementos_poli, match_original=True))
        ax.add_collection(PatchCollection(elementos_barra, match_original=True))

        plt.axis('off')
        plt.grid(b=None)

        plt.title(f'Tensões nos elementos')
        plt.show()
