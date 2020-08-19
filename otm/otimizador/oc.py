import numpy as np
from loguru import logger
from otm.manipulacao_arquivos import *
from matplotlib import pyplot as plt
from matplotlib.path import Path
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from matplotlib import patches
from matplotlib import cm
from otm.constantes import ARQUIVOS_DADOS_ZIP
from julia import Main
from scipy.spatial import KDTree
import os
from typing import Tuple
from zipfile import ZipFile


class OC:
    """Classe que implementa as características do problema de otimização pelo Optimality Criteria.
    Arquivos necessários: matrizes_elementos.npz, vetor_forcas.npz
    """
    METODO_HEAVISIDE = 0
    BETA_MAX = 150

    def __init__(self, arquivo_dados_entrada: str, x_inicial=0.5, p=3, rho_min=1e-3, rmin=0, tecnica_otimizacao=0,
                 esquema_projecao=0):
        """Se rmin == 0, a otimização será feita sem a aplicação do esquema de projeção

        Args:
            tecnica_otimizacao:
                0 -> Sem esquema de projeção.
                1 -> Com esquema de projeção linear direto.
                2 -> Com esquema de projeção não linear direto.

        """
        self.x_inicial: float = x_inicial
        self.arquivo = arquivo_dados_entrada
        self.p = p
        self.rho_min = rho_min
        self.rmin = rmin
        self.tecnica_otimizacao = tecnica_otimizacao
        self.esquema_projecao = esquema_projecao

        self.matrizes_rigidez_elementos = None
        self.vetor_forcas: np.ndarray = None
        self.vetor_apoios: np.ndarray = None
        self.graus_liberdade_elementos = None
        self.volumes_elementos_solidos = None
        self.limites_dominio = None
        self.vetor_elementos = None
        self.nos = None
        self.graus_liberdade_estrutura = None
        self.pesos_nos = None

        # Interface Julia
        self.julia = Main
        self.julia.eval('include("julia_core/Deslocamentos.jl")')

        self.ler_dados_entrada_arquivo()

        # Densidades dos nós
        self.x: np.ndarray = np.full(self.num_nos, self.x_inicial)
        # Densidades dos elementos
        self.rho: np.ndarray = np.full(self.num_elementos, self.x_inicial)

    @property
    def num_elementos(self) -> int:
        """Retorna o número de elementos finitos do problema"""
        return len(self.vetor_elementos)

    @property
    def num_nos(self) -> int:
        """Retorna o número de nós que possui a malha"""
        return self.nos.shape[0]

    def volume_estrutura(self) -> float:
        """Retorna o volume da estrutura"""
        return np.sum(self.x_inicial * self.volumes_elementos_solidos)

    def deslocamentos(self) -> np.ndarray:
        """Retorna os deslocamentos nodais em função das variáveis de projeto"""
        # Inserção das densidades relativas no problema
        return self.julia.eval(f'deslocamentos_simp(rho, p, dados)')

    def ler_dados_entrada_arquivo(self):
        """Faz a leitura do arquivo de entrada de dados"""
        # Leitura Python
        self.matrizes_rigidez_elementos = ler_arquivo_entrada_dados_numpy(self.arquivo, 7)
        self.vetor_forcas = ler_arquivo_entrada_dados_numpy(self.arquivo, 4)
        self.vetor_apoios = ler_arquivo_entrada_dados_numpy(self.arquivo, 6)
        self.graus_liberdade_elementos = ler_arquivo_entrada_dados_numpy(self.arquivo, 5)
        self.volumes_elementos_solidos = ler_arquivo_entrada_dados_numpy(self.arquivo, 8)
        self.limites_dominio = ler_arquivo_entrada_dados_numpy(self.arquivo, 3)
        self.vetor_elementos = ler_arquivo_entrada_dados_numpy(self.arquivo, 0)
        self.nos = ler_arquivo_entrada_dados_numpy(self.arquivo, 1)
        self.graus_liberdade_estrutura = ler_arquivo_entrada_dados_numpy(self.arquivo, 9)

        # Interface Julia
        self.julia.kelems = self.matrizes_rigidez_elementos
        self.julia.gls_elementos = [i + 1 for i in self.graus_liberdade_elementos]
        self.julia.gls_estrutura = [i + 1 if i != -1 else i for i in self.graus_liberdade_estrutura]
        self.julia.apoios = self.vetor_apoios + 1
        self.julia.forcas = self.vetor_forcas
        self.julia.rcm = ler_arquivo_entrada_dados_numpy(self.arquivo, 11) + 1

        self.julia.eval(f'dados = Dict("kelems" => kelems, "gls_elementos" => gls_elementos, '
                        f'"gls_estrutura" => gls_estrutura, "apoios" => apoios, "forcas" => forcas, '
                        f'"RCM" => rcm)')

        # Cálculo dos nós de influência nos elementos pelo esquema de projeção
        if self.tecnica_otimizacao != 0:
            with ZipFile(self.arquivo, 'a') as arq_zip:
                if (arq := ARQUIVOS_DADOS_ZIP[13]) not in arq_zip.namelist():
                    self.pesos_nos = self.calcular_funcoes_peso()
                    np.savez(arq, *self.pesos_nos)
                    arq_zip.write(arq)
                    os.remove(arq)
                    return

            self.pesos_nos = ler_arquivo_entrada_dados_numpy(self.arquivo, 13)

    def calcular_funcoes_peso(self):
        """Encontra os nós que se encontram dentro do raio de influência de cada elemento.
        Retorna um vetor onde cada índice corresponde a um elemento e é composto por uma matriz n x 2, onde n
        é o número de nós dentro da área de influência do elemento. A primeiro coluna do índice corresponde ao
        número do nó e a segunda à distância entre o centroide do elemento e o nó em questão.

        Se o número de nós capturados for menor que o número de nós que compõem o elemento, utiliza-se apenas
        os nós do elemento.
        """
        logger.debug(f'Calculando a influência dos nós sobre os elementos...')

        def w(r, rmin) -> float:
            if self.esquema_projecao == 0:
                # Função de projeção direta
                return (rmin - r) / rmin
            elif self.esquema_projecao == 1:
                # Função de projeção inversa
                return r / rmin

            # Vetorização da função

        vet_w = np.vectorize(w)

        kd_nos = KDTree(self.nos)
        conjunto_pesos = []

        c = 5
        for i, e in enumerate(self.vetor_elementos):
            if c <= (perc := (int(100 * (i + 1) / self.num_elementos))):
                c += 5
                logger.debug(f'{perc}%')
            # Nós do elemento
            nos_elem = self.nos[e]
            # Cálculo do centroide do elemento
            centroide = np.mean(nos_elem, axis=0)
            # Pontos que recebem a influência do elemento
            nos_influenciados = kd_nos.query_ball_point(centroide, self.rmin)
            # Um elemento receberá no mínimo a influência dos nós que o formam
            rmin_tmp = self.rmin
            if len(nos_influenciados) < e.shape[0]:
                rmin_tmp = np.max(np.linalg.norm(nos_elem - centroide, axis=1))
                nos_influenciados = kd_nos.query(nos_elem)[1]

            # Matriz contendo os números dos nós que influenciam as densidades de cada elemento na primeira
            # coluna e os pesos na segunda coluna. Os valores da primeira coluna DEVEM ser convertidos para
            # inteiros antes de serem utilizados.

            # Distância entre os nós influenciados e o centroide do elemento de referência
            dists = np.linalg.norm(self.nos[nos_influenciados] - centroide, axis=1)
            # Aplicação da função de projeção para o cálculo dos pesos
            pesos = vet_w(dists, rmin_tmp)

            conjunto_pesos.append(np.array([nos_influenciados, pesos]).T)

        return conjunto_pesos

    def calcular_densidades_elementos(self, beta) -> np.ndarray:
        """Retorna a densidade relativa do elemento a partir de sua identificação"""
        rho = np.zeros(self.num_elementos)

        for i in range(self.num_elementos):
            ids_nos = self.pesos_nos[i][:, 0].astype(int)
            pesos = self.pesos_nos[i][:, 1]
            mi = (self.x[ids_nos] @ pesos) / np.sum(pesos)
            # Adição da não linearidade
            rho[i] = self.heaviside(mi, beta, OC.METODO_HEAVISIDE)

        return rho

    def flexibilidade_media(self, u: np.ndarray) -> float:
        """Retorna a flexibilidade média da estrutura em função das variáveis de projeto"""
        return self.vetor_forcas @ u

    def sensibilidades_sem_filtro(self, u: np.ndarray) -> np.ndarray:
        """Calcula as sensibilidades dos elementos"""
        kelems = self.matrizes_rigidez_elementos

        sens = np.zeros(self.num_elementos)
        for i, gl in enumerate(self.graus_liberdade_elementos):
            sens[i] = (-self.p * self.rho[i] ** (self.p - 1)) * (u[gl] @ kelems[i] @ u[gl])

        return sens

    @staticmethod
    def heaviside(rho, beta, metodo=0, derivada=False) -> float:
        """Retorna o valor da função de projeção Heaviside suavizada.

        Args:
            rho: Densidade do elemento.
            beta: Parâmetro que define o nível de suavidade da função Heaviside.
            metodo: Método utilizado para a representação da função Heaviside suavizda.
                0=Guest (2004), 1=Sigmund (2007) e 2=Xu et al. (2010).
            derivada: True para o cálculo da derivada da função e False para o cálculo apenas da função.

        Raises:
            ValueError:
                Se o valor passado para `método` não estiver em [0, 1, 2].
        """
        # Método de Guest (2004)
        if metodo == 0:
            if derivada:
                return beta * np.exp(-beta * rho) + np.exp(-beta)
            else:
                return 1 - np.exp(-beta * rho) + rho * np.exp(-beta)
        # Método de Sigmund (2007)
        elif metodo == 1:
            if derivada:
                return beta * np.exp(beta * (rho - 1)) + np.exp(-beta)
            else:
                return np.exp(-beta * (1 - rho)) - (1 - rho) * np.exp(-beta)
        # Método de Xu et al. (2010)
        elif metodo == 2:
            if derivada:
                if rho <= 0.5:
                    return (beta * np.exp(2 * beta * rho) + 1) * np.exp(-beta)
                else:
                    return (beta * np.exp(2 * beta) + np.exp(2 * beta * rho)) * np.exp(-2 * beta * rho - beta)
            else:
                if rho <= 0.5:
                    return 1 / 2 * (2 * rho + np.exp(2 * beta * rho) - 1) * np.exp(-beta)
                else:
                    return 1 / 2 * (
                        (2 * rho + 2 * np.exp(beta) - 1) * np.exp(2 * beta * rho) - np.exp(2 * beta)) * np.exp(
                        -2 * beta * rho - beta)
        else:
            raise ValueError(f'O tipo de função Heaviside "{metodo}" não é válido!')

    def sensibilidades_esquema_projecao(self, u: np.ndarray, beta) -> tuple:
        """Calcula as sensibilidades da função objetivo e da restrição de volume.

        Returns:
            sens_fo: Vetor contendo as sensibilidades da função objetivo.
            sens_vol: Vetor contendo as sensibilidades da restrição de volume.
        """
        kelems = self.matrizes_rigidez_elementos
        # Sensibilidade da função objetivo
        sens_fo = np.zeros(self.num_nos)
        # Sensibilidade da restrição de volume
        sens_vol = np.zeros(self.num_nos)
        vols = self.volumes_elementos_solidos

        for i, gl in enumerate(self.graus_liberdade_elementos):
            # Parcela da sensibilidade devida à densidade do elemento
            sens_el = (-self.p * self.rho[i] ** (self.p - 1)) * (u[gl] @ kelems[i] @ u[gl])
            # Desmembramento da matriz de pesos
            ids_nos = self.pesos_nos[i][:, 0].astype(int)
            pesos = self.pesos_nos[i][:, 1]
            soma_pesos = np.sum(pesos)
            # Não linearidade
            mi = (self.x[ids_nos] @ pesos) / soma_pesos
            diff_rhoe_rhon = self.heaviside(mi, beta, OC.METODO_HEAVISIDE, True) * pesos / soma_pesos
            sens_fo[ids_nos] += sens_el * diff_rhoe_rhon
            sens_vol[ids_nos] += vols[i] * diff_rhoe_rhon

        return sens_fo, sens_vol

    def percentual_densidades_intermediarias(self) -> float:
        """Retorna o percentual de densidades intermediárias da topologia"""
        return 100 * sum(4 * rho * (1 - rho) for rho in self.rho) / len(self.rho)

    def atualizar_x(self, u: np.ndarray, beta=0) -> Tuple[np.ndarray, np.ndarray]:
        """Atualiza as variáveis de projeto (densidades nodais)"""
        vol_inicial = self.volume_estrutura()
        # Volume da estrutura em função das densidades correntes para os elementos
        vol_atual = self.volumes_elementos_solidos @ self.rho
        # Sensibilidades da função objetivo e da restrição de volume
        if self.tecnica_otimizacao != 0:
            sens_fo, sens_vol = self.sensibilidades_esquema_projecao(u, beta)
            x = self.x
        else:
            sens_fo = self.sensibilidades_sem_filtro(u)
            sens_vol = self.volumes_elementos_solidos * self.x_inicial
            x = self.rho

        eta = 0.5
        bm = -sens_fo / sens_vol

        l1 = 0
        l2 = 1e6
        move = 0.2
        n = len(x)
        x_novo = np.zeros(n)

        if (self.tecnica_otimizacao != 0) and (beta != 0):
            x_min = -1 / beta * np.log(1 - self.rho_min)
        else:
            x_min = self.rho_min
        x_max = 1

        while (l2 - l1) > 1e-4:
            lmid = (l2 + l1) / 2
            be = bm / lmid
            t1 = x_min + (x - x_min) * be ** eta
            t2 = np.maximum(x_min, x - move)
            t3 = np.minimum(x_max, x + move)

            for i in range(n):
                if t1[i] <= t2[i]:
                    x_novo[i] = t2[i]
                elif t2[i] < t1[i] < t3[i]:
                    x_novo[i] = t1[i]
                else:
                    x_novo[i] = t3[i]

            # Restrição de volume
            if ((vol_atual - vol_inicial) + sens_vol @ (x_novo - x)) > 0:
                l1 = lmid
            else:
                l2 = lmid

        return x_novo

    def otimizar(self, erro_max=1e-1, passo=0.5, num_max_iteracoes=50):
        """Aplica o processo de otimização aos dados da estrutura.

        Se o passo for -1, apenas um valor de p será rodado
        """
        logger.info('Iniciando a otimização da estrutura')
        # Vetor de coeficientes de penalização
        if passo != -1:
            ps = np.arange(1, self.p + 0.1, passo)
        else:
            ps = [self.p]
        # Último vetor de deslocamentos
        u_antigo = np.ones(self.num_nos * 2)
        # Contador global
        t = 0
        erro_u = 100

        def otm(p, beta):
            nonlocal u_antigo, t, erro_u

            logger.info(f'{p=}\t {beta=}\n')

            self.julia.p = self.p = p

            di_ant = 100

            for c in np.arange(1, num_max_iteracoes + 1):
                t += 1

                if self.tecnica_otimizacao != 0:
                    self.julia.rho = self.rho = self.calcular_densidades_elementos(beta)
                    u = self.deslocamentos()
                    self.x = self.atualizar_x(u, beta)
                else:
                    self.julia.rho = self.rho
                    u = self.deslocamentos()
                    self.rho = self.atualizar_x(u)

                # Cálculo do erro
                di = self.percentual_densidades_intermediarias()
                # Erro da continuação em beta
                erro_di = 100 * abs((di - di_ant) / di_ant)
                di_ant = di

                # Erro devido aos deslocamentos
                if c > 1:
                    norm1 = np.linalg.norm(u_antigo)
                    norm2 = np.linalg.norm(u)
                    erro_u = 100 * abs((norm1 - norm2) / norm1)
                u_antigo = u.copy()

                vols = self.volumes_elementos_solidos
                logger.info(f'i: {t}-{c}\t '
                            f'p: {p}\t '
                            f'beta: {beta:.2f}\t '
                            f'fo: {self.flexibilidade_media(u):.2f}\t '
                            f'vol: {(self.rho @ vols) / np.sum(vols):.3f}%\t '
                            f'erro_u: {erro_u:.5f}%\t '
                            f'erro_di: {erro_di:.5f}%\t'
                            f'di: {di:.3f}%')

                if c > (10 if di > 5 else 5):
                    if beta != 0:
                        if erro_di <= erro_max:
                            logger.info(f'Convergência alcançada pelas densidades intermediárias!')
                            break
                    else:
                        if erro_u <= erro_max:
                            logger.info(f'Convergência alcançada pelos deslocamentos!')
                            break

        # Método da continuidade
        # Continuidade no coeficiente de penalização
        for j, pi in enumerate(ps):
            otm(pi, 0)

        # Continuidade em beta
        if self.tecnica_otimizacao == 2:
            beta_i = (1 / 3)  # 1.5 * 1/3 = 0.5
            for i in range(num_max_iteracoes):
                dens = self.percentual_densidades_intermediarias()
                if beta_i < OC.BETA_MAX:
                    if dens >= 5:
                        beta_i = min(1.5 * beta_i, OC.BETA_MAX)
                    elif 1 <= dens < 5:
                        beta_i = min(beta_i + 5, OC.BETA_MAX)
                    else:
                        break
                    otm(self.p, beta_i)
                else:
                    break

    def plotar_estrutura_otimizada(self, tipo_cmap: str = 'jet'):
        """Exibe a malha final gerada. cmad jet ou binary"""
        logger.info('Criando o desenho da malha final')

        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.family'] = 'Calibri'

        elementos = self.vetor_elementos

        fig, ax = plt.subplots()
        win = plt.get_current_fig_manager()
        win.window.state('zoomed')
        ax.axis('equal')

        xmin, ymin, xmax, ymax = ler_arquivo_wkb_shapely(self.arquivo).bounds
        dx = xmax - xmin
        dy = ymax - ymin
        plt.xlim(xmin - 0.1 * dx, xmax + 0.1 * dx)
        plt.ylim(ymin - 0.1 * dy, ymax + 0.1 * dy)

        elementos_poli = []
        for j, el in enumerate(elementos):
            if tipo_cmap == 'jet':
                elementos_poli.append(patches.Polygon(self.nos[el], linewidth=0, fill=True,
                                                      facecolor=cm.jet(self.rho[j])))
            else:
                elementos_poli.append(patches.Polygon(self.nos[el], linewidth=0, fill=True,
                                                      facecolor=cm.binary(self.rho[j])))

        # Adicionar marcador do diâmetro mínimo dos elementos
        path_diam_verts = [[xmax - self.rmin * 2 - 0.01 * dx, ymax - 0.01 * dx],
                           [xmax - 0.01 * dx, ymax - 0.01 * dx]]
        path_diam_codes = [Path.MOVETO, Path.LINETO]
        path_diam = Path(path_diam_verts, path_diam_codes)
        ax.add_patch(patches.PathPatch(path_diam, linewidth=2, color='magenta'))

        ax.add_collection(PatchCollection(elementos_poli, match_original=True, antialiased=False))
        # ax.add_collection(PathCollection(elementos_barra, linewidths=0.7, edgecolors='purple'))
        plt.axis('off')
        plt.grid(b=None)

        # Título
        # Fixos
        di = f'Di: {self.percentual_densidades_intermediarias():.2f}%'
        els = f'NumElems: {self.num_elementos}'
        vf = f'vol: {self.x_inicial}%'
        # Variáveis
        rmin = ''
        tecnica_otm = 'Técnica: '

        if self.tecnica_otimizacao == 0:
            tecnica_otm += 'Sem filtro'
        elif self.tecnica_otimizacao == 1:
            rmin = f'Rmin: {self.rmin}'
            tecnica_otm += 'Linear '
            if self.esquema_projecao == 0:
                tecnica_otm += 'Direta'
            else:
                tecnica_otm += 'Inversa'
        else:
            rmin = f'Rmin: {self.rmin}'
            tecnica_otm += 'Heaviside '
            if self.esquema_projecao == 0:
                tecnica_otm += 'Direta'
            else:
                tecnica_otm += 'Inversa'

        plt.title(f'{tecnica_otm}     {els}     {vf}     {di}    {rmin}')

        plt.show()
