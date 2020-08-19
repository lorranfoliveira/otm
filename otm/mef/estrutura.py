from otm.mef.elementos_finitos.elemento_poligonal import ElementoPoligonal
from loguru import logger
from typing import List
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
from otm.constantes import ARQUIVOS_DADOS_ZIP
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.path import Path
from matplotlib import patches
import os
from otm.leitura_arquivos import *
from scipy.sparse.csgraph import reverse_cuthill_mckee
from typing import Union
from julia import Main
import zipfile
import pathlib
from otm.mef.materiais import MaterialIsotropico

__all__ = ['Estrutura']


class Estrutura:
    """Implementa as propriedades de uma estrutura"""

    def __init__(self, arquivo: pathlib.Path, concreto: MaterialIsotropico, dict_cargas: dict, dict_apoios: dict,
                 espessura: float = 1):
        # dict_cargas_apoios
        self.arquivo = arquivo
        self.concreto = concreto
        self.espessura = espessura
        self.dict_forcas = dict_cargas
        self.dict_apoios = dict_apoios

        self.nos: np.ndarray = np.array([])
        self.vetor_elementos: List[np.ndarray] = []
        self.elementos: List[ElementoPoligonal] = []
        self.vetor_apoios: np.ndarray = np.array([])
        self.kelems = None

        # Interface Julia
        julia = Main
        julia.eval('include("julia_core/Deslocamentos.jl")')

    def carregar_e_salvar_dados(self):
        # A ordem das funções abaixo deve ser mantida
        self.ler_arquivo_entrada_dados()
        self.criar_elementos_poligonais()
        self.salvar_dados_estrutura()

        logger.debug(f'Elementos: {len(self.elementos)}, Nós: {len(self.nos)}, GL: {2 * len(self.nos)}')

    def salvar_vetor_forcas(self):
        """Salva o vetor de forças da estrutura no arquivo .zip"""
        arq = self.arquivo.parent.joinpath(ARQUIVOS_DADOS_ZIP[4])

        with zipfile.ZipFile(self.arquivo, 'a', compression=zipfile.ZIP_DEFLATED) as arq_zip:
            if arq.name not in arq_zip.namelist():

                logger.debug('Salvando o vetor de forças')

                np.save(str(arq), self.converter_dict_forcas_para_vetor_forcas())
                arq_zip.write(arq.name)
                os.remove(str(arq))
            else:
                logger.warning(f'O arquivo "{arq.name}" já existe em "{self.arquivo.name}" e não '
                               f'pode ser sobrescrito! Ele deve ser removido para ser substituído.')

    def salvar_vetor_apoios(self):
        """Salva o vetor de apoios da estrutura no arquivo .zip"""
        arq = self.arquivo.parent.joinpath(ARQUIVOS_DADOS_ZIP[6])

        with zipfile.ZipFile(self.arquivo, 'a', compression=zipfile.ZIP_DEFLATED) as arq_zip:
            if arq.name not in arq_zip.namelist():

                logger.debug('Salvando o vetor de apoios')

                np.save(str(arq), self.converter_dict_apoios_para_vetor_apoios())
                arq_zip.write(arq.name)
                os.remove(str(arq))
            else:
                logger.warning(f'O arquivo "{arq.name}" já existe em "{self.arquivo.name}" e não '
                               f'pode ser sobrescrito! Ele deve ser removido para ser substituído.')

    def ler_arquivo_entrada_dados(self):
        """Faz a leitura do arquivo de entrada de dados"""
        self.nos = ler_arquivo_entrada_dados_numpy(self.arquivo, 1)
        self.vetor_elementos = ler_arquivo_entrada_dados_numpy(self.arquivo, 0)

    def salvar_dados_estrutura(self):
        """Salva os dados da estrutura"""
        # Não alterar a ordem das funções abaixo
        self.salvar_vetor_forcas()
        self.salvar_vetor_apoios()

        self.vetor_apoios = ler_arquivo_entrada_dados_numpy(self.arquivo, 6)

        self.salvar_graus_liberdade_elementos()
        self.salvar_graus_liberdade_estrutura()
        self.salvar_matrizes_de_rigidez_elementos()
        self.salvar_volumes_elementos()
        self.salvar_dados_entrada_txt()

    def salvar_dados_entrada_txt(self):
        arq = self.arquivo.parent.joinpath(ARQUIVOS_DADOS_ZIP[12])

        with zipfile.ZipFile(self.arquivo, 'a', compression=zipfile.ZIP_DEFLATED) as arq_zip:
            if arq.name not in arq_zip.namelist():

                logger.debug('Salvando o arquivo de dados em texto')

                with open(str(arq), 'w') as arq_txt:
                    arq_txt.write(f'forcas = {self.dict_forcas}\n')
                    arq_txt.write(f'apoios = {self.dict_apoios}\n')
                    arq_txt.write(f'E = {self.concreto.ec}\n')
                    arq_txt.write(f'poisson = {self.concreto.nu}\n')
                    arq_txt.write(f'espessura = {self.espessura}\n')

                arq_zip.write(arq.name)
                os.remove(str(arq))
            else:
                logger.warning(f'O arquivo "{arq.name}" já existe em "{self.arquivo.name}" e não '
                               f'pode ser sobrescrito! Ele deve ser removido para ser substituído.')

    def criar_elementos_poligonais(self):
        logger.debug('Criando os elementos finitos poligonais')

        self.elementos = []
        for i, e in enumerate(self.vetor_elementos):
            # Verificar se o elemento é poligonal
            if len(e) > 2:
                el = ElementoPoligonal(self.nos[e], self.concreto, self.espessura, e)
                self.elementos.append(el)

    def num_nos(self) -> int:
        """Retorna o número de nós da estrutura"""
        return self.nos.shape[0]

    def graus_liberdade_estrutura(self) -> np.array:
        """Vetor que contém os parâmetros de conversão dos graus de liberdade da forma B para a forma A.
        Cada índice representa o valor do graus de liberdade na forma B. O valor do índice representa o grau
        de liberdade correspondente na forma A.
        """
        # Todos os graus de liberdade impedidos deverão assumir valor -1
        gls = np.full(self.num_nos() * 2, -1)

        for e in self.elementos:
            e_glb = e.graus_liberdade()
            e_gla = e_glb.copy()
            for i in range(e_glb.size):
                if (b := e_glb[i]) not in self.vetor_apoios:
                    e_gla[i] -= len(list(filter(lambda x: x < b, self.vetor_apoios)))
                else:
                    e_gla[i] = -1
            gls[e_glb] = e_gla.copy()

        return gls

    def salvar_graus_liberdade_estrutura(self):

        arq = self.arquivo.parent.joinpath(ARQUIVOS_DADOS_ZIP[9])

        with zipfile.ZipFile(self.arquivo, 'a', zipfile.ZIP_DEFLATED) as arq_zip:
            if arq.name not in arq_zip.namelist():
                logger.debug('Salvando os graus de liberdade da estrutura')

                glest = self.graus_liberdade_estrutura()
                np.save(str(arq), glest)
                arq_zip.write(arq.name)
                os.remove(str(arq))
            else:
                logger.warning(f'O arquivo "{arq.name}" já existe em "{self.arquivo.name}" e não '
                               f'pode ser sobrescrito! Ele deve ser removido para ser substituído.')

    def num_graus_liberdade(self) -> int:
        """Retorna o número de graus de liberdade da estrutura."""
        return len(self.nos) * 2

    def salvar_permutacao_rcm(self):
        """Salva a permutação feita pelo Reverse Cuthill Mckee"""
        # Interface Julia
        julia = Main
        julia.eval('include("julia_core/Deslocamentos.jl")')

        julia.kelems = ler_arquivo_entrada_dados_numpy(self.arquivo, 7)
        julia.gls_elementos = [i + 1 for i in self.graus_liberdade_elementos()]
        julia.gls_estrutura = [i + 1 if i != -1 else i for i in self.graus_liberdade_estrutura()]
        julia.apoios = self.vetor_apoios + 1

        julia.eval('dados = Dict("kelems" => kelems, "gls_elementos" => gls_elementos, '
                   '"gls_estrutura" => gls_estrutura, "apoios" => apoios)')

        linhas, colunas, termos = julia.eval('matriz_rigidez_estrutura(dados["kelems"], dados, true)')

        ngl = len(julia.gls_estrutura) - len(self.vetor_apoios)
        k = csr_matrix((termos, (linhas - 1, colunas - 1)), shape=(ngl, ngl))

        rcm = reverse_cuthill_mckee(k)
        arq = self.arquivo.parent.joinpath(ARQUIVOS_DADOS_ZIP[11])

        with zipfile.ZipFile(self.arquivo, 'a', zipfile.ZIP_DEFLATED) as arq_zip:
            if arq.name not in arq_zip.namelist():

                logger.debug('Salvando a permutação RCM')

                np.save(str(arq), rcm)
                arq_zip.write(arq.name)
                os.remove(str(arq))
            else:
                logger.warning(f'O arquivo "{arq.name}" já existe em "{self.arquivo.name}" e não '
                               f'pode ser sobrescrito! Ele deve ser removido para ser substituído.')

    def graus_liberdade_elementos(self):
        gles = []
        for e in self.elementos:
            gles.append(e.graus_liberdade())

        return gles

    def salvar_graus_liberdade_elementos(self):
        """Salva os graus de liberdade dos elementos no arquivo .zip"""
        arq = self.arquivo.parent.joinpath(ARQUIVOS_DADOS_ZIP[5])

        with zipfile.ZipFile(self.arquivo, 'a', zipfile.ZIP_DEFLATED) as arq_zip:
            if arq.name not in arq_zip.namelist():

                logger.debug('Salvando os graus de liberdade dos elementos')

                gles = self.graus_liberdade_elementos()

                np.savez(str(arq), *gles)
                arq_zip.write(arq.name)
                os.remove(str(arq))
            else:
                logger.warning(f'O arquivo "{arq.name}" já existe em "{self.arquivo.name}" e não '
                               f'pode ser sobrescrito! Ele deve ser removido para ser substituído.')

    def _volumes_elementos(self) -> np.ndarray:
        """Retorna os volumes dos elementos finitos"""
        volumes = np.zeros(len(self.elementos))
        for i in range(len(self.elementos)):
            volumes[i] = self.espessura * self.elementos[i].poligono().area

        return volumes

    def salvar_volumes_elementos(self):
        """Salva os graus de liberdade dos elementos no arquivo .zip"""
        arq = self.arquivo.parent.joinpath(ARQUIVOS_DADOS_ZIP[8])

        with zipfile.ZipFile(self.arquivo, 'a', zipfile.ZIP_DEFLATED) as arq_zip:
            if arq.name not in arq_zip.namelist():

                logger.debug('Salvando os volumes dos elementos')

                volumes = self._volumes_elementos()
                np.save(str(arq), volumes)
                arq_zip.write(arq.name)
                os.remove(str(arq))
            else:
                logger.warning(f'O arquivo "{arq.name}" já existe em "{self.arquivo.name}" e não '
                               f'pode ser sobrescrito! Ele deve ser removido para ser substituído.')

    def matrizes_rigidez_elementos(self) -> List[np.ndarray]:
        """Retorna as matrizes de rigidez dos elementos"""
        logger.debug('Calculando as matrizes de rigidez dos elementos')

        nel = len(self.elementos)
        kels = []
        c = 5
        for i, e in enumerate(self.elementos):
            if c <= (perc := (int(100 * (i + 1) / nel))):
                c += 5
                logger.debug(f'{perc}%')
            kels.append(e.matriz_rigidez())
        return kels

    def salvar_matrizes_de_rigidez_elementos(self):
        """Adiciona as matrizes de rigidez dos elementos no arquivo zip"""
        arq = self.arquivo.parent.joinpath(ARQUIVOS_DADOS_ZIP[7])

        with zipfile.ZipFile(self.arquivo, 'a', zipfile.ZIP_DEFLATED) as arq_zip:
            if arq.name not in arq_zip.namelist():

                logger.debug('Salvando as matrizes de rigidez dos elementos')

                ks_el = self.matrizes_rigidez_elementos()

                # Salvar as matrizes de rigidez
                np.savez(str(arq), *ks_el)
                arq_zip.write(arq.name)
                os.remove(str(arq))
            else:
                logger.warning(f'O arquivo "{arq.name}" já existe em "{self.arquivo.name}" e não '
                               f'pode ser sobrescrito! Ele deve ser removido para ser substituído.')

        # Salvar o padrão RCM
        self.salvar_permutacao_rcm()

    def converter_dict_apoios_para_vetor_apoios(self) -> np.ndarray:
        vet_apoios = []
        for no in self.dict_apoios:
            gls = ElementoPoligonal.id_no_para_grau_liberdade(no)
            for i, ap in enumerate(self.dict_apoios[no]):
                if ap == 1:
                    vet_apoios.append(gls[i])
                elif ap != 0:
                    raise ValueError(f'O valor não é aceito como restrição de deslocamento do nó {no}')
        return np.array(vet_apoios, dtype=int)

    def converter_dict_forcas_para_vetor_forcas(self) -> np.ndarray:
        cargas = np.zeros(self.num_graus_liberdade())
        for no in self.dict_forcas:
            gls = ElementoPoligonal.id_no_para_grau_liberdade(no)
            cargas[gls] = self.dict_forcas[no]
        return cargas

    @staticmethod
    def converter_vetor_forcas_em_dict(forcas) -> dict:
        """Converte um vetor de forças ou de apoios em uma relação do numpy.
        dic = {no: [dado_x, dado_y]}
        """
        dic = {}

        for i in range(0, forcas.shape[0], 2):
            v = forcas[[i, i + 1]]
            if any(vtmp for vtmp in v):
                no = int(i / 2)
                dic[no] = v.tolist()

        return dic

    @staticmethod
    def converter_vetor_apoios_em_dict(num_graus_liberdade, apoios) -> dict:
        dic = {}

        vetor_apoios_def = np.zeros(num_graus_liberdade, dtype=int)
        vetor_apoios_def[apoios] = 1

        for i in range(0, vetor_apoios_def.shape[0], 2):
            v = vetor_apoios_def[[i, i + 1]]
            if any(vtmp for vtmp in v):
                no = int(i / 2)
                dic[no] = v.tolist()

        return dic

    @staticmethod
    def deslocamentos_arquivo(arquivo: pathlib.Path) -> np.ndarray:
        """Retorna os deslocamentos da estrutura a partir da leitura do arquivo de entrada de dados"""
        # Interface Julia
        julia = Main
        julia.eval('include("julia_core/Deslocamentos.jl")')
        arqs = ARQUIVOS_DADOS_ZIP
        # Leitura Julia
        arqs_julia = [arqs[7], arqs[5], arqs[9], arqs[6], arqs[4], arqs[11]]
        with zipfile.ZipFile(arquivo, 'r') as arq_zip:
            for i in arqs_julia:
                arq_zip.extract(i)

        julia.eval(f'dados = ler_arquivos_entrada("{arqs[7]}", "{arqs[5]}", "{arqs[9]}", "{arqs[6]}", '
                   f'"{arqs[4]}", "{arqs[11]}")')

        for i in arqs_julia:
            os.remove(i)

        julia.eval('rho = ones(length(dados["kelems"]))')
        julia.eval('p = 1')

        return julia.eval(f'deslocamentos(dados["kelems"], dados)')

    @staticmethod
    def deslocamentos_por_no(u) -> np.ndarray:
        num_nos = int(u.shape[0] / 2)
        u_no = np.zeros((num_nos, 2))

        for n in range(num_nos):
            u_no[n] = u[ElementoPoligonal.id_no_para_grau_liberdade(n)]

        return u_no

    @staticmethod
    def posicao_nos_deformados(nos, u, multiplicador_deslocs=1) -> np.ndarray:
        """Retorna a posição dos nós deslocados levando-se em conta um fator multiplicador."""
        return nos + multiplicador_deslocs * Estrutura.deslocamentos_por_no(u)

    @staticmethod
    def plotar_estrutura_deformada(arquivo, multiplicador_deslocs=1):
        """Exibe a malha final gerada"""
        logger.debug('Plotando a estrutura deformada')

        # Leitura dos dados importantes
        nos = ler_arquivo_entrada_dados_numpy(arquivo, 1)
        poli = ler_arquivo_wkb_shapely(arquivo, 10)
        vetor_elementos = ler_arquivo_entrada_dados_numpy(arquivo, 0)
        vetor_forcas = ler_arquivo_entrada_dados_numpy(arquivo, 4)
        vetor_apoios = ler_arquivo_entrada_dados_numpy(arquivo, 6)

        # Deslocamentos
        u = Estrutura.deslocamentos_arquivo(arquivo)

        fig, ax = plt.subplots()
        win = plt.get_current_fig_manager()
        win.window.state('zoomed')
        ax.axis('equal')

        xmin, ymin, xmax, ymax = poli.bounds
        dx = xmax - xmin
        dy = ymax - ymin
        plt.xlim(xmin - 0.1 * dx, xmax + 0.1 * dx)
        plt.ylim(ymin - 0.1 * dy, ymax + 0.1 * dy)

        nos_def = Estrutura.posicao_nos_deformados(nos, u, multiplicador_deslocs)

        elementos_poli_original = []
        elementos_poli_deformado = []
        elementos_barra = []
        for el in vetor_elementos:
            codes = []
            verts_original = []
            verts_deformado = []
            if len(el) == 2:
                verts_original = [nos[el[0]], nos[el[1]]]
                # verts_deformado = [nos_def[el[0]], nos_def[el[1]]]
                codes = [Path.MOVETO, Path.LINETO]
                elementos_barra.append(Path(verts_original, codes))
            elif len(el) > 2:
                for i, v in enumerate((nos[j], nos_def[j]) for j in el):
                    verts_original.append(v[0])
                    verts_deformado.append(v[1])
                    if i == 0:
                        codes.append(Path.MOVETO)
                    else:
                        codes.append(Path.LINETO)

                verts_original.append(verts_original[0])
                verts_deformado.append(verts_deformado[0])
                codes.append(Path.CLOSEPOLY)
                elementos_poli_original.append(Path(verts_original, codes))
                elementos_poli_deformado.append(Path(verts_deformado, codes))

        # Desenhar as cargas
        esc = min(dx, dy)
        dict_forcas = Estrutura.converter_vetor_forcas_em_dict(vetor_forcas)
        dict_apoios = Estrutura.converter_vetor_apoios_em_dict(vetor_forcas.size, vetor_apoios)

        for no in dict_forcas:
            for i, cg in enumerate(dict_forcas[no]):
                if cg != 0:
                    delta_x, delta_y = (0.1 * esc, 0) if i == 0 else (0, 0.1 * esc)
                    delta_x = -delta_x if i == 0 and cg < 0 else delta_x
                    delta_y = -delta_y if i == 1 and cg < 0 else delta_y

                    ax.add_patch(
                        patches.Arrow(nos_def[no, 0], nos_def[no, 1], delta_x, delta_y, facecolor='blue',
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

                    path_apoios.append(Path([p0, p1], [Path.MOVETO, Path.LINETO]))

        # ax.add_collection(PathCollection(elementos_poli, linewidths=1, edgecolors='black', facecolors='#6fe8af'))
        ax.add_collection(PathCollection(elementos_poli_original, linewidths=0.7, edgecolors=(0, 0, 0, 0.5),
                                         facecolors=(0, 0, 0, 0), linestyles='--'))
        ax.add_collection(PathCollection(elementos_poli_deformado, linewidths=0.7, edgecolors='black',
                                         facecolors=(76 / 255, 191 / 255, 63 / 255, 0.4)))
        ax.add_collection(PathCollection(path_apoios, linewidths=2, edgecolors='red'))
        ax.add_collection(PathCollection(elementos_barra, linewidths=0.7, edgecolors='purple'))

        plt.axis('off')
        plt.grid(b=None)
        plt.show()
