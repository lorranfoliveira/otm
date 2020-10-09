from otm.mef.elementos import ElementoPoligonal, ElementoBarra
from loguru import logger
from typing import List, Optional, Union
from scipy.sparse import csr_matrix
import os
import numpy as np
from otm.constantes import ARQUIVOS_DADOS_ZIP
from scipy.sparse.csgraph import reverse_cuthill_mckee
from otm.dados import Dados
from julia import Main
import shelve

__all__ = ['Estrutura']


class Estrutura:
    """Implementa as propriedades de uma estrutura."""

    def __init__(self, dados: Dados, dict_cargas: dict, dict_apoios: dict, espessura: float = 1):
        """Construtor.

        Args:
            dados: Objeto que intermedia o acesso e a gravação dos dados do arquivo `.zip`.
            dict_cargas: Dicionário contendo a numeração do nó na chave e uma tupla com as cargas em x e y
                no nó. {num_no: (carga_x, carga_y)}.
            dict_apoios: Dicionário contendo a numeração do nó na chave e uma tupla contendo o tipo de
                restrição do nó. Se a restrição for 1, o grau de liberdade estará impedido. Se for 0, o grau
                de liberdade estará livre. {num_no: (deslocabilidade_x, deslocabilidade_y)}.
            espessura: Espessura dos elementos da malha bidimensional.
        """
        self.dados = dados
        self.espessura = espessura
        self.dict_forcas = dict_cargas
        self.dict_apoios = dict_apoios

        self.elementos: List[Union[ElementoPoligonal, ElementoBarra]] = []

        # Interface Julia
        julia = Main
        julia.eval('include("julia_core/Deslocamentos.jl")')

    def salvar_dados_estrutura(self):
        """Salva os dados da estrutura necessários para a análise estrutural."""
        # Não alterar a ordem das funções abaixo
        self.criar_elementos()
        # Vetor de forças.
        self.dados.salvar_arquivo_numpy(self.criar_vetor_forcas(), 4)
        # Vetor de apoios.
        self.dados.salvar_arquivo_numpy(self.criar_vetor_apoios(), 6)
        # Graus de liberdade por elemento.
        self.dados.salvar_arquivo_numpy(self.graus_liberdade_elementos(), 5)
        # Graus de liberdade da estrutura.
        self.dados.salvar_arquivo_numpy(self.graus_liberdade_estrutura(), 9)
        # Salvar matrize cinemáticas nodais na origem dos elementos.
        # TODO introduzir um verificador para evitar que dados já salvos sejam recalculados.
        self.dados.salvar_arquivo_numpy(self.matrizes_b_origem(), 18)
        # Salvar os dados constantes para a integração numérica em função da matriz
        # constitutiva elástica.
        self.salvar_matrizes_b_pontos_integracao()
        # Salvar matrizes de rigidez dos elementos de barra com o módulo de elasticidade e a área unitários.
        self.dados.salvar_arquivo_numpy(self.matrizes_rigidez_barras(), 20)
        # Volumes dos elementos sólidos.
        self.dados.salvar_arquivo_numpy(self.volumes_elementos_poligonais(), 8)
        # Comprimentos das barras.
        self.dados.salvar_arquivo_numpy(self.comprimentos_barras(), 21)
        # Permutação RCM.
        self.salvar_permutacao_rcm()
        # Salvar deslocamentos nodais da estrutura original sólida.
        self.salvar_deslocamentos_estrutura_original()

        logger.debug(f'Elementos: {len(self.dados.elementos)}, Nós: {len(self.dados.nos)}, '
                     f'GL: {2 * len(self.dados.nos)}')

    def criar_elementos(self):
        """Cria os objetos que representam os elementos finitos poligonais."""
        self.elementos = []
        nos = self.dados.nos
        for i, e in enumerate(self.dados.elementos):
            # Verificar se o elemento é poligonal ou de barra.
            if len(e) > 2:
                el = ElementoPoligonal(nos[e], self.dados.concreto, self.espessura, e)
            else:
                el = ElementoBarra(nos[e], self.dados.aco, e)
            self.elementos.append(el)

    def graus_liberdade_estrutura(self) -> np.array:
        """Vetor que contém os parâmetros de conversão dos graus de liberdade da forma B para a forma A.
        Forma A: Numeração sequencial que CONSIDERA os graus de liberdade impedidos.
        Forma B: Numeração sequencial que NÃO CONSIDERA os graus de liberdade impedidos. Neste caso, os
            graus de liberdade impedidos possuem valor -1.
        """
        # Todos os graus de liberdade impedidos deverão assumir valor -1
        gls = np.full(self.dados.num_nos() * 2, -1)
        apoios = self.dados.apoios

        for e in self.elementos:
            e_glb = e.graus_liberdade()
            e_gla = e_glb.copy()
            for i in range(e_glb.size):
                if (b := e_glb[i]) not in apoios:
                    e_gla[i] -= len(list(filter(lambda x: x < b, apoios)))
                else:
                    e_gla[i] = -1
            gls[e_glb] = e_gla.copy()

        return gls

    def matrizes_b_origem(self) -> List[np.ndarray]:
        """Retorna as matrizes cinemáticas nodais calculadas na origem de cada elemento finito poligonal."""
        mb = []

        for i in range(self.dados.num_elementos_poli):
            mb.append(self.elementos[i].matriz_b_origem())

        return mb

    def salvar_permutacao_rcm(self):
        """Salva a permutação feita pelo algoritmo de redução da banda da matriz de rigidez Reverse Cuthill Mckee."""
        # Interface Julia
        julia = Main
        julia.eval('include("julia_core/Deslocamentos.jl")')

        julia.kelems = self.matrizes_rigidez_elementos_poligonais(self.dados) + self.matrizes_rigidez_barras()
        julia.gls_elementos = [i + 1 for i in self.dados.graus_liberdade_elementos]
        julia.gls_estrutura = [i + 1 if i != -1 else i for i in self.dados.graus_liberdade_estrutura]
        julia.apoios = self.dados.apoios + 1

        julia.eval('dados = Dict("kelems" => kelems, "gls_elementos" => gls_elementos, '
                   '"gls_estrutura" => gls_estrutura, "apoios" => apoios)')

        linhas, colunas, termos = julia.eval('matriz_rigidez_estrutura(kelems, dados, true)')

        # Número de graus de liberdade livres.
        ngl = self.dados.num_graus_liberdade() - len(self.dados.apoios)

        k = csr_matrix((termos, (linhas - 1, colunas - 1)), shape=(ngl, ngl))
        rcm = reverse_cuthill_mckee(k)

        # Salvar o vetor rcm.
        self.dados.salvar_arquivo_numpy(rcm, 11)

    def salvar_matrizes_b_pontos_integracao(self):
        """Salva as matrizes b calculadas em cada ponto de integração em um arquivo .dat com o shelve."""
        logger.debug('Calculando as matrizes de cinemáticas nodais dos elementos...')

        nome_arq = ARQUIVOS_DADOS_ZIP[19]
        dados = []
        # Número de elementos poligonais.
        nels_poli = self.dados.num_elementos_poli

        c = 5
        for i in range(nels_poli):
            if c <= (perc := (int(100 * (i + 1) / nels_poli))):
                c += 5
                logger.info(f'{perc}%')
            dados.append(self.elementos[i].matrizes_b_pontos_integracao())

        with shelve.open(nome_arq) as arq:
            arq['0'] = dados

        self.dados.salvar_arquivo_generico_em_zip(f'{nome_arq}.dat')
        self.dados.salvar_arquivo_generico_em_zip(f'{nome_arq}.dir')
        os.remove(f'{nome_arq}.bak')

    def graus_liberdade_elementos(self) -> List[np.ndarray]:
        """Retorna uma lista contendo vetores com os graus de liberdade de cada elemento da malha."""
        return [e.graus_liberdade() for e in self.elementos]

    def volumes_elementos_poligonais(self) -> np.ndarray:
        """Retorna um vetor com os volumes dos elementos finitos."""
        return np.array([self.espessura * el.poligono().area for el in self.elementos[:self.dados.num_elementos_poli:]])

    def comprimentos_barras(self):
        """Retorna um vetor que contém os comprimentos dos elementos de barra."""
        return np.array([e.comprimento() for e in self.elementos[self.dados.num_elementos_poli::]])

    def matrizes_rigidez_elementos(self) -> List[np.ndarray]:
        """Retorna as matrizes de rigidez dos elementos."""
        logger.debug('Calculando as matrizes de rigidez dos elementos...')

        nel = len(self.elementos)
        kels = []
        c = 5
        for i, e in enumerate(self.elementos):
            if c <= (perc := (int(100 * (i + 1) / nel))):
                c += 5
                logger.debug(f'{perc}%')
            kels.append(e.matriz_rigidez())
        return kels

    @staticmethod
    def matrizes_rigidez_elementos_poligonais(dados: Dados, tensoes: Optional[np.ndarray] = None,
                                              deformacoes: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """Atualiza as matrizes de rigidez dos elementos em função das tensões atuantes em cada um.

        Args:
            dados:
            tensoes: Tensões atuantes nos elementos.
            deformacoes:
        """
        # Matrizes b e pesos.
        mb_pesos = dados.matrizes_b_pontos_integracao
        # Matrizes de rigidez dos elementos.
        kelems = []
        # Elementos poligonais.
        for i in range(dados.num_elementos_poli):
            # Número de nós do elemento.
            n = len(dados.elementos[i])
            kel = np.zeros((2 * n, 2 * n))
            # Matriz constitutiva elástica.
            if (dados.tipo_concreto == 0) or (tensoes is None):
                d = dados.concreto.matriz_constitutiva_isotropico()
            else:
                d = dados.concreto.matriz_constitutiva_ortotropica_rotacionada(tensoes[i], deformacoes[i])
            # Iteração sobre os pontos de integração de cada elemento.
            for j in range(len(mb_pesos[i])):
                # Matriz cinemática nodal do elemento no ponto de integração j.
                b_j = mb_pesos[i][j][0]
                # Produto entre o peso da integração, a espessura do elemento e o jacobiano.
                wtj = mb_pesos[i][j][1]
                # Matriz de rigidez do elemento atualizada.
                kel += b_j.T @ d @ b_j * wtj
            kelems.append(kel)
        return kelems

    def matrizes_rigidez_barras(self) -> List[np.ndarray]:
        """Retorna as matrizes de rigidez dos elementos de barra considerando-se unitários a área da seção
        e o módulo de elasticidade."""
        kelems = []
        # Variável auxiliar para a captura dos elementos de barra que são posicionados na lista na sequência
        # dos elementos poligonais
        for i in range(self.dados.num_elementos_poli, self.dados.num_elementos):
            kelems.append(self.elementos[i].matriz_rigidez())
        return kelems

    @staticmethod
    def matrizes_rigidez_barras_tensoes(dados: Dados, tensoes: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """Retorna as matrizes de rigidez das barras calculadas em função das tensões."""
        kelems = []
        if tensoes is None:
            return dados.matrizes_rigidez_barras
        else:
            for i, s in enumerate(tensoes[dados.num_elementos_poli::]):
                if s > 0:
                    kelems.append(dados.aco.et * dados.matrizes_rigidez_barras[i])
                else:
                    kelems.append(dados.aco.ec * dados.matrizes_rigidez_barras[i])
        return kelems

    def criar_vetor_apoios(self) -> np.ndarray:
        """Cria um vetor com a identificação dos graus de liberdade apoiados em função do dicionário de
        apoios nodais."""
        vet_apoios = []
        for no in self.dict_apoios:
            gls = ElementoPoligonal.id_no_para_grau_liberdade(no)
            for i, ap in enumerate(self.dict_apoios[no]):
                if ap == 1:
                    vet_apoios.append(gls[i])
                elif ap != 0:
                    raise ValueError(f'O valor não é aceito como restrição de deslocamento do nó {no}')
        return np.array(vet_apoios, dtype=int)

    def criar_vetor_forcas(self) -> np.ndarray:
        """Cria o vetor de forças da estrutura a partir do dicionário de forças."""
        cargas = np.zeros(self.dados.num_graus_liberdade())
        for no in self.dict_forcas:
            gls = ElementoPoligonal.id_no_para_grau_liberdade(no)
            cargas[list(gls)] = self.dict_forcas[no]
        return cargas

    @staticmethod
    def converter_vetor_forcas_em_dict(dados: Dados) -> dict:
        """Converte um vetor de forças em um dicionário de forças.
        dic = {no: [dado_x, dado_y]}.

        Args:
            dados: Objeto que intermedia o acesso aos dados do problema.
        """
        dic = {}
        forcas = dados.forcas
        for i in range(0, forcas.shape[0], 2):
            v = forcas[[i, i + 1]]
            if any(vtmp for vtmp in v):
                no = int(i / 2)
                dic[no] = v.tolist()
        return dic

    @staticmethod
    def converter_vetor_apoios_em_dict(dados: Dados) -> dict:
        """Converte um vetor de apoios em um dicionário de apoios.
        dic = {no: [desloc_x, desloc_y]}.

        Args:
            dados: Objeto que intermedia o acesso aos dados do problema.
        """
        dic = {}

        vetor_apoios_def = np.zeros(dados.num_graus_liberdade(), dtype=int)
        vetor_apoios_def[dados.apoios] = 1

        for i in range(0, vetor_apoios_def.shape[0], 2):
            v = vetor_apoios_def[[i, i + 1]]
            if any(vtmp for vtmp in v):
                no = int(i / 2)
                dic[no] = v.tolist()

        return dic

    def salvar_deslocamentos_estrutura_original(self) -> np.ndarray:
        """Salva o vetor de deslocamentos nodais da estrutura original (domínio estendido sólido).

        Returns:
            Vetor de deslocamentos nodais da estrutura original.
        """
        # Interface Julia
        julia = Main
        julia.eval('include("julia_core/Deslocamentos.jl")')
        julia.kelems = self.matrizes_rigidez_elementos_poligonais(self.dados) + self.matrizes_rigidez_barras()
        julia.gls_elementos = [i + 1 for i in self.dados.graus_liberdade_elementos]
        julia.gls_estrutura = [i + 1 if i != -1 else i for i in self.dados.graus_liberdade_estrutura]
        julia.apoios = self.dados.apoios + 1
        julia.forcas = self.dados.forcas
        julia.rcm = self.dados.rcm + 1
        julia.eval(f'dados = Dict("kelems" => kelems, "gls_elementos" => gls_elementos, '
                   f'"gls_estrutura" => gls_estrutura, "apoios" => apoios, "forcas" => forcas, '
                   f'"RCM" => rcm)')

        u = julia.eval(f'deslocamentos(kelems, dados)')
        self.dados.salvar_arquivo_numpy(u, 17)
        return u

    @staticmethod
    def deformacoes_elementos(dados: Dados, u: np.ndarray) -> List[Union[np.ndarray, float]]:
        """Retorna um vetor contendo as deformações dos elementos finitos.
        Para os elementos poligonais, as deformações são calculadas em seus centroides."""
        defs = []
        num_els_poli = dados.num_elementos_poli

        for i in range(dados.num_elementos):
            el = dados.elementos[i]
            if len(dados.elementos[i]) > 2:
                # Deformações no sistema global.
                defs.append(dados.matrizes_b_centroide[i] @ u[dados.graus_liberdade_elementos[i]])
            else:
                # Deformação de uma barra.
                gls = dados.graus_liberdade_elementos[i]
                # Nós originais.
                nos = dados.nos[el]
                # Nós deslocados.
                nos_desloc = nos + u[gls].reshape(2, 2)
                # Comprimento original.
                c = dados.comprimentos_barras[i - num_els_poli]
                # Comprimento deformado.
                c_def = np.linalg.norm(nos_desloc[1] - nos_desloc[0])
                defs.append((c_def - c) / c)

        return defs

    @staticmethod
    def tensoes_elementos(dados: Dados, u: np.ndarray, tensoes_ant=None) -> List[Union[np.ndarray, float]]:
        """Retorna um vetor com as tensões no sistema global que atuam no centroide dos elementos
        poligonais.

        Args:
            dados:
            u:
            tensoes_ant: Tensões anteriores. Se as tensões forem None, será utilizado o concreto isotrópico
                com o módulo de elasticidade do concreto à compressão."""
        num_els_poli = dados.num_elementos_poli
        tensoes = []
        # Deformações no sistema global.
        defs = Estrutura.deformacoes_elementos(dados, u)
        # Matriz constitutiva elástica.
        for i in range(dados.num_elementos):
            if i < num_els_poli:
                if (dados.tipo_concreto == 0) or (tensoes_ant is None):
                    tensoes.append(dados.concreto.matriz_constitutiva_isotropico() @ defs[i])
                else:
                    tensoes.append(dados.concreto.matriz_constitutiva_ortotropica_rotacionada(tensoes_ant[i],
                                                                                              defs[i]) @ defs[i])
            else:
                if tensoes_ant is None:
                    tensoes.append(dados.aco.et * defs[i])
                else:
                    if tensoes_ant[i] > 0:
                        tensoes.append(dados.aco.et * defs[i])
                    else:
                        tensoes.append(dados.aco.ec * defs[i])

        return tensoes

    @staticmethod
    def maior_tensao_principal_elementos(dados: Dados, u: np.ndarray, tensoes: Optional[np.ndarray] = None):
        """Retorna um vetor contendo as tensões principais de maiores módulos em cada elementos.
        Retorna uma tensão por elemento."""
        if tensoes is None:
            tensoes = Estrutura.tensoes_elementos(dados, u, tensoes)

        for i in range(len(tensoes)):
            sx, sy, txy = tensoes[i]
            # Tensões principais
            p1 = (sx + sy) / 2
            p2 = np.sqrt(((sx + sy) / 2) ** 2 + txy ** 2)
            s1 = p1 + p2
            s2 = p1 - p2
            tensoes[i] = s1 if abs(s1) >= abs(s2) else s2

        return tensoes
