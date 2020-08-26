from otm.mef.elementos_finitos.elemento_poligonal import ElementoPoligonal
from loguru import logger
from typing import List
from scipy.sparse import csr_matrix
import numpy as np
from otm.constantes import ARQUIVOS_DADOS_ZIP
from scipy.sparse.csgraph import reverse_cuthill_mckee
from otm.dados import Dados
from julia import Main
from otm.mef.materiais import MaterialIsotropico

__all__ = ['Estrutura']


class Estrutura:
    """Implementa as propriedades de uma estrutura."""

    def __init__(self, dados: Dados, concreto: MaterialIsotropico, dict_cargas: dict, dict_apoios: dict,
                 espessura: float = 1):
        """Construtor.

        Args:
            dados: Objeto que intermedia o acesso e a gravação dos dados do arquivo `.zip`.
            concreto: Material utilizado na malha bidimensional.
            dict_cargas: Dicionário contendo a numeração do nó na chave e uma tupla com as cargas em x e y
                no nó. {num_no: (carga_x, carga_y)}.
            dict_apoios: Dicionário contendo a numeração do nó na chave e uma tupla contendo o tipo de
                restrição do nó. Se a restrição for 1, o grau de liberdade estará impedido. Se for 0, o grau
                de liberdade estará livre. {num_no: (deslocabilidade_x, deslocabilidade_y)}.
            espessura: Espessura dos elementos da malha bidimensional.
        """
        self.dados = dados
        self.concreto = concreto
        self.espessura = espessura
        self.dict_forcas = dict_cargas
        self.dict_apoios = dict_apoios

        self.elementos_poligonais: List[ElementoPoligonal] = []

        # Interface Julia
        julia = Main
        julia.eval('include("julia_core/Deslocamentos.jl")')

    def salvar_dados_estrutura(self):
        """Salva os dados da estrutura necessários para a análise estrutural."""
        # Não alterar a ordem das funções abaixo
        self.criar_elementos_poligonais()
        # Vetor de forças.
        self.dados.salvar_arquivo_numpy(self.criar_vetor_forcas(), 4)
        # Vetor de apoios.
        self.dados.salvar_arquivo_numpy(self.criar_vetor_apoios(), 6)
        # Graus de liberdade por elemento.
        self.dados.salvar_arquivo_numpy(self.graus_liberdade_elementos(), 5)
        # Graus de liberdade da estrutura.
        self.dados.salvar_arquivo_numpy(self.graus_liberdade_estrutura(), 9)
        # Matrizes de rigidez dos elementos.
        self.dados.salvar_arquivo_numpy(self.matrizes_rigidez_elementos(), 7)
        # Volumes dos elementos sólidos.
        self.dados.salvar_arquivo_numpy(self.volumes_elementos(), 8)
        # Permutação RCM.
        self.salvar_permutacao_rcm()
        self.salvar_dados_entrada_txt()
        # Salvar deslocamentos nodais da estrutura original sólida.
        self.salvar_deslocamentos_estrutura_original()

        logger.debug(f'Elementos: {len(self.dados.elementos)}, Nós: {len(self.dados.nos)}, '
                     f'GL: {2 * len(self.dados.nos)}')

    def salvar_dados_entrada_txt(self):
        n = 12
        arq = self.dados.arquivo.parent.joinpath(ARQUIVOS_DADOS_ZIP[n])
        with open(str(arq), 'w') as arq_txt:
            arq_txt.write(f'forcas = {self.dict_forcas}\n')
            arq_txt.write(f'apoios = {self.dict_apoios}\n')
            arq_txt.write(f'E = {self.concreto.ec}\n')
            arq_txt.write(f'poisson = {self.concreto.nu}\n')
            arq_txt.write(f'espessura = {self.espessura}\n')

        self.dados.salvar_arquivo_generico_em_zip(ARQUIVOS_DADOS_ZIP[n])

    def criar_elementos_poligonais(self):
        """Cria os objetos que representam os elementos finitos poligonais."""
        logger.debug('Criando os elementos finitos poligonais...')

        self.elementos_poligonais = []
        nos = self.dados.nos
        for i, e in enumerate(self.dados.elementos):
            # Verificar se o elemento é poligonal ou de barra.
            if len(e) > 2:
                el = ElementoPoligonal(nos[e], self.concreto, self.espessura, e)
                self.elementos_poligonais.append(el)

    def graus_liberdade_estrutura(self) -> np.array:
        """Vetor que contém os parâmetros de conversão dos graus de liberdade da forma B para a forma A.
        Forma A: Numeração sequencial que CONSIDERA os graus de liberdade impedidos.
        Forma B: Numeração sequencial que NÃO CONSIDERA os graus de liberdade impedidos. Neste caso, os
            graus de liberdade impedidos possuem valor -1.
        """
        # Todos os graus de liberdade impedidos deverão assumir valor -1
        gls = np.full(self.dados.num_nos() * 2, -1)
        apoios = self.dados.apoios

        for e in self.elementos_poligonais:
            e_glb = e.graus_liberdade()
            e_gla = e_glb.copy()
            for i in range(e_glb.size):
                if (b := e_glb[i]) not in apoios:
                    e_gla[i] -= len(list(filter(lambda x: x < b, apoios)))
                else:
                    e_gla[i] = -1
            gls[e_glb] = e_gla.copy()

        return gls

    def salvar_permutacao_rcm(self):
        """Salva a permutação feita pelo algoritmo de redução da banda da matriz de rigidez Reverse Cuthill Mckee."""
        # Interface Julia
        julia = Main
        julia.eval('include("julia_core/Deslocamentos.jl")')

        julia.kelems = self.dados.k_elems
        julia.gls_elementos = [i + 1 for i in self.dados.graus_liberdade_elementos]
        julia.gls_estrutura = [i + 1 if i != -1 else i for i in self.dados.graus_liberdade_estrutura]
        julia.apoios = self.dados.apoios + 1

        julia.eval('dados = Dict("kelems" => kelems, "gls_elementos" => gls_elementos, '
                   '"gls_estrutura" => gls_estrutura, "apoios" => apoios)')

        linhas, colunas, termos = julia.eval('matriz_rigidez_estrutura(dados["kelems"], dados, true)')

        # Número de graus de liberdade livres.
        ngl = self.dados.num_graus_liberdade() - len(self.dados.apoios)

        k = csr_matrix((termos, (linhas - 1, colunas - 1)), shape=(ngl, ngl))
        rcm = reverse_cuthill_mckee(k)

        # Salvar o vetor rcm.
        self.dados.salvar_arquivo_numpy(rcm, 11)

    def graus_liberdade_elementos(self) -> List[np.ndarray]:
        """Retorna uma lista contendo vetores com os graus de liberdade de cada elemento da malha."""
        return [e.graus_liberdade() for e in self.elementos_poligonais]

    def volumes_elementos(self) -> np.ndarray:
        """Retorna um vetor com os volumes dos elementos finitos."""
        return np.array([self.espessura * el.poligono().area for el in self.elementos_poligonais])

    def matrizes_rigidez_elementos(self) -> List[np.ndarray]:
        """Retorna as matrizes de rigidez dos elementos."""
        logger.debug('Calculando as matrizes de rigidez dos elementos...')

        nel = len(self.elementos_poligonais)
        kels = []
        c = 5
        for i, e in enumerate(self.elementos_poligonais):
            if c <= (perc := (int(100 * (i + 1) / nel))):
                c += 5
                logger.debug(f'{perc}%')
            kels.append(e.matriz_rigidez())
        return kels

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
        julia.kelems = self.dados.k_elems
        julia.gls_elementos = [i + 1 for i in self.dados.graus_liberdade_elementos]
        julia.gls_estrutura = [i + 1 if i != -1 else i for i in self.dados.graus_liberdade_estrutura]
        julia.apoios = self.dados.apoios + 1
        julia.forcas = self.dados.forcas
        julia.rcm = self.dados.rcm + 1
        julia.eval(f'dados = Dict("kelems" => kelems, "gls_elementos" => gls_elementos, '
                   f'"gls_estrutura" => gls_estrutura, "apoios" => apoios, "forcas" => forcas, '
                   f'"RCM" => rcm)')

        u = julia.eval(f'deslocamentos(dados["kelems"], dados)')
        self.dados.salvar_arquivo_numpy(u, 17)
        return u
