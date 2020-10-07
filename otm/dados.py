import pathlib
from typing import Optional, Union, List
from shapely.geometry import Polygon
import numpy as np
import zipfile
from shapely.wkb import loads
from otm.constantes import ARQUIVOS_DADOS_ZIP
from loguru import logger
from otm.mef.materiais import Concreto, Aco
import shelve
import os
from otm.mef.elementos import ElementoPoligonal

__all__ = ['Dados']


class Dados:
    """Classe dedicada à extração dos dados dos arquivos internos ao `.zip` do problema.
    As properties são implementadas de forma que os arquivos que elas retornam sejam carregados
    apenas uma vez."""
    IDS_DADOS_MALHA = [0, 1, 10]
    IDS_DADOS_ANALISE = [4, 5, 6, 7, 8, 9, 11]
    IDS_DADOS_OTIMIZACAO = [2]
    IDS_DADOS_RESULTADOS = [14, 15]

    def __init__(self, arquivo: pathlib.Path, concreto: Concreto, aco: Optional[Aco] = None, tipo_concreto=0):
        """Construtor.

        Args:
            arquivo: Diretório do arquivo `.zip` do problema.
            concreto:
            tipo_concreto: Se o tipo do concreto for 0, será utilizado o modelo isotrópico. Se for 1, será
                utilizado o modelo ortotrópico.
        TODO introduzir o aço como material.
        """
        self.arquivo = arquivo
        self.concreto = concreto
        self.aco = aco
        self.tipo_concreto = tipo_concreto

        # Dados da malha.
        self._elementos: Optional[np.ndarray] = None
        self._nos: Optional[np.ndarray] = None
        self._poligono_dominio_estendido: Optional[Polygon] = None
        self._num_elementos: Optional[int] = None
        self._num_elementos_poli: Optional[int] = None
        self._num_elementos_barra: Optional[int] = None
        self._comprimentos_barras: Optional[np.ndarray] = None

        # Dados da análise.
        self._deslocamentos_estrutura_original: Optional[np.ndarray] = None
        self._forcas: Optional[np.ndarray] = None
        self._graus_liberdade_elementos: Optional[List[np.ndarray]] = None
        self._apoios: Optional[np.ndarray] = None
        self._k_elems: Optional[List[np.ndarray]] = None
        self._volumes_elementos_solidos: Optional[np.ndarray] = None
        self._graus_liberdade_estrutura: Optional[np.ndarray] = None
        self._rcm: Optional[np.ndarray] = None
        self._matrizes_b_centroide: Optional[List[np.ndarray]] = None
        self._matrizes_b_pontos_integracao: Optional[List[list]] = None
        self._matrizes_rigidez_barras: Optional[List[np.ndarray]] = None

        # Dados da otimização.
        self._pesos_esquema_projecao: Optional[List[np.array]] = None

        # Dados dos resultados.
        self._resultados_rho: Optional[np.ndarray] = None
        self._resultados_gerais: Optional[np.ndarray] = None

    @property
    def deslocamentos_estrutura_original(self) -> np.ndarray:
        """Deslocamentos da estrutura considerando sólidos todos os elementos do domínio estendido."""
        if self._deslocamentos_estrutura_original is None:
            self._deslocamentos_estrutura_original = self.ler_arquivo_entrada_dados_numpy(17)
        return self._deslocamentos_estrutura_original

    @property
    def elementos(self) -> List[np.ndarray]:
        """Lista contendo os nós que compõem cada elemento finito da discretização."""
        if self._elementos is None:
            self._elementos = self.ler_arquivo_entrada_dados_numpy(0)
        return self._elementos

    @property
    def num_elementos_poli(self):
        if self._num_elementos_poli is None:
            self._num_elementos_poli = len([e for e in self.elementos if len(e) > 2])
        return self._num_elementos_poli

    @property
    def num_elementos_barra(self):
        if self._num_elementos_barra is None:
            self._num_elementos_barra = self.num_elementos - self.num_elementos_poli
        return self._num_elementos_barra

    @property
    def num_elementos(self) -> int:
        """Retorna a quantidade de elementos que compõem a malha."""
        if self._num_elementos is None:
            self._num_elementos = len(self.elementos)
        return self._num_elementos

    @property
    def comprimentos_barras(self):
        if self._comprimentos_barras is None:
            self._comprimentos_barras = self.ler_arquivo_entrada_dados_numpy(21)
        return self._comprimentos_barras

    @property
    def nos(self) -> np.ndarray:
        """Nós que compõem a malha."""
        if self._nos is None:
            self._nos = self.ler_arquivo_entrada_dados_numpy(1)
        return self._nos

    @property
    def forcas(self) -> np.ndarray:
        """Vetor de forças."""
        if self._forcas is None:
            self._forcas = self.ler_arquivo_entrada_dados_numpy(4)
        return self._forcas

    @property
    def graus_liberdade_elementos(self) -> List[np.ndarray]:
        """Lista com os graus de liberdade de cada elemento finito."""
        if self._graus_liberdade_elementos is None:
            self._graus_liberdade_elementos = self.ler_arquivo_entrada_dados_numpy(5)
        return self._graus_liberdade_elementos

    @property
    def apoios(self) -> np.ndarray:
        """Vetor de graus de liberdade com deslocamentos restritos."""
        if self._apoios is None:
            self._apoios = self.ler_arquivo_entrada_dados_numpy(6)
        return self._apoios

    @property
    def kelems(self) -> List[np.ndarray]:
        """Lista com a matriz de rigidez de todos os elementos finitos sólidos."""
        if self._k_elems is None:
            self._k_elems = self.ler_arquivo_entrada_dados_numpy(7)
        return self._k_elems

    @property
    def matrizes_b_centroide(self):
        """Retorna as matrizes cinemáticas nodais calculadas com as coordenadas do centroide dos elementos"""
        if self._matrizes_b_centroide is None:
            self._matrizes_b_centroide = self.ler_arquivo_entrada_dados_numpy(18)
        return self._matrizes_b_centroide

    @property
    def matrizes_b_pontos_integracao(self) -> List[list]:
        """Lista que contém as matrizes B e o produto entre o peso, a espessura e o jacobiano de cada elemento.
        Cada elemento é representado por uma lista contendo as lista que representam os valores para cada ponto
        de integração."""
        # Arquivo a ser lido.
        if self._matrizes_b_pontos_integracao is None:
            arq_1 = f'{ARQUIVOS_DADOS_ZIP[19]}.dat'
            arq_2 = f'{ARQUIVOS_DADOS_ZIP[19]}.dir'
            with zipfile.ZipFile(self.arquivo, 'r') as arq_zip:
                arq_zip.extract(arq_1, str(self.arquivo.parent))
                arq_zip.extract(arq_2, str(self.arquivo.parent))

            with shelve.open(ARQUIVOS_DADOS_ZIP[19]) as arq_dat:
                self._matrizes_b_pontos_integracao = arq_dat['0'].copy()

            try:
                os.remove(str(self.arquivo.parent.joinpath(arq_1)))
                os.remove(str(self.arquivo.parent.joinpath(arq_2)))
            except PermissionError as erro:
                logger.warning(erro)

        return self._matrizes_b_pontos_integracao

    @property
    def matrizes_rigidez_barras(self):
        if self._matrizes_rigidez_barras is None:
            self._matrizes_rigidez_barras = self.ler_arquivo_entrada_dados_numpy(20)
        return self._matrizes_rigidez_barras

    @property
    def volumes_elementos_solidos(self) -> np.ndarray:
        """Vetor contendo os volumes sólidos dos elementos finitos da malha."""
        if self._volumes_elementos_solidos is None:
            self._volumes_elementos_solidos = self.ler_arquivo_entrada_dados_numpy(8)
        return self._volumes_elementos_solidos

    @property
    def graus_liberdade_estrutura(self) -> np.ndarray:
        """Vetor contendo a correspondência entre os graus de liberdade sequencialmente numerados e os
        numerados com a retirada dos graus de liberdade impedidos."""
        if self._graus_liberdade_estrutura is None:
            self._graus_liberdade_estrutura = self.ler_arquivo_entrada_dados_numpy(9)
        return self._graus_liberdade_estrutura

    @property
    def poligono_dominio_estendido(self) -> Polygon:
        """Polígono do Shapely que representa a geometria do domínio estendido."""
        if self._poligono_dominio_estendido is None:
            self._poligono_dominio_estendido = self.ler_arquivo_wkb_shapely()
        return self._poligono_dominio_estendido

    @property
    def rcm(self) -> np.ndarray:
        """Vetor de permutação da matriz de rigidez pelo método de redução de banda Reverse Cuthill Machee."""
        if self._rcm is None:
            self._rcm = self.ler_arquivo_entrada_dados_numpy(11)
        return self._rcm

    @property
    def pesos_esquema_projecao(self) -> List[np.ndarray]:
        """Pesos que representam a influência dos nós sobre os elementos. Em cada índice, a primeira coluna
        da matriz contém a id do nó e a segunda seu peso para o elemento do índice em questão."""
        if self._pesos_esquema_projecao is None:
            self._pesos_esquema_projecao = self.ler_arquivo_entrada_dados_numpy(13)
        return self._pesos_esquema_projecao

    @property
    def resultados_rho(self) -> np.ndarray:
        """Matriz contendo os `rhos` de todos os elementos em cada iteração da otimização."""
        if self._resultados_rho is None:
            self._resultados_rho = self.ler_arquivo_entrada_dados_numpy(14)
        return self._resultados_rho

    @property
    def resultados_gerais(self) -> np.ndarray:
        """Matriz que contém os demais resultados. Cada linha representa uma iteração, assim como
        ocorre com `resultados_rho`. Os índices obededecem a seguinte ordem:
        0 -> Id da iteração para valores constantes de `p` e `beta`, ou `c` (do código abaixo).
        1 -> `p`.
        2 -> `beta`.
        3 -> Valor da função objetivo.
        4 -> Percentual de volume da estrutura após a otimização em relação ao volume inicial.
        5 -> Percentual de densidades intermediárias.
        6 -> Erro relacionado aos deslocamentos.
        7 -> Erro relacionado ao percentual de densidades intermediárias.
        """
        if self._resultados_gerais is None:
            self._resultados_gerais = self.ler_arquivo_entrada_dados_numpy(15)
        return self._resultados_gerais

    # region Leitura escrita de arquivos
    def ler_arquivo_entrada_dados_numpy(self, n: int) -> Union[np.ndarray, List[np.ndarray]]:
        """Lê o arquivo de entrada de dados.

        Args:
            n: Número de referência do arquivo no módulo 'constantes'

        Raises:
            FileNotFoundError:
                Se o arquivo .zip não existir.
            FileNotFoundError:
                Se o arquivo do numpy não existir dentro do arquivo da estrutura.
            ValueError:
                Se o arquivo do numpy não for .npy ou .npz.

        Returns:
            Se o arquivo for `.npy`, retorna `np.ndarray`. Se for `.npz`, retorna `List[np.ndarray]`.
        """
        # Arquivo do numpy a ser lido
        arq = ARQUIVOS_DADOS_ZIP[n]
        arq_str = str(self.arquivo.parent.joinpath(arq))

        logger.info(f'Lendo "{arq}"...')

        # Carregamento do arquivo da estrutura (.zip)
        with zipfile.ZipFile(self.arquivo, 'r') as arq_zip:
            # Extração do arquivo do numpy
            try:
                arq_zip.extract(arq, str(self.arquivo.parent))
            except FileNotFoundError:
                raise FileNotFoundError(f'O arquivo do numpy "{arq}" não existe!')

        # Carregamento do arquivo do numpy
        try:
            mat = np.load(arq_str)

            # Identifica o tipo de dado do arquivo, se npy ou npz
            if arq.endswith('.npy'):
                os.remove(arq_str)
                return mat
            elif arq.endswith('.npz'):
                mat_list = [mat[mat.files[i]] for i in range(len(mat.files))]
                del mat
                os.remove(arq_str)
                return mat_list
            else:
                raise ValueError(f'O arquivo de entrada para o numpy deve possuir extensão .npy ou .npz!')
        except ValueError:
            os.remove(arq_str)
            raise ValueError(f'O arquivo "{arq}" não é um arquivo válido do numpy!')

    def salvar_arquivo_numpy(self, dados: Union[np.ndarray, List[np.ndarray]], n: int):
        """Salva um conjunto de dados em um arquivo do numpy (`.npy` ou `.npz`).
        O tipo de arquivo a ser salvo é definido automaticamente. Se o conjunto de dados for do tipo
        `np.ndarray`, será salvo um arquivo `.npy`. Se o conjunto de dados for do tipo `List[np.ndarray]`,
        será salvo um arquivo com extensão `.npz`.

        Args:
            dados: Conjunto de dados a serem salvos.
            n: Número de referência do arquivo no módulo `constantes`.
        """
        arq_np = self.arquivo.parent.joinpath(ARQUIVOS_DADOS_ZIP[n])
        arq_np_str = str(arq_np)
        with zipfile.ZipFile(self.arquivo, 'a', compression=zipfile.ZIP_DEFLATED) as arq_zip:
            if arq_np.name not in arq_zip.namelist():
                logger.info(f'Salvando "{arq_np.name}" em "{self.arquivo.name}..."')

                if arq_np_str.endswith('.npy'):
                    if isinstance(dados, np.ndarray):
                        np.save(str(arq_np_str), dados)
                    else:
                        raise TypeError(f'Para salvar um arquivo ".npy" é necessário que "dados" seja '
                                        f'do tipo "np.ndarray".')
                elif arq_np_str.endswith('.npz'):
                    if isinstance(dados, list) and all(isinstance(i, np.ndarray) for i in dados):
                        np.savez(arq_np_str, *dados)
                    else:
                        raise TypeError(f'Para salvar um arquivo ".npz" é necessário que "dados" seja '
                                        f'do tipo "List[np.ndarray]".')

                arq_zip.write(arq_np.name)
                os.remove(arq_np_str)
            else:
                logger.warning(f'O arquivo "{arq_np.name}" já existe em "{self.arquivo.name}". Faça sua exclusão '
                               f'manualmente para substituí-lo.')

    def salvar_arquivo_generico_em_zip(self, nome_arquivo: str):
        arquivo_generico = self.arquivo.parent.joinpath(nome_arquivo)
        with zipfile.ZipFile(self.arquivo, 'a', compression=zipfile.ZIP_DEFLATED) as arq_zip:
            if arquivo_generico.name not in arq_zip.namelist():
                logger.info(f'Salvando "{arquivo_generico.name}" em "{self.arquivo.name}..."')
                arq_zip.write(arquivo_generico.name)
                os.remove(str(arquivo_generico))
            else:
                logger.warning(f'O arquivo "{arquivo_generico.name}" já existe em "{self.arquivo.name}" e não '
                               f'pode ser sobrescrito! Ele deve ser removido para ser substituído.')

    def ler_arquivo_wkb_shapely(self):
        """Lê um arquivo de enrada de dados de um arquivo zip

        Raises:
            FileNotFoundError:
                Se o arquivo .zip não existir.
            FileNotFoundError:
                Se o arquivo .wkb não existir dentro do arquivo da estrutura.

        """
        # Identificador do arquivo wkb dentro do `.zip`.
        n = 10
        arq_wkb = ARQUIVOS_DADOS_ZIP[n]
        arq_wkb_str = str(self.arquivo.parent.joinpath(arq_wkb))

        logger.info(f'Lendo "{arq_wkb}"...')

        # Leitura do arquivo da estrutura (.zip)
        with zipfile.ZipFile(self.arquivo, 'r') as arq_zip:
            # Extração do binário wkb
            try:
                arq_zip.extract(arq_wkb, str(self.arquivo.parent))
            except FileNotFoundError:
                raise FileNotFoundError(f'O arquivo wkb "{arq_wkb}" não existe!')

        # Leitura do binário wkb e conversão para um objeto do Shapely
        with open(arq_wkb_str, 'rb') as arq_b:
            sh_geo = loads(arq_b.read())

        # Exclusão do arquivo extraído
        os.remove(arq_wkb_str)
        return sh_geo

    # endregion

    # region Dados malha
    def num_nos(self) -> int:
        """Retorna a quantidade de nós que compõem a malha."""
        return self.nos.shape[0]

    # endregion

    # region Estrutura deformada
    def deslocamentos_por_no(self) -> np.ndarray:
        """Retorna uma matriz que contém os deslocamentos nodais em x e em y de cada nó da estrutura original."""
        num_nos = self.num_nos()
        u = self.deslocamentos_estrutura_original
        u_nos = np.zeros((num_nos, 2))

        for n in range(num_nos):
            u_nos[n] = u[ElementoPoligonal.id_no_para_grau_liberdade(n)]

        return u_nos

    # endregion

    # region Dados análise
    def num_graus_liberdade(self) -> int:
        """Retorna o número total de graus de liberdade livres e impedidos que a estrutura possui.
        São considerados 2 graus de liberdade por nó."""
        return 2 * self.num_nos()

    # endregin

    # Resultados
    def rhos_iteracao_final(self) -> np.ndarray:
        """Retorna um vetor com as densidades relativas dos elementos ao fim da última iteração."""
        return self.resultados_rho[-1]

    def resultados_gerais_iteracao_final(self) -> np.ndarray:
        """Retorna um vetor com os resultado gerais referentes à última iteração da otimização."""
        return self.resultados_gerais[-1]

    def num_iteracoes(self) -> int:
        """Retorna o número de iterações feitas durante a otimização."""
        return self.resultados_gerais.shape[0]

    # enregion
