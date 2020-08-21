import pathlib
from typing import Optional, Union, List
from shapely.geometry import Polygon
import numpy as np
import zipfile
from shapely.wkb import loads
from otm.constantes import ARQUIVOS_DADOS_ZIP
from loguru import logger
import os

__all__ = ['Dados']


class Dados:
    """Classe dedicada à extração dos dados dos arquivos internos ao `.zip` do problema."""
    IDS_DADOS_MALHA = [0, 1, 10]
    IDS_DADOS_ANALISE = [4, 5, 6, 7, 8, 9, 11]
    IDS_DADOS_OTIMIZACAO = [2]
    IDS_DADOS_RESULTADOS = [14, 15]

    def __init__(self, arquivo: pathlib.Path):
        """Construtor.

        Args:
            arquivo: Diretório do arquivo `.zip` do problema.
        """
        self.arquivo = arquivo

        # Dados da malha.
        self.elementos: Optional[np.ndarray] = None
        self.nos: Optional[np.ndarray] = None
        self.poligono_dominio_estendido: Optional[Polygon] = None

        # Dados da análise.
        self.forcas: Optional[np.ndarray] = None
        self.graus_liberdade_elementos: Optional[List[np.ndarray]] = None
        self.apoios: Optional[np.ndarray] = None
        self.k_elems: Optional[List[np.ndarray]] = None
        self.volumes_elementos: Optional[np.ndarray] = None
        self.graus_liberdade_estrutura: Optional[np.ndarray] = None
        self.vetor_reverse_cuthill_mckee: Optional[np.ndarray] = None

        # Dados da otimização.
        self.pesos_esq_proj: Optional[List[np.array]] = None

        # Dados dos resultados.
        self.resultados_rho: Optional[np.ndarray] = None
        self.resultados_gerais: Optional[np.ndarray] = None

    # region
    def ler_arquivo_entrada_dados_numpy(self, n: int) -> Union[np.ndarray, list]:
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

    def carregar_arquivos(self, *args):
        """Carrega os arquivos especificados em `args`. A codificação deve ser a que consta em `constantes`.
        Arquivos já carregados serão ignorados caso seu carregamento seja novamente solicitado.

        Args:
            args: Iterável que contém o código dos arquivos que serão carregados.
        """
        if ((n := 0) in args) and (self.elementos is None):
            self.elementos = self.ler_arquivo_entrada_dados_numpy(n)

        if ((n := 1) in args) and (self.nos is None):
            self.nos = self.ler_arquivo_entrada_dados_numpy(n)

        if ((n := 4) in args) and (self.forcas is None):
            self.forcas = self.ler_arquivo_entrada_dados_numpy(n)

        if ((n := 5) in args) and (self.graus_liberdade_elementos is None):
            self.graus_liberdade_elementos = self.ler_arquivo_entrada_dados_numpy(n)

        if ((n := 6) in args) and (self.apoios is None):
            self.apoios = self.ler_arquivo_entrada_dados_numpy(n)

        if ((n := 7) in args) and (self.k_elems is None):
            self.k_elems = self.ler_arquivo_entrada_dados_numpy(n)

        if ((n := 8) in args) and (self.volumes_elementos is None):
            self.volumes_elementos = self.ler_arquivo_entrada_dados_numpy(n)

        if ((n := 9) in args) and (self.graus_liberdade_estrutura is None):
            self.graus_liberdade_estrutura = self.ler_arquivo_entrada_dados_numpy(n)

        if (10 in args) and (self.poligono_dominio_estendido is None):
            self.poligono_dominio_estendido = self.ler_arquivo_wkb_shapely()

        if ((n := 11) in args) and (self.vetor_reverse_cuthill_mckee is None):
            self.vetor_reverse_cuthill_mckee = self.ler_arquivo_entrada_dados_numpy(n)

        if ((n := 13) in args) and (self.pesos_esq_proj is None):
            self.pesos_esq_proj = self.ler_arquivo_entrada_dados_numpy(n)

        if ((n := 14) in args) and (self.resultados_rho is None):
            self.resultados_rho = self.ler_arquivo_entrada_dados_numpy(n)

        if ((n := 15) in args) and (self.resultados_gerais is None):
            self.resultados_gerais = self.ler_arquivo_entrada_dados_numpy(n)

    # Resultados
    def rhos_iter_final(self) -> np.ndarray:
        """Retorna um vetor com as densidades relativas dos elementos ao fim da última iteração."""
        return self.resultados_rho[-1]

    def resultados_gerais_iter_final(self)->np.ndarray:
        """Retorna um vetor com os resultado gerais referentes à última iteração da otimização."""
        return self.resultados_gerais[-1]

    def num_iteracoes(self) -> int:
        """Retorna o número de iterações feitas durante a otimização."""
        return self.resultados_gerais.shape[0]

    # enregion
