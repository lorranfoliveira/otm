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
        self._elementos: Optional[np.ndarray] = None
        self._nos: Optional[np.ndarray] = None
        self._poligono_dominio_estendido: Optional[Polygon] = None

        # Dados da análise.
        self._forcas: Optional[np.ndarray] = None
        self._graus_liberdade_elementos: Optional[List[np.ndarray]] = None
        self._apoios: Optional[np.ndarray] = None
        self._k_elems: Optional[List[np.ndarray]] = None
        self._volumes_elementos_solidos: Optional[np.ndarray] = None
        self._graus_liberdade_estrutura: Optional[np.ndarray] = None
        self._rcm: Optional[np.ndarray] = None

        # Dados da otimização.
        self._pesos_esquema_projecao: Optional[List[np.array]] = None

        # Dados dos resultados.
        self._resultados_rho: Optional[np.ndarray] = None
        self._resultados_gerais: Optional[np.ndarray] = None

    @property
    def elementos(self) -> List[np.ndarray]:
        if self._elementos is None:
            self._elementos = self.ler_arquivo_entrada_dados_numpy(0)
        return self._elementos

    @property
    def nos(self) -> np.ndarray:
        if self._nos is None:
            self._nos = self.ler_arquivo_entrada_dados_numpy(1)
        return self._nos

    @property
    def forcas(self) -> np.ndarray:
        if self._forcas is None:
            self._forcas = self.ler_arquivo_entrada_dados_numpy(4)
        return self._forcas

    @property
    def graus_liberdade_elementos(self) -> List[np.ndarray]:
        if self._graus_liberdade_elementos is None:
            self._graus_liberdade_elementos = self.ler_arquivo_entrada_dados_numpy(5)
        return self._graus_liberdade_elementos

    @property
    def apoios(self) -> np.ndarray:
        if self._apoios is None:
            self._apoios = self.ler_arquivo_entrada_dados_numpy(6)
        return self._apoios

    @property
    def k_elems(self) -> List[np.ndarray]:
        if self._k_elems is None:
            self._k_elems = self.ler_arquivo_entrada_dados_numpy(7)
        return self._k_elems

    @property
    def volumes_elementos_solidos(self) -> np.ndarray:
        if self._volumes_elementos_solidos is None:
            self._volumes_elementos_solidos = self.ler_arquivo_entrada_dados_numpy(8)
        return self._volumes_elementos_solidos

    @property
    def graus_liberdade_estrutura(self) -> np.ndarray:
        if self._graus_liberdade_estrutura is None:
            self._graus_liberdade_estrutura = self.ler_arquivo_entrada_dados_numpy(9)
        return self._graus_liberdade_estrutura

    @property
    def poligono_dominio_estendido(self) -> Polygon:
        if self._poligono_dominio_estendido is None:
            self._poligono_dominio_estendido = self.ler_arquivo_wkb_shapely()
        return self._poligono_dominio_estendido

    @property
    def rcm(self) -> np.ndarray:
        if self._rcm is None:
            self._rcm = self.ler_arquivo_entrada_dados_numpy(11)
        return self._rcm

    @property
    def pesos_esquema_projecao(self) -> List[np.ndarray]:
        if self._pesos_esquema_projecao is None:
            self._pesos_esquema_projecao = self.ler_arquivo_entrada_dados_numpy(13)
        return self._pesos_esquema_projecao

    @property
    def resultados_rho(self) -> np.ndarray:
        if self._resultados_rho is None:
            self._resultados_rho = self.ler_arquivo_entrada_dados_numpy(14)
        return self._resultados_rho

    @property
    def resultados_gerais(self) -> np.ndarray:
        if self._resultados_gerais is None:
            self._resultados_gerais = self.ler_arquivo_entrada_dados_numpy(15)
        return self._resultados_gerais

    # region Leitura escrita de arquivos
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
            if self.arquivo.name not in arq_zip.namelist():
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

    def num_elementos(self) -> int:
        """Retorna a quantidade de elementos que compõem a malha."""
        return len(self.elementos)

    # endregion

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
