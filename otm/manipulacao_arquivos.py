import numpy as np
import zipfile
from otm.constantes import ARQUIVOS_DADOS_ZIP
from typing import Union, List
from loguru import logger
import os
from shapely.wkb import loads
import pathlib

__all__ = ['ler_arquivo_entrada_dados_numpy', 'salvar_arquivo_numpy', 'ler_arquivo_wkb_shapely']


def ler_arquivo_entrada_dados_numpy(arquivo: Union[str, pathlib.Path], n: int) -> Union[np.ndarray, list]:
    """Lê o arquivo de entrada de dados.

    Args:
        arquivo: Arquivo de entrada de dados em formato zip
        n: Número de referência do arquivo no módulo 'constantes'

    Raises:
        FileNotFoundError:
            Se o arquivo .zip não existir.
        FileNotFoundError:
            Se o arquivo do numpy não existir dentro do arquivo da estrutura.
        ValueError:
            Se o arquivo do numpy não for .npy ou .npz.
    """
    # Arquivo do numpy a ser lido
    arq = ARQUIVOS_DADOS_ZIP[n]

    logger.info(f'Lendo "{arq}"...')

    # Carregamento do arquivo da estrutura (.zip)
    try:
        with zipfile.ZipFile(arquivo, 'r') as arq_zip:
            # Extração do arquivo do numpy
            try:
                arq_zip.extract(arq, arquivo.parent)
            except FileNotFoundError:
                raise FileNotFoundError(f'O arquivo do numpy "{arq}" não existe!')
    except FileNotFoundError:
        raise FileNotFoundError(f'O arquivo "{arquivo}" com os dados da estrutura não existe!')

    # Carregamento do arquivo do numpy
    try:
        mat = np.load(arquivo.parent.joinpath(arq))

        # Identifica o tipo de dado do arquivo, se npy ou npz
        if 'npy' in arq:
            os.remove(arquivo.parent.joinpath(arq))
            return mat
        elif 'npz' in arq:
            mat_list = [mat[mat.files[i]] for i in range(len(mat.files))]
            del mat
            os.remove(arquivo.parent.joinpath(arq))
            return mat_list
        else:
            raise ValueError(f'O arquivo de entrada para o numpy deve possuir extensão .npy ou .npz!')
    except ValueError:
        os.remove(arquivo.parent.joinpath(arq))
        raise ValueError(f'O arquivo "{arq}" não é um arquivo válido do numpy!')


def salvar_arquivo_numpy(arquivo: pathlib.Path, dados: Union[np.ndarray, List[np.ndarray]], n: int):
    """Salva um conjunto de dados em um arquivo do numpy (`.npy` ou `.npz`).
    O tipo de arquivo a ser salvo é definido automaticamente. Se o conjunto de dados for do tipo
    `np.ndarray`, será salvo um arquivo `.npy`. Se o conjunto de dados for do tipo `List[np.ndarray]`,
    será salvo um arquivo com extensão `.npz`.

    Args:
        arquivo: Nome do arquivo `.zip`.
        dados: Conjunto de dados a serem salvos.
        n: Número de referência do arquivo no módulo `constantes`.
    """
    arq_np = arquivo.parent.joinpath(ARQUIVOS_DADOS_ZIP[n])
    arq_np_str = str(arq_np)
    try:
        with zipfile.ZipFile(arquivo, 'a', compression=zipfile.ZIP_DEFLATED) as arq_zip:
            if arquivo.name not in arq_zip.namelist():
                logger.info(f'Salvando "{arq_np.name}" em "{arquivo.name}..."')

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
    except FileNotFoundError:
        raise FileNotFoundError(f'O arquivo "{arquivo}" com os dados da estrutura não existe!')


def ler_arquivo_wkb_shapely(arquivo: Union[str, pathlib.Path], n: int = 10):
    """Lê um arquivo de enrada de dados de um arquivo zip

    Args:
        arquivo: Arquivo de entrada de dados em formato zip
        n: Número de referência do arquivo no módulo 'constantes'

    Raises:
        FileNotFoundError:
            Se o arquivo .zip não existir.
        FileNotFoundError:
            Se o arquivo .wkb não existir dentro do arquivo da estrutura.

    """
    arq_wkb = ARQUIVOS_DADOS_ZIP[n]
    # Leitura do arquivo da estrutura (.zip)
    try:
        with zipfile.ZipFile(arquivo, 'r') as arq_zip:
            # Extração do binário wkb
            try:
                arq_zip.extract(arq_wkb, arquivo.parent)
            except FileNotFoundError:
                raise FileNotFoundError(f'O arquivo wkb "{arq_wkb}" não existe!')
    except FileNotFoundError:
        raise FileNotFoundError(f'O arquivo "{arquivo}" com os dados da estrutura não existe!')

    # Leitura do binário wkb e conversão para um objeto do Shapely
    with open(arquivo.parent.joinpath(arq_wkb), 'rb') as arq_b:
        sh_geo = loads(arq_b.read())

    # Exclusão do arquivo extraído
    os.remove(arquivo.parent.joinpath(arq_wkb))
    return sh_geo
