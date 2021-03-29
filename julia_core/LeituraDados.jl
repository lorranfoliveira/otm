using PyCall

py"""
import numpy as np

def carregar_npz(arquivo):
    ks = np.load(arquivo)
    return [ks[ks.files[i]] for i in range(len(ks.files))]

def carregar_npy(arquivo):
    return np.load(arquivo)
"""

# using NPZ
#
# function carregar_npz(arquivo)
#     arq = npzread(arquivo)
#     arq_def = [v[2] for v in arq]
# end
#
# function carregar_npy(arquivo)
#     npzread(arquivo)
# end

ctes = Dict(1 => "kelems",
            2 => "gls_elementos",
            3 => "gls_estrutura",
            4 => "apoios",
            5 => "forcas",
            6 => "RCM",
            7 => "volumes_elementos",
            8 => "nos")


"""
Corrige os números dos graus de liberdade da notação do Python para a notação
de Julia. Soma 1 aos graus de liberdade não impedidos.
"""
function corrigir_graus_lib(x)
    if x != -1
        x + 1
    else
        x
    end
end


"""
Retorna um dicionário com as leituras dos arquivos necessários para a resolução do sistema
de equações. Os arquivos já devem ter sido extraídos para que a leitura seja efetiva.
"""
function ler_arquivos_entrada(arq_kelems, arq_gls_elementos, arq_gls_estrutura,
    arq_apoios, arq_forcas, arq_rcm)
    # , arq_volumes_elementos, arq_nos

    dados = Dict(
        ctes[1] => py"carregar_npz"(arq_kelems),
        ctes[2] => py"carregar_npz"(arq_gls_elementos),
        ctes[3] => py"carregar_npy"(arq_gls_estrutura),
        ctes[4] => py"carregar_npy"(arq_apoios),
        ctes[5] => py"carregar_npy"(arq_forcas),
        ctes[6] => py"carregar_npy"(arq_rcm)
    )

    # dados = Dict(
    #     ctes[1] => carregar_npz(arq_kelems),
    #     ctes[2] => carregar_npz(arq_gls_elementos),
    #     ctes[3] => carregar_npy(arq_gls_estrutura),
    #     ctes[4] => carregar_npy(arq_apoios),
    #     ctes[5] => carregar_npy(arq_forcas),
    #     ctes[6] => carregar_npy(arq_rcm),
    #     ctes[7] => carregar_npy(arq_volumes_elementos),
    #     ctes[8] => carregar_npy(arq_nos)
    # )

    # A todos os graus de liberdade deve ser somado 1 para a compatibilização com Julia
    for (i, gl) in enumerate(dados[ctes[2]])
        dados[ctes[2]][i] = map(corrigir_graus_lib, gl)
    end

    # gls_estrutura
    dados[ctes[3]] = map(corrigir_graus_lib, dados[ctes[3]])
    # apoios
    dados[ctes[4]] = map(corrigir_graus_lib, dados[ctes[4]])
    # RCM
    dados[ctes[6]] = map(corrigir_graus_lib, dados[ctes[6]])

    # Return
    dados
end
