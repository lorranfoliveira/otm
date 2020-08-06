include("julia_core\\LeituraDados.jl")
include("julia_core\\Otimizador.jl")

@info "Testando o info no Main"

dados = ler_arquivos_entrada("ANALISE_matrizes_rigidez_elementos.npz",
"ANALISE_graus_liberdade_por_elemento.npz",
"ANALISE_graus_liberdade_estrutura.npy",
"ANALISE_apoios.npy",
"ANALISE_forcas.npy",
"ANALISE_vetor_reverse_cuthill_mckee.npy",
"ANALISE_volumes_elementos.npy",
"MALHA_nos.npy")

oc = OC(dados[ctes[1]],
        dados[ctes[2]],
        dados[ctes[3]],
        dados[ctes[4]],
        dados[ctes[5]],
        dados[ctes[6]],
        dados[ctes[7]],
        dados[ctes[8]],
        5,
        0.4,
        [])

otimizar(oc, 0.5, 20, 75, 1e-3)

print(oc.x)
