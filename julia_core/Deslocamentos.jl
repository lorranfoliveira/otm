using SparseArrays
include("LeituraDados.jl")

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
Faz a montagem da matriz de rigidez da estrutura. Se retornar vetores for true, retorna 
os três vetores que compõem a matriz esparsa. Caso contrário, retorna a matriz esparsa.
No caso da saída com vetores, o retorno é feito com uma lista com linhas, colunas e termos, 
respectivamente. 
"""
function matriz_rigidez_estrutura(kelems, dados::Dict, retornar_vetores::Bool=false)
    gls_elems = dados[ctes[2]]
    gls_est = dados[ctes[3]]
    apoios = dados[ctes[4]]

    # Tamanho dos vetores
    num_it = sum([size(ke)[1]^2 for ke in kelems])

    linhas = ones(Int64, num_it)
    colunas = ones(Int64, num_it)
    termos = zeros(num_it)

    # Número de graus de liberdade livres
    num_gls_livres = length(gls_est) - length(apoios)

    t = 1

    for i in range(1, stop=length(gls_elems))
        gle = gls_elems[i]
        ke = kelems[i]

        # Graus de liberdade do elemento com apoios
        gle_com_apoios = gls_est[gle]

        # Posições livres do elemento
        pos_livres = [i for (i, gl) in enumerate(gle_com_apoios) if gl != -1]

        # Graus de liberdade livres na forma B
        gle_livres = gle_com_apoios[pos_livres]

        # Matriz de rigidez com os graus de liberdade impedidos
        ke_livre = ke[pos_livres, pos_livres]

        num_gle_livres = length(gle_livres)

        if length(gle_livres) > 0
            for ii in range(1, stop=num_gle_livres)
                for jj in range(1, stop=num_gle_livres)
                    linhas[t] = gle_livres[ii]
                    colunas[t] = gle_livres[jj]
                    termos[t] = ke_livre[ii, jj]
                    t += 1
                end
            end
        end
    end
    k = sparse(linhas, colunas, termos)
    dropzeros!(k) 
    if retornar_vetores
        return linhas, colunas, termos
    else
        return k
    end
end

"""
Resolve o sistema de equações e retorna os deslocamentos nodais.
"""
function deslocamentos(kelems, dados)
    k = matriz_rigidez_estrutura(kelems, dados)

    forcas = dados["forcas"]
    apoios = dados["apoios"]

    gls_livres = [i for (i, f) in enumerate(forcas) if i ∉ apoios]
    forcas_com_apoios = forcas[gls_livres]

    # Aplicação do RCM
    rcm = dados["RCM"]
    k_rcm = k[rcm, rcm]
    forcas_rcm = forcas_com_apoios[rcm]

    # Solução do sistema
    u0 = k_rcm \ forcas_rcm

    # Reversão do RCM
    rcm_rev = zeros(Int64, length(rcm))

    for (i, v) in enumerate(rcm)
        rcm_rev[v] = i
    end

    u1 = u0[rcm_rev]
    u1 = k \ forcas_com_apoios

    u = zeros(length(forcas))
    u[gls_livres] = u1

    return u
end