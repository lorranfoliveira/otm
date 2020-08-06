using SparseArrays
using LinearAlgebra

mutable struct OC
    matrizes_rigidez_elementos
    graus_liberdade_elementos
    graus_liberdade_estrutura
    vetor_apoios
    vetor_forcas
    rcm
    volumes_elementos_solidos
    nos
    p
    x_inicial
    x
end

function aplicar_simp(oc::OC)
    [oc.x[i]^oc.p * oc.matrizes_rigidez_elementos[i] for i in range(1, stop=length(oc.x))]
end

"""
Faz a montagem da matriz de rigidez da estrutura.
"""
function matriz_rigidez_estrutura(oc::OC, kelems::Array)
    gls_elems = oc.graus_liberdade_elementos
    gls_est = oc.graus_liberdade_estrutura
    apoios = oc.vetor_apoios

    # Tamanho dos vetores
    num_it = sum([size(ke)[1]^2 for ke in kelems])

    linhas = ones(Int64, num_it)
    colunas = ones(Int64, num_it)
    dados = zeros(num_it)

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
                    dados[t] = ke_livre[ii, jj]
                    t += 1
                end
            end
        end
    end
    k = sparse(linhas, colunas, dados)
    dropzeros(k)
end

"""
Resolve o sistema de equações e retorna os deslocamentos nodais.
"""
function deslocamentos(oc::OC, kelems)
    k = matriz_rigidez_estrutura(oc, kelems)

    forcas = oc.vetor_forcas
    apoios = oc.vetor_apoios

    gls_livres = [i for (i, f) in enumerate(forcas) if i ∉ apoios]
    forcas_com_apoios = forcas[gls_livres]

    # Aplicação do RCM
    rcm = oc.rcm
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

    u = zeros(length(forcas))
    u[gls_livres] = u1

    return u
end

"""Calcula os deslocamentos nodais da estrutura considerando o modelo SIMP"""
function deslocamentos_simp(oc::OC)
    kelems_ef = aplicar_simp(oc)
    return deslocamentos(oc, kelems_ef)
end


function funcao_objetivo_flexibilidade(oc::OC, u::Array)
    transpose(oc.vetor_forcas) * u
end

function sensibilidades_funcao_objetivo(oc::OC, u::Array)
    kelems = oc.matrizes_rigidez_elementos

    sensibilidade = zeros(size(kelems))
    for (i, gl) in enumerate(oc.graus_liberdade_elementos)
        sensibilidade[i] = (-oc.p * oc.x[i] ^ (oc.p - 1)) * (transpose(u[gl]) * kelems[i] * u[gl])
        if 0 < sensibilidade[i]
            sensibilidade[i] = 0
        end
    end
    return sensibilidade
end

function atualizar(oc::OC, u::Array)
    vol = sum(oc.x .* oc.volumes_elementos_solidos)
    sens_fo = sensibilidades_funcao_objetivo(oc, u)

    # A sensibilidade de volume é o volume dos elementos em função das densidades relativas
    sens_vol = oc.x .* oc.volumes_elementos_solidos
    nel = length(oc.matrizes_rigidez_elementos)

    l1 = 0
    l2 = 1e6
    move = 0.2
    eta = 0.5
    xnovo = zeros(nel)
    xmin = 1e-3

    while (l2 - l1) > 1e-4
        # Multiplicado de Lagrange médio
        lmid = (l2 + l1) / 2
        be = -sens_fo ./ (lmid * sens_vol)
        t1 = oc.x .* be .^ eta
        t2 = max.(xmin, oc.x .- move)
        t3 = min.(1, oc.x .+ move)

        for i in range(1, stop=nel)
            if t1[i] <= t2[i]
                xnovo[i] = t2[i]
            elseif t2[i] < t1[i] < t3[i]
                xnovo[i] = t2[i]
            else
                xnovo[i] = t3[i]
            end
        end

        # 0.001 é o mínimo valor que qualquer x pode assumir. Eta é 1/2, o que resulta nas raízes quadradas
        # xnovo = np.maximum(self.xmin, np.maximum(x - move, np.minimum(1, np.minimum(x + move, x * np.sqrt(
        #     -sens_fo / (lmid * sens_vol))))))

        if (sum(xnovo .* oc.volumes_elementos_solidos) .- vol) > 0
            l1 = lmid
        else
            l2 = lmid
        end
    end
    oc.x = copy(xnovo)
end

function otimizar(oc::OC, passo=0.5, min_iter=20, max_iter=75, erro_max=1e-3)
    @info "Iniciando a otimização da estrutura"
    oc.x = fill(oc.x_inicial, length(oc.matrizes_rigidez_elementos))

    p_inicial = oc.p

    if passo != -1
        ps = range(1, stop=p_inicial, step=passo)
    else
        ps = [oc.p, oc.p]
    end

    u_antigo = ones(length(oc.vetor_forcas))

    for p in ps
        @info "COEFICIENTE DE PENALIZAÇÃO: p=$p\n"

        oc.p = p
        erro = 10
        c = 1

        while (erro > erro_max) && (c <= max_iter) || (c <= min_iter)
            u = deslocamentos_simp(oc)

            atualizar(oc, u)
            fo = funcao_objetivo_flexibilidade(oc, u)

            if c == 1 == p
                erro = 1
            else
                norm_u = norm(u)
                norm_uant = norm(u_antigo)
                erro = abs((norm_uant - norm_u) / norm_uant)
            end

            u_antigo = copy(u)

            @info "i: $c\t p: $p\t fo: $fo\t vol: $(sum(oc.x .* oc.volumes_elementos_solidos))\t erro: $(100 * erro)%"

            c += 1
        end

        if passo == -1
            break
        end
    end
    oc.p = p_inicial
end
