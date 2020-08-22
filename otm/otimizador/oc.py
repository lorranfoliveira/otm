import numpy as np
from loguru import logger
from julia import Main
from scipy.spatial import KDTree
from typing import Tuple
from otm.dados import Dados


class OC:
    """Classe que implementa as características do problema de otimização pelo Optimality Criteria."""

    # Método utilizado para a representação da função Heaviside regularizada.
    # 0 -> Guest (2004)
    # 1 -> Sigmund (2007)
    # 2 -> Xu et al. (2010).
    METODO_HEAVISIDE = 0
    # Máximo valor que pode ser assumido por beta.
    BETA_MAX = 150
    # Quantidade mínima de iterações.
    NUM_MIN_ITERS = 5
    # Quantidade máxima de iterações.
    NUM_MAX_ITERS = 50
    # Técnicas de otimização.
    # 0 -> Sem filtro.
    # 1 -> Com esquema de projeção linear direto.
    # 2 -> Com esquema de projeção Heaviside direto.
    # 3 -> Com esquema de projeção linear inverso.
    # 4 -> Com esquema de projeção Heaviside inverso.
    TECNICA_OTM_SEM_FILTRO = [0]
    TECNICA_OTM_EP_DIRETO = [1, 2]
    TECNICA_OTM_EP_INVERSO = [3, 4]
    TECNICA_OTM_EP_LINEAR = [1, 3]
    TECNICA_OTM_EP_HEAVISIDE = [2, 4]

    def __init__(self, dados: Dados, x_inicial: float = 0.5, p: float = 3, rho_min: float = 1e-3,
                 rmin: float = 0, tecnica_otimizacao: int = 0):
        """Se rmin == 0, a otimização será feita sem a aplicação do esquema de projeção

        Args:
            tecnica_otimizacao:
                0 -> Sem filtro.
                1 -> Com esquema de projeção linear direto.
                2 -> Com esquema de projeção Heaviside direto.
                3 -> Com esquema de projeção linear inverso.
                4 -> Com esquema de projeção Heaviside inverso.

        """
        self.x_inicial: float = x_inicial
        self.dados = dados
        self.p = p
        self.rho_min = rho_min
        self.rmin = rmin
        self.tecnica_otimizacao = tecnica_otimizacao

        # Interface Julia
        self.julia = Main
        self.julia.eval('include("julia_core/Deslocamentos.jl")')

        self.carregar_interface_julia()

        # Densidades dos nós
        self.x: np.ndarray = np.full(self.dados.num_nos(), self.x_inicial)
        # Densidades dos elementos
        self.rho: np.ndarray = np.full(self.dados.num_elementos(), self.x_inicial)

    def volume_estrutura_inicial(self) -> float:
        """Retorna o volume da estrutura."""
        return np.sum(self.x_inicial * self.dados.volumes_elementos_solidos)

    def deslocamentos_nodais(self) -> np.ndarray:
        """Retorna os deslocamentos nodais em função das variáveis de projeto"""
        # Inserção das densidades relativas no problema
        return self.julia.eval(f'deslocamentos_simp(rho, p, dados)')

    def carregar_interface_julia(self):
        """Faz a leitura do arquivo de entrada de dados"""
        # Interface Julia
        self.julia.kelems = self.dados.k_elems
        self.julia.gls_elementos = [i + 1 for i in self.dados.graus_liberdade_elementos]
        self.julia.gls_estrutura = [i + 1 if i != -1 else i for i in self.dados.graus_liberdade_estrutura]
        self.julia.apoios = self.dados.apoios + 1
        self.julia.forcas = self.dados.forcas
        self.julia.rcm = self.dados.rcm + 1

        self.julia.eval(f'dados = Dict("kelems" => kelems, "gls_elementos" => gls_elementos, '
                        f'"gls_estrutura" => gls_estrutura, "apoios" => apoios, "forcas" => forcas, '
                        f'"RCM" => rcm)')

        # Cálculo dos nós de influência nos elementos pelo esquema de projeção. Esse cálculo só
        # é executado quando o esquema de projeção é utilizado.
        if self.tecnica_otimizacao != 0:
            self.dados.salvar_arquivo_numpy(self.calcular_funcoes_peso(), 13)

    def calcular_funcoes_peso(self):
        """Encontra os nós que se encontram dentro do raio de influência de cada elemento.
        Retorna um vetor onde cada índice corresponde a um elemento e é composto por uma matriz n x 2, onde n
        é o número de nós dentro da área de influência do elemento. A primeiro coluna do índice corresponde ao
        número do nó e a segunda à distância entre o centroide do elemento e o nó em questão.

        Se o número de nós capturados for menor que o número de nós que compõem o elemento, utiliza-se apenas
        os nós do elemento.
        """
        logger.debug(f'Calculando a influência dos nós sobre os elementos...')

        def w(r, rmin) -> float:
            if self.tecnica_otimizacao in OC.TECNICA_OTM_EP_DIRETO:
                # Função de projeção direta
                return (rmin - r) / rmin
            elif self.tecnica_otimizacao in OC.TECNICA_OTM_EP_INVERSO:
                # Função de projeção inversa
                return r / rmin

        num_elems = self.dados.num_elementos()
        # Vetorização da função
        vet_w = np.vectorize(w)
        nos = self.dados.nos
        kd_nos = KDTree(nos)
        conjunto_pesos = []

        c = 5
        for i, e in enumerate(self.dados.elementos):
            if c <= (perc := (int(100 * (i + 1) / num_elems))):
                c += 5
                logger.debug(f'{perc}%')
            # Nós do elemento
            nos_elem = nos[e]
            # Cálculo do centroide do elemento
            centroide = np.mean(nos_elem, axis=0)
            # Pontos que recebem a influência do elemento
            nos_influenciados = kd_nos.query_ball_point(centroide, self.rmin)
            # Um elemento receberá no mínimo a influência dos nós que o formam
            rmin_tmp = self.rmin
            if len(nos_influenciados) < e.shape[0]:
                rmin_tmp = np.max(np.linalg.norm(nos_elem - centroide, axis=1))
                nos_influenciados = kd_nos.query(nos_elem)[1]

            # Matriz contendo os números dos nós que influenciam as densidades de cada elemento na primeira
            # coluna e os pesos na segunda coluna. Os valores da primeira coluna DEVEM ser convertidos para
            # inteiros antes de serem utilizados.

            # Distância entre os nós influenciados e o centroide do elemento de referência
            dists = np.linalg.norm(nos[nos_influenciados] - centroide, axis=1)
            # Aplicação da função de projeção para o cálculo dos pesos
            pesos = vet_w(dists, rmin_tmp)

            conjunto_pesos.append(np.array([nos_influenciados, pesos]).T)

        return conjunto_pesos

    def calcular_densidades_elementos(self, beta) -> np.ndarray:
        """Retorna a densidade relativa do elemento a partir de sua identificação"""
        num_elems = self.dados.num_elementos()
        rho = np.zeros(num_elems)
        pesos_elems = self.dados.pesos_esquema_projecao

        for i in range(num_elems):
            ids_nos = pesos_elems[i][:, 0].astype(int)
            pesos_i = pesos_elems[i][:, 1]
            mi = (self.x[ids_nos] @ pesos_i) / np.sum(pesos_i)
            # Adição da não linearidade
            rho[i] = self.heaviside(mi, beta, OC.METODO_HEAVISIDE)

        return rho

    def flexibilidade_media(self, u: np.ndarray) -> float:
        """Retorna a flexibilidade média da estrutura em função das variáveis de projeto"""
        return self.dados.forcas @ u

    def sensibilidades_sem_filtro(self, u: np.ndarray) -> np.ndarray:
        """Calcula as sensibilidades dos elementos"""
        kelems = self.dados.k_elems
        gl_elems = self.dados.graus_liberdade_elementos
        sens = np.zeros(self.dados.num_elementos())

        for i, gl in enumerate(gl_elems):
            sens[i] = (-self.p * self.rho[i] ** (self.p - 1)) * (u[gl] @ kelems[i] @ u[gl])

        return sens

    @staticmethod
    def heaviside(rho, beta, metodo=0, derivada=False) -> float:
        """Retorna o valor da função de projeção Heaviside suavizada.

        Args:
            rho: Densidade do elemento.
            beta: Parâmetro que define o nível de suavidade da função Heaviside.
            metodo: Método utilizado para a representação da função Heaviside regularizada.
                0=Guest (2004), 1=Sigmund (2007) e 2=Xu et al. (2010).
            derivada: True para o cálculo da derivada da função e False para o cálculo apenas da função.

        Raises:
            ValueError:
                Se o valor passado para `método` não estiver em [0, 1, 2].
        """
        # Método de Guest (2004)
        if metodo == 0:
            if derivada:
                return beta * np.exp(-beta * rho) + np.exp(-beta)
            else:
                return 1 - np.exp(-beta * rho) + rho * np.exp(-beta)
        # Método de Sigmund (2007)
        elif metodo == 1:
            if derivada:
                return beta * np.exp(beta * (rho - 1)) + np.exp(-beta)
            else:
                return np.exp(-beta * (1 - rho)) - (1 - rho) * np.exp(-beta)
        # Método de Xu et al. (2010)
        elif metodo == 2:
            if derivada:
                if rho <= 0.5:
                    return (beta * np.exp(2 * beta * rho) + 1) * np.exp(-beta)
                else:
                    return (beta * np.exp(2 * beta) + np.exp(2 * beta * rho)) * np.exp(-2 * beta * rho - beta)
            else:
                if rho <= 0.5:
                    return 1 / 2 * (2 * rho + np.exp(2 * beta * rho) - 1) * np.exp(-beta)
                else:
                    return 1 / 2 * (
                        (2 * rho + 2 * np.exp(beta) - 1) * np.exp(2 * beta * rho) - np.exp(2 * beta)) * np.exp(
                        -2 * beta * rho - beta)
        else:
            raise ValueError(f'O tipo de função Heaviside "{metodo}" não é válido!')

    def sensibilidades_esquema_projecao(self, u: np.ndarray, beta) -> tuple:
        """Calcula as sensibilidades da função objetivo e da restrição de volume.

        Returns:
            sens_fo: Vetor contendo as sensibilidades da função objetivo.
            sens_vol: Vetor contendo as sensibilidades da restrição de volume.
        """
        kelems = self.dados.k_elems
        # Sensibilidade da função objetivo
        sens_fo = np.zeros(self.dados.num_nos())
        # Sensibilidade da restrição de volume
        sens_vol = np.zeros(self.dados.num_nos())
        vols = self.dados.volumes_elementos_solidos
        gl_elems = self.dados.graus_liberdade_elementos
        pesos_elems = self.dados.pesos_esquema_projecao

        for i, gl in enumerate(gl_elems):
            # Parcela da sensibilidade devida à densidade do elemento
            sens_el = (-self.p * self.rho[i] ** (self.p - 1)) * (u[gl] @ kelems[i] @ u[gl])
            # Desmembramento da matriz de pesos
            ids_nos = pesos_elems[i][:, 0].astype(int)
            pesos_i = pesos_elems[i][:, 1]
            soma_pesos = np.sum(pesos_i)
            # Não linearidade
            mi = (self.x[ids_nos] @ pesos_i) / soma_pesos
            diff_rhoe_rhon = self.heaviside(mi, beta, OC.METODO_HEAVISIDE, True) * pesos_i / soma_pesos
            sens_fo[ids_nos] += sens_el * diff_rhoe_rhon
            sens_vol[ids_nos] += vols[i] * diff_rhoe_rhon

        return sens_fo, sens_vol

    def percentual_densidades_intermediarias(self) -> float:
        """Retorna o percentual de densidades intermediárias da topologia"""
        return 100 * sum(4 * rho * (1 - rho) for rho in self.rho) / len(self.rho)

    def atualizar_x(self, u: np.ndarray, beta=0) -> Tuple[np.ndarray, np.ndarray]:
        """Atualiza as variáveis de projeto (densidades nodais)"""
        vol_inicial = self.volume_estrutura_inicial()
        # Volume da estrutura em função das densidades correntes para os elementos
        vols_elems_solidos = self.dados.volumes_elementos_solidos
        vol_atual = vols_elems_solidos @ self.rho
        # Sensibilidades da função objetivo e da restrição de volume
        if self.tecnica_otimizacao != 0:
            sens_fo, sens_vol = self.sensibilidades_esquema_projecao(u, beta)
            x = self.x
        else:
            sens_fo = self.sensibilidades_sem_filtro(u)
            sens_vol = vols_elems_solidos * self.x_inicial
            x = self.rho

        eta = 0.5
        bm = -sens_fo / sens_vol

        l1 = 0
        l2 = 1e6
        move = 0.2
        n = len(x)
        x_novo = np.zeros(n)

        if (self.tecnica_otimizacao != 0) and (beta != 0):
            x_min = -1 / beta * np.log(1 - self.rho_min)
        else:
            x_min = self.rho_min
        x_max = 1

        while (l2 - l1) > 1e-4:
            lmid = (l2 + l1) / 2
            be = bm / lmid
            t1 = x_min + (x - x_min) * be ** eta
            t2 = np.maximum(x_min, x - move)
            t3 = np.minimum(x_max, x + move)

            for i in range(n):
                if t1[i] <= t2[i]:
                    x_novo[i] = t2[i]
                elif t2[i] < t1[i] < t3[i]:
                    x_novo[i] = t1[i]
                else:
                    x_novo[i] = t3[i]

            # Restrição de volume
            if ((vol_atual - vol_inicial) + sens_vol @ (x_novo - x)) > 0:
                l1 = lmid
            else:
                l2 = lmid

        return x_novo

    def otimizar_estrutura(self, erro_max=0.1, passo=0.5, num_max_iteracoes=50):
        """Aplica o processo de otimização aos dados da estrutura.

        Se o passo for -1, apenas um valor de p será rodado
        """
        logger.info('Iniciando a otimização da estrutura')

        # Variáveis para o auxílio do salvamento dos resultados no arquivo.
        # Vetor com as densidades relativas dos elementos. Cada linha corresponde a uma iteração, sendo
        # que último valor (resultados_rho[-1]) correspondente às densidades relativas da topologia final.
        resultados_rho = []

        # Matriz que contém os demais resultados. Cada linha representa uma iteração, assim como
        # ocorre com `resultados_rho`. Os índices obededecem a seguinte ordem:
        # 0 -> Id da iteração para valores constantes de `p` e `beta`, ou `c` (do código abaixo).
        # 1 -> `p`.
        # 2 -> `beta`.
        # 3 -> Valor da função objetivo.
        # 4 -> Percentual de volume da estrutura após a otimização em relação ao volume inicial.
        # 5 -> Percentual de densidades intermediárias.
        # 6 -> Erro relacionado aos deslocamentos.
        # 7 -> Erro relacionado ao percentual de densidades intermediárias.
        resultados_gerais = []

        # Último vetor de deslocamentos
        u_ant = np.zeros(self.dados.num_nos() * 2)
        # Contador de iterações global
        it = 0

        def otimizar_p_beta_fixos(p, beta):
            """Otimiza as variáveis de projeto para um valor fixo de `p` e `beta`. Este processo é
            repetido para vários valores de `p` e de `beta` diferentes no método da continuação.

            Args:
                p: Coeficiente de penalização do modelo SIMP.
                beta: Coeficiente que define o nível de suavização da função Heaviside regularizada.
            """
            logger.info(f'{10 * "-"} {p=}\t {beta=} {10 * "-"}\n')

            nonlocal u_ant, it

            # Erro inicial dado em porcentagem.
            erro_u = 100
            # Percentual de densidadaes intermediárias inicial.
            di_ant = 100
            # Interface com Julia
            self.julia.p = self.p = p

            for c in np.arange(1, num_max_iteracoes + 1):
                it += 1

                # As densidades relativas dos elementos são densidades nodais apenas em otimização sem
                # esquema de projeção. Essa otimização é feita com `tecnica_otimizacao = 0` apenas.
                if self.tecnica_otimizacao != 0:
                    self.julia.rho = self.rho = self.calcular_densidades_elementos(beta)
                    u = self.deslocamentos_nodais()
                    self.x = self.atualizar_x(u, beta)
                else:
                    self.julia.rho = self.rho
                    u = self.deslocamentos_nodais()
                    self.rho = self.atualizar_x(u)

                # Cálculo dos erros.
                # Erro entre as duas densidades intermediárias mais atuais.
                di = self.percentual_densidades_intermediarias()
                erro_di = 100 * abs((di - di_ant) / di_ant)
                di_ant = di

                # Erro entre as duas normas mais recentes dos deslocamentos nodais.
                if c > 1:
                    norm_u_ant = np.linalg.norm(u_ant)
                    norm_u = np.linalg.norm(u)
                    erro_u = 100 * abs((norm_u_ant - norm_u) / norm_u_ant)
                u_ant = u.copy()

                # Log da iteração.
                vols = self.dados.volumes_elementos_solidos
                # Função objetivo.
                fo = self.flexibilidade_media(u)
                # Percentual do volume atual em relação ao volume inicial de material.
                vol_perc = (self.rho @ vols) / np.sum(vols)
                logger.info(f'i: {it}-{c}\t '
                            f'p: {p}\t '
                            f'beta: {beta:.2f}\t '
                            f'fo: {fo:.2f}\t '
                            f'vol: {vol_perc:.3f}%\t '
                            f'di: {di:.3f}%\t'
                            f'erro_u: {erro_u:.5f}%\t '
                            f'erro_di: {erro_di:.5f}%\t')

                # Adição dos resultados da iteração aos vetores de resultados.
                resultados_rho.append(self.rho.copy())
                resultados_gerais.append([c, p, beta, fo, vol_perc, di, erro_u, erro_di])

                # Aplicação dos critérios de convergência.
                # A convergência ocorre apenas pela variação dos deslocamentos nodais quando `beta = 0`.
                # Quando `beta` é diferente de 0, a convergência deixa de ocorrer pela variação dos
                # deslocamentos nodais e passa a ocorrer exclusivamente pela variação do percentual de
                # densidades intermediárias.
                # Em cada otimização, pe respeitado um valor mínimo de iterações `OC.NUM_MIN_ITERS`.
                if c >= OC.NUM_MIN_ITERS:
                    if beta > 0:
                        if erro_di <= erro_max:
                            logger.info(f'Convergência alcançada pelas densidades intermediárias!')
                            break
                    else:
                        if erro_u <= erro_max:
                            logger.info(f'Convergência alcançada pelos deslocamentos!')
                            break
            logger.success(f'Finalizada a otimização para p:{p} e beta:{beta}.\n')

        # Método da continuidade.
        # Vetor iterável de coeficientes de penalização.
        if passo != -1:
            ps = np.arange(1, self.p + 0.1, passo)
        else:
            ps = [self.p]

        # Continuidade no coeficiente de penalização com `beta = 0`.
        for p_i in ps:
            otimizar_p_beta_fixos(p_i, 0)

        # Continuidade em beta.
        if self.tecnica_otimizacao in OC.TECNICA_OTM_EP_HEAVISIDE:
            # Beta inicial. Adotado 1/3 para que seu primeiro valor seja 0.5.
            # 1.5 * 1/3 = 0.5
            beta_i = 1 / 3
            for i in range(num_max_iteracoes):
                dens_i = self.percentual_densidades_intermediarias()
                if beta_i < OC.BETA_MAX:
                    if (dens_i >= 5) or (i < 10):
                        beta_i = min(1.5 * beta_i, OC.BETA_MAX)
                    elif 1 <= dens_i < 5:
                        beta_i = min(beta_i + 5, OC.BETA_MAX)
                    else:
                        break
                    otimizar_p_beta_fixos(self.p, beta_i)
                else:
                    break

        # Salvar resultados no arquivo `.zip`.
        self.dados.salvar_arquivo_numpy(np.array(resultados_rho), 14)
        self.dados.salvar_arquivo_numpy(np.array(resultados_gerais), 15)
