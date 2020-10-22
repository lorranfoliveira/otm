import ezdxf
from ezdxf.groupby import groupby
from shapely.geometry import Point, Polygon, MultiPoint, MultiLineString, LineString
from shapely.ops import linemerge, nearest_points
from shapely.affinity import scale, translate
from shapely.wkb import dumps
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from otm.dados import Dados
import os
import pathlib
from otm.constantes import ARQUIVOS_DADOS_ZIP
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.sparse import lil_matrix
from scipy.spatial import KDTree
from loguru import logger
from random import uniform
import matplotlib.pyplot as plt
import zipfile

__all__ = ['Malha']


def _multipoint_para_numpy(pontos: MultiPoint) -> np.ndarray:
    """Converte um objeto MultiPoint em um vetor de coordenadas de pontos com dimensões nx2."""
    pontos_np = np.zeros((len(pontos.geoms), 2))
    for i, p in enumerate(pontos):
        pontos_np[i] = p.coords[0]
    return pontos_np


class Malha:
    """Classe responsável pela criação da malha de uma estrutura a partir de um arquivo DXF."""

    # Palavras chave
    LAYER_CARGA = 'carga'
    LAYER_FURO = 'furo'
    LAYER_GEOMETRIA = 'geometria'
    LAYER_RASTREADOR_NOS = 'rastreador_nos'
    LAYERS = [LAYER_CARGA, LAYER_FURO, LAYER_GEOMETRIA, LAYER_RASTREADOR_NOS]

    def __init__(self, dados: Dados, num_elems: int, tipo_malha='poligonal'):
        """Construtor.

        Args:
            dados: Objeto que manipula os dados dos arquivos do problema.
            num_elems: Número de elementos finitos que serão gerados.
            tipo_malha: Tipo de distribuição das sementes dos diagramas de Voronoi. Pode ser 'poligonal' ou 'retangular'.
        """
        self.dados = dados
        self.num_elems = num_elems
        self.tipo_malha = tipo_malha
        # Agrupamento das linhas do dxf por layer
        arquivo_dxf = self.dados.arquivo.with_suffix('.dxf').name
        self.layers: dict = groupby(entities=ezdxf.readfile(arquivo_dxf).modelspace(), dxfattrib='layer')

    def _dicionario_linhas_dxf(self) -> Dict[str, np.ndarray]:
        """Lê o arquivo dxf e retorna os pontos que formam as linhas contidas no layer.
        Deve ser utilizada apenas para a leitura da geometria externa e dos furos. Cada chave se refere
        a um layer e cada índice dessas chaves corresponde a uma lista com os dois nós que
        formam a linha em questão.

        Todos os pontos são aproximados para 3 casas decimais para se evitar erros de truncamento.

        Exemplo - {'geometria' : [[No1, No2], [No3, No4]]}
        """
        dic = {}
        for layer in self.layers:
            if any(layer.startswith(i) for i in Malha.LAYERS):
                n = len(self.layers[layer])
                dic[layer] = np.zeros((n, 4))
                for j in range(n):
                    dic[layer][j] = list(map(lambda v: round(v, 3), self.layers[layer][j].dxf.start[:2])) + \
                                    list(map(lambda v: round(v, 3), self.layers[layer][j].dxf.end[:2]))
        return dic

    def _contorno_geometria(self) -> MultiLineString:
        """Retorna um objeto contendo todas as linhas do contorno da geometria, excluindo-se os furos."""
        lin_layers = self._dicionario_linhas_dxf()
        linhas: List[Optional[LineString]] = len(lin_layers[Malha.LAYER_GEOMETRIA]) * [None]
        for i, lin in enumerate(lin_layers[Malha.LAYER_GEOMETRIA]):
            x1, y1, x2, y2 = lin
            linhas[i] = LineString([(x1, y1), (x2, y2)])

        return MultiLineString(linhas)

    def _salvar_nos_rastreados(self, vertices: np.ndarray):
        """Salva os nós rastreados pelo layer 'rastrear_nos{i}'"""
        lin_layers = self._dicionario_linhas_dxf()
        dmed = self.diametro_medio_elementos(self.poligono_estrutura())

        # KDTree
        kd = KDTree(vertices)

        multipontos = MultiPoint(vertices)

        nome_arq_txt = ARQUIVOS_DADOS_ZIP[16]

        if any(i.startswith(Malha.LAYER_RASTREADOR_NOS) for i in lin_layers):
            with open(nome_arq_txt, 'w') as arq:
                for rast in lin_layers:
                    if rast.startswith(Malha.LAYER_RASTREADOR_NOS):
                        linhas = []
                        for lin in lin_layers[rast]:
                            x1, y1, x2, y2 = lin
                            linhas.append(LineString([(x1, y1), (x2, y2)]))
                        multiline_i = MultiLineString(linhas)

                        # Identificar pontos
                        multiline_i = multiline_i.buffer(0.1 * dmed)
                        pontos_contorno = multiline_i.intersection(multipontos)

                        # Identificação dos nós do rastreador i
                        ids = []
                        if not isinstance(pontos_contorno, Point):
                            if len(pontos_contorno) > 0:
                                for p in pontos_contorno:
                                    ids.append(kd.query(p.coords[:][0])[1])
                            else:
                                pprox = nearest_points(multiline_i, multipontos)[1]
                                ids.append(kd.query(pprox.coords[:][0])[1])
                        else:
                            ids.append(kd.query(pontos_contorno.coords[:][0])[1])

                        # Salvar a identificação do ponto no arquivo
                        arq.write(f'{rast} = {sorted(ids)}\n')

            # Salvar no arquivo zip
            with zipfile.ZipFile(self.dados.arquivo.name, 'a', compression=zipfile.ZIP_DEFLATED) as arq_zip:
                arq_zip.write(pathlib.Path(nome_arq_txt).name)

            # Apagar o arquivo txt
            os.remove(nome_arq_txt)

    def _contorno_furos(self) -> List[MultiLineString]:
        """Retorna um objeto contendo todas as linhas do contorno dos furos da geometria."""
        furos = []
        lin_layers = self._dicionario_linhas_dxf()
        for fur in lin_layers:
            if fur.startswith(Malha.LAYER_FURO):
                linhas = []
                for lin in lin_layers[fur]:
                    x1, y1, x2, y2 = lin
                    linhas.append(LineString([(x1, y1), (x2, y2)]))
                furos.append(MultiLineString(linhas))

        return furos

    def poligono_estrutura(self) -> Polygon:
        """Retorna um polígono do shapely com a geometria e os furos."""
        furos = list(map(linemerge, self._contorno_furos()))
        return Polygon(linemerge(self._contorno_geometria()), furos)

    def diametro_medio_elementos(self, poligono: Polygon) -> float:
        """Retorna o diâmetro médio dos elementos poligonais com base na área de um círculo.

        Args:
            poligono: Polígono Shapely contendo o domínio da malha.
        """
        return 2 * np.sqrt((poligono.area / self.num_elems) / np.pi)

    def _pontos_iniciais(self, poligono: Polygon) -> MultiPoint:
        """Retorna os primeiros pontos aleatórios dentro da geometria.

        Args:
            poligono: Polígono Shapely contendo o domínio da malha.
        """
        logger.info(f'Criando {self.num_elems} pontos aleatórios dentro do domínio')

        xmin, ymin, xmax, ymax = poligono.bounds

        if self.tipo_malha == 'poligonal':
            pontos = np.zeros((self.num_elems, 2))
            c = 0
            while c < self.num_elems:
                p = Point(uniform(xmin, xmax), uniform(ymin, ymax))
                if poligono.contains(p):
                    pontos[c] = p.coords[:][0]
                    c += 1
        elif self.tipo_malha == 'retangular':
            a = xmax - xmin
            b = ymax - ymin
            razao = a / b
            n = self.num_elems

            num_pts_b = np.sqrt(n * b / a)
            num_pts_a = num_pts_b * razao
            while not (np.isclose(num_pts_a, int(num_pts_a)) and
                       np.isclose(num_pts_b, int(num_pts_b))):
                num_pts_b = int(num_pts_b) + 1
                num_pts_a = num_pts_b * razao

            num_pts_a = int(num_pts_a)
            num_pts_b = int(num_pts_b)
            n = num_pts_a * num_pts_b

            # Criação das coordenadas dos pontos
            pontos = np.zeros((n, 2))
            # Dimensões dos elementos
            d = a / num_pts_a
            # Posição dos pontos iniciais
            c = 0
            for j in range(num_pts_b):
                for i in range(num_pts_a):
                    pontos[c] = np.array([(xmin + d / 2) + i * d, (ymin + d / 2) + j * d])
                    c += 1
        else:
            raise ValueError(f'O tipo de malha especificado não é válido!')

        return MultiPoint(pontos)

    @staticmethod
    def _pontos_auxiliares(poligono: Polygon, contorno_discretizado: List[MultiLineString], pontos_inicio: MultiPoint,
                           espessura_buffer: float) -> MultiPoint:
        """Gera os pontos auxiliares refletidos refletidos em relação ao contorno do domínio. Os pontos
        dentro de uma determinada faixa (largura_faixa_espelhamento) são refletidos em relação ao ponto
        mais próximo que esteja sobre o contorno.

        Args:
            poligono: Polígono Shapely representando o domínio da malha.
            contorno_discretizado: Contorno do polígono discretizado em LineStrings separadas.
            pontos_inicio: Pontos de referência para o espelhamento.
            espessura_buffer: Espessura da faixa que parte das linhas de referência para o espelhamento.

        Returns: Conjuntos de pontos auxiliares obtidos pelo espelhamento.

        """
        pts_aux = []
        contorno = []

        for i in contorno_discretizado:
            contorno += list(i.geoms)

        cont_mult = MultiLineString(contorno)

        for linha in contorno:
            eta = 0.9
            # Criação do buffer
            lin_buff = linha.buffer(espessura_buffer, cap_style=2)
            # Pontos contidos no buffer
            pts_buff = lin_buff.intersection(pontos_inicio)
            # Iteração sobre os pontos para o espelhamento individual
            if not isinstance(pts_buff, (MultiPoint, Point)):
                continue
            elif isinstance(pts_buff, Point):
                pts_buff = MultiPoint([pts_buff])

            for p in pts_buff:
                # Ponto da linha mais próximo de "p"
                p_prox = nearest_points(linha, p)[0]
                # Espelhamento do ponto em relação ao ponto "pprox", que configura no ponto de
                # intersecção na linha entre o ponto original e seu espelho
                p_esp = scale(p, -1, -1, origin=p_prox)
                # O ponto espelhado não pode estar contido no polígono
                if not poligono.contains(p_esp):
                    # O ponto da geometria mais próximo do ponto espelhado deve, necessariamente, ser o ponto
                    # de espelhamento.
                    d1 = p.distance(p_prox)
                    d2 = p_esp.distance(cont_mult)
                    if d1 > eta * d2:
                        pts_aux.append(p_esp)

        return MultiPoint(pts_aux)

    @staticmethod
    def _excluir_elementos_fora_do_dominio(voronoi: Voronoi, poligono: Polygon) -> Tuple[List[List[int]], np.ndarray]:
        """Separa os elementos contidos no domínio da estrutura, descartando os externos à mesma.
        Os elementos que possuírem maior parte de sua área dentro do domínio serão conservados,
        enquanto que os demais serão descartados.

        Args:
            voronoi: Diagrama de voronoi bruto gerado pelo scipy.
            poligono: Polígono representando o domínio da geometria.

        Returns: Listas com os vértices que formam cada elemento e um vetor de pontos com
            todos os vértices, incluindo os que formam apenas os elementos que foram excluídos.

        """
        logger.info('Excluindo os elementos majoritariamente externos ao domínio')

        regioes = []

        # Criação dos elementos como polígonos shapely
        c = 0
        for reg in (voronoi.regions[i] for i in voronoi.point_region):
            if -1 not in reg:
                coords = [(voronoi.vertices[i][0], voronoi.vertices[i][1]) for i in reg]
                # Célula de Voronoi completa
                # São válidas apenas as células internas à geometria ou que tocam a borda
                if not (cel := Polygon(coords)).disjoint(poligono):
                    if cel.intersects(poligono):
                        c += 1
                        # Parcela da célula que está contida no polígono
                        cel_int = cel.intersection(poligono)
                        # Parcela da célula que está fora do polígono
                        cel_ext = cel.difference(poligono)
                        if cel_int.area < cel_ext.area:
                            continue

                    regioes.append(reg)

        return regioes, voronoi.vertices

    @staticmethod
    def _colapsar_pontos_proximos(regioes: List[List[int]], vertices: np.ndarray, diam_med_elems: float,
                                  fator_tol: float = 0.05) -> Tuple[List[List[int]], np.ndarray]:
        """Colapsa os pontos que são suficientemente próximos para um terceiro ponto posicionado no centroide
        dos dois primeiros. Também exclui os vértices não utilizados e renumera os vértices de regiões.

        Args:
            regioes: Regiões dos elementos.
            vertices: Vértices que compõem o diagrama de Voronoi bruto gerado pelo ScyPy.
            diam_med_elems: Diâmetro médio dos elementos finitos.
            fator_tol: Este fator multiplicado pelo raio médio resulta na distância mínima tolerável
                entre dois pontos.

        Returns: Regiões dos elementos e os vértices utilizados em sua formas e valores finais.
        """
        dist_min = fator_tol * diam_med_elems
        logger.info(f'Colapsando os pontos que estão mais próximos que {fator_tol * 100}% do raio médio')

        pontos_excluidos = {}
        pontos_utilizados = []
        # Colapsar vértices próximos
        for i, reg in enumerate(regioes):
            # Esta cópia da região é utilizada para se atualizar a numeração dos vértices, tendo em vista
            # que alguns serão colapsados
            regiao_nova = reg.copy()
            # Iteração sobre os pontos da região
            for j, id_ponto_ref in enumerate(reg):
                if id_ponto_ref in pontos_excluidos:
                    id_ponto_ref = pontos_excluidos[id_ponto_ref]
                    regiao_nova[j] = id_ponto_ref

                # Ponto de referência
                ponto_ref = vertices[id_ponto_ref]
                # Os pontos cujas distâncias entre si forem calculadas terão sua identificação
                # armazenada na lista abaixo. Isto será utilizado para evitar o cálculo da distância
                # entre os mesmos pontos mais de uma vez
                pares_analisados = []
                # Verificação se há algum ponto na região que seja mais próximo que a tolerância imposta.
                for k, id_ponto_compar in enumerate(reg):
                    if id_ponto_compar in pontos_excluidos:
                        id_ponto_compar = pontos_excluidos[id_ponto_compar]
                        regiao_nova[k] = id_ponto_compar

                    # O ponto de comparação não pode ser o mesmo ponto de referência
                    if id_ponto_ref != id_ponto_compar:
                        vet_pts = [id_ponto_ref, id_ponto_compar]
                        # Verifica se o par de pontos já teve sua distância calculada pra evitar retrabalho
                        if not (vet_pts in pares_analisados or vet_pts[::-1] in pares_analisados):
                            pares_analisados.append(vet_pts.copy())
                            ponto_compar = vertices[id_ponto_compar]

                            # Cálculo da distância entre os pontos e verificação do atendimento à tolerância
                            if np.linalg.norm(ponto_ref - ponto_compar) < dist_min:
                                # Se a tolerância não for atendida, os pontos serão colapsados para o
                                # ponto que fica no centroide dos dois
                                ponto_medio = np.array([np.mean([ponto_ref[0], ponto_compar[0]]),
                                                        np.mean([ponto_ref[1], ponto_compar[1]])])
                                # O ponto cuja identificação possuir menor valor cederá sua identificação
                                # para o novo ponto formado
                                menor = min(id_ponto_ref, id_ponto_compar)
                                maior = max(id_ponto_ref, id_ponto_compar)
                                # O ponto de maior número de identificação é excluído e o ponto de menor
                                # número de identificação é atualizado.
                                pontos_excluidos[maior] = menor

                                vertices[menor] = ponto_medio
                                pontos_utilizados.append(menor)
                                continue
                            else:
                                if id_ponto_ref not in pontos_utilizados:
                                    pontos_utilizados.append(id_ponto_ref)
                                if id_ponto_compar not in pontos_utilizados:
                                    pontos_utilizados.append(id_ponto_compar)

            # Exclusão dos vértices repetidos em cada region
            if len(idt := set(regiao_nova)) != len(regiao_nova):
                for pt in idt:
                    while len(list(filter(lambda x: x == pt, regiao_nova))) > 1:
                        del regiao_nova[regiao_nova.index(pt)]

            regioes[i] = regiao_nova.copy()

        # Exclusão dos pontos excluídos que permaneceram em pontos utilizados. Nenhuma das chaves
        # de pontos excluídos é um ponto utilizado válido, mas apenas uma referência.
        for i in pontos_excluidos:
            if i in pontos_utilizados:
                del pontos_utilizados[pontos_utilizados.index(i)]

        # Renumeração dos vértices dentro das regiões por causa das exclusões
        pontos_utilizados.sort()

        logger.info('Reenumerando os números dos nós da malha')

        for i in range(len(regioes)):
            indices_repetidos = []
            for j in range(len(regioes[i])):
                if regioes[i][j] in pontos_excluidos:
                    if (p := pontos_excluidos[regioes[i][j]]) not in regioes[i]:
                        regioes[i][j] = pontos_utilizados.index(p)
                    else:
                        indices_repetidos.append(j)
                else:
                    regioes[i][j] = pontos_utilizados.index(regioes[i][j])
            if bool(indices_repetidos):
                for idx in indices_repetidos:
                    del regioes[i][idx]

        return regioes, vertices[np.ix_(pontos_utilizados)]

    def _criar_diagrama_voronoi(self, esp_faixa_espelhamento: float = 2.5, num_max_iteracoes=200, num_min_iteracoes=30,
                                erro_max=1e-3) -> Voronoi:
        """Cria o diagrama de Voronoi bruto para posterior extração de dados. É utilizado o algoritmo
        de Lloyd para a regularização da malha."""

        logger.info('Iniciando a criação do diagrama de Voronoi')

        poligono = self.poligono_estrutura()
        pontos_inic = self._pontos_iniciais(poligono)
        contorno_disc = [self._contorno_geometria()] + self._contorno_furos()
        dim_med_els = self.diametro_medio_elementos(poligono)
        erro = 10

        fig, ax = plt.subplots()
        win = plt.get_current_fig_manager()
        win.window.state('zoomed')
        plt.show(block=False)
        plt.title = 'Malha por diagrama de Voronoi'
        ax.axis('equal')

        vor = None

        logger.info('Aplicando o algoritmo de Lloyd')
        # erro > tol and
        it = 0
        while ((it := it + 1) <= num_max_iteracoes) and (erro >= erro_max) or (it <= num_min_iteracoes):
            logger.debug(f'Iteração: {it}\t Erro: {100 * erro:.4f}%')

            ax.clear()
            ax.set_title(f'Iteração: {it:4}      Erro: {100 * erro:10.4f}%')

            pontos_aux = _multipoint_para_numpy(
                self._pontos_auxiliares(poligono, contorno_disc, pontos_inic, esp_faixa_espelhamento * dim_med_els))
            pontos_inic_np = _multipoint_para_numpy(pontos_inic)
            vor = Voronoi(np.concatenate((pontos_inic_np, pontos_aux)))

            voronoi_plot_2d(vor, ax, show_vertices=False, show_points=False, point_size=2)

            pontos_inic_ant_np = pontos_inic_np.copy()
            pontos_inic_np = np.zeros(pontos_inic_np.shape)
            pontos_inic_list = len(pontos_inic) * [None]
            n = -1

            for reg, p in zip((vor.regions[i] for i in vor.point_region), vor.points):
                n += 1
                psh = Point(*p)
                if not poligono.disjoint(psh):
                    coords = [(vor.vertices[i][0], vor.vertices[i][1]) for i in reg]
                    cel = Polygon(coords)
                    c = cel.centroid
                    pontos_inic_np[n, :] = [c.x, c.y]
                    pontos_inic_list[n] = c

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            erro = np.linalg.norm((pontos_inic_np - pontos_inic_ant_np) / pontos_inic_ant_np)
            pontos_inic = MultiPoint(pontos_inic_list)

            if self.tipo_malha == 'retangular':
                break

        # plt.savefig(f'{self.arquivo_dxf.replace("dxf", "svg")}')
        # plt.show()
        plt.close('all')

        return vor

    def _criar_trelica_hiperconectada(self, poligono: Polygon, regioes: List[List[int]], vertices: np.ndarray,
                                      d: int = 1, nivel_conect: int = 2, espacamento: int = 2) -> tuple:
        """

        Args:
            poligono:
            vertices:
            nivel_conect:
            espacamento:
            d: Distância entre os nós da malha base para os elementos de barra. O espaçamento que separa
                os nós que serão efetivamente utilizados é dado sobre a malha separada por d. Caso d==None,
                adota-se automaticamente para d o diâmetro médio do elemento

        Returns:

        """
        logger.info(f'Gerando treliça hiperconectada')

        elementos = {}
        dmed = self.diametro_medio_elementos(poligono)
        contorno_sem_buff = poligono.boundary

        def posicao_matriz_para_ponto(i, j) -> tuple:
            """Retorna as coordenadas de um ponto em função de sua posição dentro da matriz
            da ground structure."""
            nonlocal xmin, ymax, d
            return xmin + d * (j + 1), ymax - d * (i + 1)

        def coordenadas_para_id_ponto(ponto: tuple) -> int:
            """Retorna o número de um ponto em função de suas coordenadas.
            O ponto deve estar contido em vertices."""
            nonlocal vertices
            return np.where(np.logical_and(np.isclose(vertices[:, 0], ponto[0]),
                                           np.isclose(vertices[:, 1], ponto[1])))[0][0]

        # Extremidades da geometria
        xmin, ymin, xmax, ymax = poligono.bounds
        dx = xmax - xmin
        dy = ymax - ymin
        # Corretor de erros numéricos
        t = 0.1 * dmed / 2
        contorno_com_buff = contorno_sem_buff.buffer(t)
        # Matriz de pontos
        malha_gs = lil_matrix((round((dy - 2 * d) / d + 1), round((dx - d) / d + 1)), dtype=int)

        logger.info('Compatibilizando a malha de polígonos com a de barras')
        # Adaptação dos pontos da malha de polígonos aos pontos da groun structure
        for i in range(0, malha_gs.shape[0], espacamento):
            for j in range(0, malha_gs.shape[1], espacamento):
                # Ponto de referência na ground structure
                ponto_gs = posicao_matriz_para_ponto(i, j)
                ponto_gs_geo = Point(*ponto_gs)
                for elem in regioes:
                    # Verifica se o ponto está fora do polígono convexo que representa o elemento elem
                    if (poli_elem := Polygon([vertices[n] for n in elem])).contains(ponto_gs_geo):
                        # TODO Verificar depois
                        # Modificar apenas pontos
                        if contorno_sem_buff.distance(ponto_gs_geo) >= 0.9 * d:
                            # Ponto do elemento que é mais próximo do ponto da malha da ground structure
                            ponto_prox = nearest_points(MultiPoint(poli_elem.boundary.coords), ponto_gs_geo)[0]
                            # Diferenças em x e em y entre os pontos
                            dxt = ponto_gs_geo.x - ponto_prox.x
                            dyt = ponto_gs_geo.y - ponto_prox.y
                            # Fator de redução do elemento em função do diâmetro médio dos elementos
                            # e do deslocamento feito pelo elemento
                            fat_red = 1 - (ponto_gs_geo.distance(ponto_prox) / dmed)
                            # Novo elemento reduzido e deslocado
                            poli_elem_transl = translate(poli_elem, dxt, dyt)  # Translada
                            poli_elem_novo = scale(poli_elem_transl, fat_red, fat_red, origin=ponto_gs_geo)  # Reduz
                            # Atualização dos pontos para as novas coordenadas (apagando a última que é repetida)
                            coords_novas = poli_elem_novo.boundary.coords[:-1]
                            # coords_antigas = poli_elem.boundary.coords[:-1]
                            for ii, vv in enumerate(elem):
                                # Impedir que os pontos situados sobre o contorno mudem para uma posição fora
                                # do contorno
                                # if np.isclose(contorno_sem_buff.distance(Point(*coords_antigas[ii])), 0):
                                #     vertices[vv] = coords_antigas[ii]
                                # else:
                                # TODO Verificar depois
                                vertices[vv] = coords_novas[ii]

        # TODO Unir ao procedimento anterior para economizar processamento
        for i in range(0, malha_gs.shape[0], espacamento):
            for j in range(0, malha_gs.shape[1], espacamento):
                if not poligono.disjoint(pt := Point(*(posicao_matriz_para_ponto(i, j)))):
                    if pt.distance(contorno_sem_buff) >= 0.9 * d:
                        try:
                            malha_gs[i, j] = coordenadas_para_id_ponto(pt.coords[0])
                        except IndexError:
                            pass

        logger.info('Gerando os elementos de barra')
        # Obtenção dos elementos de barra da ground structure
        for i in range(0, malha_gs.shape[0], espacamento):
            for j in range(0, malha_gs.shape[1], espacamento):
                if (p1_num := malha_gs[i, j]) != 0:
                    # Ponto de referência
                    p1 = vertices[p1_num]
                    for m in range(i - nivel_conect * espacamento, i + nivel_conect * espacamento + 1):
                        if not (0 <= m <= malha_gs.shape[0] - 1):
                            continue
                        for n in range(j - nivel_conect * espacamento, j + nivel_conect * espacamento + 1):
                            if 0 <= n <= malha_gs.shape[1] - 1:
                                if (p2_num := malha_gs[m, n]) != 0:
                                    # Ponto de comparação
                                    p2 = vertices[p2_num]
                                    if not np.allclose(p1, p2):
                                        linha = LineString([p1, p2])
                                        if not any(linha.contains(e) for e in elementos.values()):
                                            if poligono.contains(linha) or contorno_com_buff.contains(linha):
                                                elementos[(p1_num, p2_num)] = linha

        return elementos, vertices

    @staticmethod
    def verificar_vertices_ociosos(vertices, elementos) -> tuple:
        # Apagar os vértices que não são utilizados
        verts_usados = []
        for e in elementos:
            verts_usados = list(set(verts_usados + e))
        verts_usados = np.array(verts_usados)
        verts_nao_usados = np.setdiff1d(np.array(range(np.max(verts_usados) + 1)), verts_usados)

        # Corrigindo os vértices dos elementos
        if verts_nao_usados.size > 0:
            min_vert_nao_usado = np.min(verts_nao_usados)
            for i in range(len(elementos)):
                for j in range(len(elementos[i])):
                    if (el := elementos[i][j]) > min_vert_nao_usado:
                        elementos[i][j] -= len(list(filter(lambda x: x < el, verts_nao_usados)))

        return vertices[verts_usados], elementos

    @staticmethod
    def _nos_do_elemento_em_sentido_anti_horario(nos: np.ndarray) -> bool:
        """Verifica se 3 nós estão formando uma linha em sentido anti-horário. Se sim, retorna True"""
        tmp = np.ones((3, 3))
        tmp[:, 1:] = nos
        return np.linalg.det(tmp) > 0

    def corrigir_sentido_anti_horario_dos_nos_elementos(self, vertices, elementos) -> List[list]:
        """Verifica se os nós dos elementos estão dispostos em sentido anti-horário e corrige quando necessário."""
        els_tmp = elementos.copy()
        for i, e in enumerate(elementos):
            nos_e = vertices[e[:3]]
            if not self._nos_do_elemento_em_sentido_anti_horario(nos_e):
                els_tmp[i].reverse()

        return els_tmp

    def salvar_malha(self, vertices, elementos):
        """Salva os dados da malha em um arquivo zip"""

        logger.debug(f'Criando o arquivo de entrada de dados "{self.dados.arquivo.name}"')

        # Salvar arquivos numpy
        # Salvar elementos
        arq_elementos = []
        for el in elementos:
            arq_elementos.append(np.array(el))
        self.dados.salvar_arquivo_numpy(arq_elementos, 0)
        # Salvar nós
        self.dados.salvar_arquivo_numpy(vertices, 1)
        # Salvar polígono do domínio estendido
        with open(ARQUIVOS_DADOS_ZIP[10], 'wb') as arq_wkb:
            arq_wkb.write(dumps(self.poligono_estrutura()))
        self.dados.salvar_arquivo_generico_em_zip(ARQUIVOS_DADOS_ZIP[10])
        # Salvar arquivo `.dxf`.
        self.dados.salvar_arquivo_generico_em_zip(self.dados.arquivo.with_suffix('.dxf').name)

    def criar_malha(self, esp_faixa_espelhamento: float, num_iteracoes: int,
                    tol_colapso_pontos: float = 0.2) -> tuple:
        """Executa todas as funções necessárias para a criação da malha completa do domínio.

        Args:
            esp_faixa_espelhamento: Espessura da faixa de espelhamento (auxiliar para os pontos auxiliares).
            num_iteracoes: Número máximo de iterações.
            tol_colapso_pontos: Fator de tolerância para o colapso de pontos próximos.
            nivel_conect_gs: Nível de conectividade da treliça hiperconectada.
            espacamento_gs: Quantidade de vezes em que o espaçamento dos nós da treliça hiperconectada
                é maior que o dos elementos poligonais regulares.
            d: Distância entre os pontos da malha primária da ground structure.

        Returns: Regiões dos elementos e os vértices utilizados em sua formas e valores finais.

        """
        logger.info(f'Iniciando a criação da malha de elementos finitos do domínio "{self.dados.arquivo.stem}"')

        poligono = self.poligono_estrutura()
        diam_medio_elems = self.diametro_medio_elementos(poligono)

        vor = self._criar_diagrama_voronoi(esp_faixa_espelhamento, num_iteracoes)

        elementos, vertices = self._excluir_elementos_fora_do_dominio(vor, poligono)

        # Há a necessidade de colapso dos pontos próximos apenas em malhas com elementos poligonais genéricos
        if self.tipo_malha == 'poligonal':
            elementos, vertices = self._colapsar_pontos_proximos(elementos, vertices, diam_medio_elems,
                                                                 tol_colapso_pontos)

        vertices_final, elementos = self.verificar_vertices_ociosos(vertices, elementos)
        elementos_final = self.corrigir_sentido_anti_horario_dos_nos_elementos(vertices_final, elementos)

        # Adição dos elementos da treliça hiperconectada
        elementos_barra, vertices_final = self._criar_trelica_hiperconectada(poligono, elementos_final,
                                                                             vertices_final, d=4, nivel_conect=2,
                                                                             espacamento=21)

        elementos_final += elementos_barra
        logger.success(f'Malha finalizada com {len(elementos_final)} elementos, {len(vertices_final)} nós e '
                       f'{len(vertices_final) * 2} graus de liberdade')

        self.salvar_malha(vertices_final, elementos_final)
        self._salvar_nos_rastreados(vertices_final)

        return elementos_final, vertices_final
