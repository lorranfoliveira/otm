from matplotlib.patches import Polygon, Rectangle, Patch, Arrow
from matplotlib.path import Path
from matplotlib.collections import PatchCollection, PathCollection
import matplotlib.pyplot as plt
import numpy as np
from copy import copy, deepcopy
from loguru import logger
from otm.mef.elementos import EPTQ4, TP2


class Plot:
    """Classe responsável pela plotagem da estrutura e dos resultados da análise"""

    def __init__(self, estrutura):
        self.estrutura = estrutura

    def plotar_estrutura(self, comprimento_apoios, comprimento_cargas):
        logger.info('Iniciando a plotagem da estrutura inicial')

        fig, ax = plt.subplots()
        patches_elems_q4 = []
        patches_elems_tp2 = []
        patches_aps = []
        patches_cargas = []

        # Apoios
        logger.debug('Gerando os desenhos das cargas e dos apoios')
        for n in self.estrutura.nos:
            if n.apoiado():
                if n.dict_apoios[0] == 1:
                    patches_aps.append(Path([[n.x, n.y], [n.x - comprimento_apoios, n.y]],
                                            [Path.MOVETO, Path.LINETO]))
                if n.dict_apoios[1] == 1:
                    patches_aps.append(Path([[n.x, n.y], [n.x, n.y - comprimento_apoios]],
                                            [Path.MOVETO, Path.LINETO]))
            # Cargas
            if n.carregado():
                fatx = 1 if n.dict_forcas[0] >= 0 else -1
                faty = 1 if n.dict_forcas[1] >= 0 else -1

                if n.dict_forcas[0] != 0:
                    patches_cargas.append(Arrow(n.x, n.y, fatx * comprimento_cargas, 0))
                if n.dict_forcas[1] != 0:
                    patches_cargas.append(Arrow(n.x, n.y, 0, faty * comprimento_cargas))

        # Elementos
        logger.debug('Gerando os desenhos dos elementos')
        for e in self.estrutura.vetor_elementos:
            if isinstance(e, EPTQ4):
                p = np.array([[no.x, no.y] for no in e.nos])
                patches_elems_q4.append(Polygon(p))
            elif isinstance(e, TP2):
                xi = e.nos[0].x
                yi = e.nos[0].y
                xf = e.nos[1].x
                yf = e.nos[1].y

                p = patches_elems_tp2.append(Path([[xi, yi], [xf, yf]],
                                                  [Path.MOVETO, Path.LINETO]))
                patches_elems_tp2.append(p)

        logger.debug('Criando as coleções de figuras')
        col_elems_q4 = PatchCollection(patches_elems_q4, facecolors='lightblue', edgecolors='blue',
                                       linewidths=1)
        col_elems_tp2 = PathCollection(patches_elems_tp2, edgecolors='black', linewidths=1)
        col_aps = PathCollection(patches_aps, edgecolors='red', linewidths=2)
        col_cargas = PatchCollection(patches_cargas, facecolors='magenta', edgecolors='magenta',
                                     linewidths=2)

        logger.debug('Adicionando as coleções no gráfico')
        ax.add_collection(col_elems_q4)
        ax.add_collection(col_elems_tp2)
        ax.add_collection(col_aps)
        ax.add_collection(col_cargas)
        ax.axis('equal')
        ax.margins(0.5, 0.5)

        logger.success('Gráfico finalizado')

        plt.show()

    def plotar_deslocamentos(self, deslocamentos, comprimento_apoios, comprimento_cargas, escala=100):
        logger.info('Iniciando a plotagem da estrutura deformada')

        u = deslocamentos
        fig, ax = plt.subplots()

        # Elementos deslocados
        patches_elems_q4 = []
        patches_elems_tp2 = []
        patches_aps = []
        patches_cargas = []

        # Nós deslocados
        nos_desl = np.zeros((len(self.estrutura.nos), 2))

        # Adição dos deslocamentos nos nós
        logger.debug('Desenhando as cargas, os apoios e adicionando os deslocamentos nos nós')
        for n in self.estrutura.nos:
            gl = n.graus_lib()

            # Nós deslocados
            xd = nos_desl[n.idt - 1, 0] = n.x + escala * u[gl[0] - 1]
            yd = nos_desl[n.idt - 1, 1] = n.y + escala * u[gl[1] - 1]

            # Apoios
            if n.apoiado():
                if n.dict_apoios[0] == 1:
                    patches_aps.append(Path([[xd, yd], [xd - comprimento_apoios, yd]],
                                            [Path.MOVETO, Path.LINETO]))
                if n.dict_apoios[1] == 1:
                    patches_aps.append(Path([[xd, yd], [xd, yd - comprimento_apoios]],
                                            [Path.MOVETO, Path.LINETO]))
            # Cargas
            if n.carregado():
                fatx = 1 if n.dict_forcas[0] >= 0 else -1
                faty = 1 if n.dict_forcas[1] >= 0 else -1

                if n.dict_forcas[0] != 0:
                    patches_cargas.append(Arrow(xd, yd, fatx * comprimento_cargas, 0))
                if n.dict_forcas[1] != 0:
                    patches_cargas.append(Arrow(xd, yd, 0, faty * comprimento_cargas))

        # Elementos
        logger.debug('Gerando os desenhos dos elementos')
        for e in self.estrutura.vetor_elementos:
            if isinstance(e, EPTQ4):
                p = np.array([[nos_desl[no.idt - 1, 0], nos_desl[no.idt - 1, 1]] for no in e.nos])
                patches_elems_q4.append(Polygon(p))
            elif isinstance(e, TP2):
                xi = nos_desl[e.nos[0].idt - 1, 0]
                yi = nos_desl[e.nos[0].idt - 1, 1]
                xf = nos_desl[e.nos[1].idt - 1, 0]
                yf = nos_desl[e.nos[1].idt - 1, 1]

                p = patches_elems_tp2.append(Path([[xi, yi], [xf, yf]],
                                                  [Path.MOVETO, Path.LINETO]))
                patches_elems_tp2.append(p)

        logger.debug('Criando as coleções de figuras')
        col_elems_q4 = PatchCollection(patches_elems_q4, facecolors='lightblue', edgecolors='black',
                                       linewidths=1)
        col_elems_tp2 = PathCollection(patches_elems_tp2, edgecolors='purple', linewidths=1)
        col_aps = PathCollection(patches_aps, edgecolors='red', linewidths=2)
        col_cargas = PatchCollection(patches_cargas, facecolors='magenta', edgecolors='magenta',
                                     linewidths=2)

        logger.debug('Adicionando as coleções no gráfico')
        ax.add_collection(col_elems_q4)
        ax.add_collection(col_elems_tp2)
        ax.add_collection(col_aps)
        ax.add_collection(col_cargas)
        ax.axis('equal')
        ax.margins(0.5, 0.5)

        logger.success('Gráfico finalizado')

        plt.show()
