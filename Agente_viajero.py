import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import networkx as nx

def generarArbol(mtx):
    nNodos = len(mtx) 
    visitado = []  # List to track the visitado nodes
    vertices = []  # To store the MST edges
    root = 0  # Starting node
    visitado.append(root)
    peso = 0

    # Repeat until all nodes are in the visitado list (MST)
    while len(visitado) < nNodos:
        menor = 1000  # Start with an infinite value for the smallest edge
        u, v = -1, -1  # Variables to store the selected nodes of the smallest edge

        # Loop through all visitado nodes to find the smallest edge to an unvisitado node
        for i in visitado:
            for j in range(nNodos):
                if j not in visitado and mtx[i][j] > 0 and mtx[i][j] < menor:
                    menor = mtx[i][j]
                    
                    nIni, nFin = i, j
        peso += menor

        # Add the found smallest edge to the MST and mark the node as visitado
        if nFin != -1:
            vertices.append((nIni, nFin, menor))  # Append the edge (u, v) with peso 'menor'
            visitado.append(nFin)  # Add the new node to the visitado list

    return vertices,peso


def mtxAdyacencia(mst):
    listaAd = defaultdict(list)
    
    # Loop over all edges and add to adjacency list
    for nIni, nFin,_ in mst:
        listaAd[nIni].append(nFin)
        listaAd[nFin].append(nIni)  # Since it's an undirected graph, add both directions
    
    return listaAd

def preOrden(listaAd, nodo, visitado):
    visitado.add(nodo)
    preOrd=[]
    preOrd.append(nodo)
    #print(nodo, end=" ")  # Visit the nodo
    
    # Recur for all the vertices adjacent to this vertex
    for vecino in listaAd[nodo]:
        if vecino not in visitado:
            preOrd.extend(preOrden(listaAd, vecino, visitado))
    return preOrd

def crearMtx(n):
    A = np.random.randint(low=0, high =20,size =(n,n))
    A_T=A.transpose()
    B = (A + A_T)/2
    #print(A)
    B = B.astype('int')
    np.fill_diagonal(B,0)
    return B


def mostrarMtx(mtx, n):
    mtx = B
    
    # Step 1: Create row and column labels (axis values)
    nomFil = [f"{i}" for i in range(mtx.shape[0])]
    nomCol = [f"{j}" for j in range(mtx.shape[1])]

    # Step 2: Create a figure and axis
    fig,ax = plt.subplots()

    # Step 3: Hide the axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)  # Remove the border

    # Step 4: Create the table from the NumPy array with row and column labels
    table = ax.table(cellText=mtx, rowLabels=nomFil, colLabels=nomCol, loc='center', cellLoc='center')

    # Step 5: Adjust the layout to fit the table
    if n > 12:
        table.scale(1.5, 3)  # Adjust scaling for larger matrices
        table.auto_set_font_size(False)
        table.set_fontsize(12)
    else:
        table.scale(1, 3)  # Adjust scaling for smaller matrices
        table.auto_set_font_size(False)
        table.set_fontsize(12)

    # Show the table
    plt.tight_layout()
    plt.savefig('matriz.png',format='png',bbox_inches = 'tight')
    plt.close()

def mostrargrafo(B,mst,preO):
    # Step 1: Define a symmetric peso matrix (example with 4 nodes)
    mtx = B

    # Step 2: Create a graph from the peso matrix
    G = nx.Graph()
    bordes = mst

    hamiltoniano=preO
    # Add edges with weights based on the matrix
    n = mtx.shape[0]  # Number of nodes
    for i in range(n):
        for j in range(i+1, n):  # Only look at upper triangular matrix (since it's symmetric)
            if mtx[i, j] != 0:  # Add an edge only if the peso is non-zero
                G.add_edge(i, j, peso=mtx[i, j])
    G.add_weighted_edges_from(bordes)
    # Step 3: Draw the graph
    pos = nx.spring_layout(G)  # Use spring layout for node positions
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=700, font_size=15)
    bordesH = list(zip(hamiltoniano, hamiltoniano[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=bordesH, edge_color="red", width=2)
    nx.draw_networkx_nodes(G, pos, nodelist=hamiltoniano, node_color="violet", node_size=700)

    # Step 4: Draw edge labels (weights)
    edge_labels = {(i, j): f'{d["peso"]}' for i, j, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Show the plot
    plt.savefig('grafo.png',format='png')
    plt.close()
    
def grafoLimpio(B,mst,preO):
    mtx = B

    # Step 2: Create a graph from the peso matrix
    G = nx.Graph()
    bordes=mst

    hamiltoniano=preO
    # Add edges with weights based on the matrix
    n = mtx.shape[0]  # Number of nodes
    for i in range(n):
        for j in range(i+1, n):  # Only look at upper triangular matrix (since it's symmetric)
            if mtx[i, j] != 0:  # Add an edge only if the peso is non-zero
                G.add_edge(i, j, peso=mtx[i, j])
    G.add_weighted_edges_from(bordes)
    # Step 3: Draw the graph
    pos = nx.spring_layout(G)  # Use spring layout for node positions
    bordesH = list(zip(hamiltoniano, hamiltoniano[1:]))
    edges_to_remove = [edge for edge in G.edges() if edge not in bordesH and (edge[1], edge[0]) not in bordesH]
    G.remove_edges_from(edges_to_remove)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=700, font_size=15)
    nx.draw_networkx_edges(G, pos, edgelist=bordesH, edge_color="red", width=2)
    nx.draw_networkx_nodes(G, pos,nodelist=hamiltoniano, node_color="violet", node_size=700)
    pesos = {(i, j): f'{d["peso"]}' for i, j, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=pesos, font_size=8)

    plt.savefig('grafoLimpio.png',format='png')
    plt.close()


while True:
    nstr = input("Ingrese un numero entre 8 y 16: ")
    n = int(nstr)
    if n>=8 and n<=16:
        break
    else:
        print("Revise el numero ingresado")

B=crearMtx(n)
mostrarMtx(B,n)

mst,peso = generarArbol(B)
print(mst)
listaAd = mtxAdyacencia(mst)
visitado = set()  # To keep track of visitado nodes
preO = preOrden(listaAd, 0, visitado)
preO.append(preO[0])
print(peso)
peso += B[preO[-2]][preO[-1]]
print(preO,"\n",peso)
mostrargrafo(B,mst,preO)
grafoLimpio(B,mst,preO)