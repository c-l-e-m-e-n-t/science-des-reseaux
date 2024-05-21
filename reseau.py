import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import requests
import osmnx as ox
import random
from osmnx import graph_from_place

place_query = {'city':'Toulouse', 'state':'Toulouse', 'country':'France'}

try:
    G = graph_from_place(place_query, network_type='drive')
    G_copy = G.copy()
except requests.exceptions.ConnectTimeout:
    print("The request timed out. Please check your internet connection or try again later.")

#affichage du réseau routier en graph networkx
print("Nombre de noeuds : ", G.number_of_nodes())
print("Nombre d'arêtes : ", G.number_of_edges())

# Percolation
for node in list(G.nodes()):
    if np.random.rand() < 0.5:
        G.remove_node(node)

# Identifier les clusters
clusters = list(nx.connected_components(G.to_undirected()))

# Trouver le plus grand cluster
largest_cluster = max(clusters, key=len)
node_colors = ["red" if node in largest_cluster else "gray" for node in G.nodes()]

# Calculer la probabilité qu'un point se trouve dans le plus grand cluster
probability = len(largest_cluster) / (100 * 100)
print('La probabilité qu\'un point se trouve dans le plus grand cluster est :', probability)

pos = {node:node for node in G.nodes()}
# ox.plot_graph(G, node_color=node_colors, node_size=10, node_alpha=0.5, edge_linewidth=0.5, edge_alpha=0.5, show=False, close=False)
# ne pas afficher en networkx sinon c'est les problemes
#plt.show()

#faire varier la probabilité de percolation
probabilities = np.arange(0, 1, 0.01)
res = []
for i in probabilities:
    lissage = []
    for _ in range(1):
        cpt = 0
        G = G_copy.copy()

        # Percolation
        for node in list(G.nodes()):
            if np.random.rand() < i:
                G.remove_node(node)
                cpt += 1

        # Identifier les clusters
        clusters = list(nx.connected_components(G.to_undirected()))

        # Trouver le plus grand cluster
        if len(clusters) == 0:
            largest_cluster = []
        else:
            largest_cluster = max(clusters, key=len)

        # Calculer la probabilité qu'un point se trouve dans le plus grand cluster
        probability = len(largest_cluster) / ((100 * 100)-cpt)
        lissage.append(probability)
    res.append(np.mean(lissage))
    
# plt.plot(probabilities, res)
# plt.xlabel('Probabilité de percolation')
# plt.ylabel('Probabilité que le point se trouve dans le plus grand cluster')
#plt.show()



def update_states(G, phi):
    states = nx.get_node_attributes(G, 'state').copy()
    for node, state in states.items():
        if state == 0:
            neighbors = list(G.neighbors(node))
            num_neighbors = len(neighbors)
            if num_neighbors > 0:
                num_failed_neighbors = sum([states[neighbor] for neighbor in neighbors])
                if num_failed_neighbors / num_neighbors >= phi:
                    G.nodes[node]['state'] = 1

n = 100
phi = np.arange(0, 1, 0.01)
res = []

for i in phi:
    experiment_results = []
    for _ in range(1):
        G = G_copy.copy()

        nx.set_node_attributes(G, 0, 'state')

        #on met un noeux a 1 (il est cassé)(random)
        original_node = random.choice(list(G.nodes))
        original_node_index = list(G.nodes).index(original_node)
        G.nodes[original_node]['state'] = 1

        for j in range (100):
            update_states(G, i)
            
        experiment_results.append(sum(nx.get_node_attributes(G, 'state').values())/n*100)
    
    res.append(np.mean(experiment_results))

        
        

#plot
node_colors = ["red" if G.nodes[node]['state'] == 1 else "grey" for node in G.nodes()]
node_colors[original_node_index] = "blue"
ox.plot_graph(G, node_color=node_colors, node_size=10, node_alpha=0.5, edge_linewidth=0.5, edge_alpha=0.5, show=False, close=False)
plt.figure()
plt.plot(phi, res)
plt.xlabel('phi')
plt.ylabel('% de nodes contaminées')
plt.show()