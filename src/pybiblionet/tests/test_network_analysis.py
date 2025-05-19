import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pybiblionet.bibliometric_analysis.core import extract_metrics_to_csv, cluster_graph, extract_clusters_to_csv
from pybiblionet.bibliometric_analysis.charts_network import show_clustered_graph, show_cluster_statistics,show_graph_statistics
from matplotlib import colors

# This script performs several operations on a citation graph of academic articles (recovered using test_openalex.py):
# 1. Loads a graph from a GML file using `networkx`.
# 2. Extracts a set of centrality and degree metrics from the graph (betweenness centrality, closeness centrality,
#    eigenvector centrality, page rank, in-degree, and out-degree), and saves them along with additional article fields
#    into a CSV file.
# 3. Applies the Louvain clustering algorithm to the graph to detect communities (clusters) of articles.
# 4. Saves the extracted cluster information (e.g., node IDs) to another CSV file.
# 5. Visualizes the clustered graph using `show_clustered_graph`, showing different clusters of articles.
# 6. Generates and displays a bar chart with statistics about the clusters, such as the number of nodes per cluster.
# This analysis allows for exploring the structure of the network of academic citations and understanding the distribution
# of various metrics and clusters within the data.

if __name__ == "__main__":
    # Read the graph

    json_file_path="query_results/query_result_1ea020320b230de5a973a39682eaa53dce89a9bb026b441a5f825232.json"
    network_file_name= "15minute_citation_graph.GML"

    G = nx.read_gml(network_file_name)
    print("Graph loaded.")
    print(G.number_of_nodes(), G.number_of_edges())

    # Extract and save metrics to CSV
    metrics = ["betweenness_centrality",
            "closeness_centrality",
            #"eigenvector_centrality",
            "page_rank",
            "in_degree",
            "out_degree"]
    fields = ["id", "title","cited_by_count","core","primary_topic display_name",
                            "primary_topic subfield display_name",
                            "primary_topic field display_name",
                            "primary_topic domain display_name"]
    csv_file_path = "metrics_and_fields_citation.csv"

    print("Extracting metrics...")
    extract_metrics_to_csv(G, metrics, fields, csv_file_path)
    #exit()
    #show_graph_statistics(G,csv_file_path)

    print("Metrics extracted and saved to CSV.")

    print("\nClustering the graph...")
    fields = ["id"]
    clustered_graph = cluster_graph(G, algorithm='louvain',)
    csv_file_path = "cluster_and_fields_citation.csv"
    extract_clusters_to_csv(clustered_graph, fields, csv_file_path)
    #exit()
    print("Clusters extracted and saved to CSV.")
    # Visualize clusters
    print("Visualizing clusters...")
    show_clustered_graph(clustered_graph, image_size=(800, 800),
                         n_clusters=5,
                         topics_level="field",
                         #min_node_radius=3,
                         #max_node_radius=5,
                         #max_pie_radius=5,
                         #min_pie_radius=3,
                         #size_node_font=12
                         )

    show_cluster_statistics(csv_file_path, image_size=(800, 800),
                 n_clusters=5)