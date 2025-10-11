import networkx as nx
from pybiblionet.bibliometric_analysis.core import extract_metrics_to_csv, cluster_graph, extract_clusters_to_csv
from pybiblionet.bibliometric_analysis.charts_network import show_clustered_graph, show_cluster_statistics, \
    show_graph_statistics

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

    for network_type in ["citation","coauthorship",]:

        if network_type=="citation":
            metrics = ['betweenness_centrality', 'closeness_centrality',
                    'page_rank', 'in_degree', 'out_degree']
            fields  = ["id", "doi", "title", "publication_year", "language", "type",
            "authorships display_name", "primary_topic display_name",
            "primary_topic subfield display_name",
            "primary_topic field display_name",
            "primary_topic domain display_name",
            "cited_by_count", "root_set"]
        else:
            metrics = ['betweenness_centrality', 'closeness_centrality',
                    'page_rank', 'degree']
            fields  = None



        network_file_name= f"15minute_{network_type}_graph.GML"

        G = nx.read_gml(network_file_name)
        print("Graph loaded.")
        print(G.number_of_nodes(), G.number_of_edges())


        csv_file_path = f"metrics_and_fields_{network_type}.csv"

        print("Extracting metrics...")
        extract_metrics_to_csv(G, metrics, fields, csv_file_path)
        show_graph_statistics(G,csv_file_path)

        print("Metrics extracted and saved to CSV.")

        print("\nClustering the graph...")

        clustered_graph = cluster_graph(G, algorithm='louvain',)


        # you can save the clustered graph so it can be imported into visualization or network analysis tools (e.g., Gephi)
        # nx.write_gml(clustered_graph, "clustered_"+network_file_name)
        # clustered_graph=nx.read_gml("clustered_"+network_file_name)

        csv_file_path = "cluster_and_fields_citation.csv"
        extract_clusters_to_csv(clustered_graph, fields, csv_file_path)

        print("Clusters extracted and saved to CSV.")
        # Visualize clusters
        print("Visualizing clusters...")
        n_cluster_colors=["#d7191c","#fdae61","#ffffbf","#abd9e9","#2c7bb6"]
        m_entry_colors = ["#8c510a","#d8b365","#f6e8c3","#c7eae5","#5ab4ac","#01665e"]
        stats_df, latex_output=show_cluster_statistics(csv_file_path, n_cluster_colors=n_cluster_colors, n_clusters=5)



        print(latex_output)
        show_clustered_graph(clustered_graph, image_size=(800, 800),
                             n_clusters=5,
                             topics_level="field",
                             m_entry_colors=m_entry_colors,
                             n_cluster_colors=n_cluster_colors,

                             )

