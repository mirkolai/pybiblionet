import csv
import json
from typing import List

import community.community_louvain as community_louvain
from networkx.algorithms.community.centrality import girvan_newman
import networkx as nx
from infomap import Infomap
from sklearn.cluster import SpectralClustering
import igraph as ig

def _customize_for_network_type(G):
    if G.is_directed():
        type_of_graph = "directed citation"
        available_metrics = {
            "betweenness_centrality": nx.betweenness_centrality,
            "closeness_centrality": nx.closeness_centrality,
            "eigenvector_centrality": nx.eigenvector_centrality,
            "page_rank": nx.pagerank,
            "in_degree": nx.in_degree_centrality,
            "out_degree": nx.out_degree_centrality,
        }
        available_fields = ["id", "doi", "title", "publication_year", "language", "type",
                            "authorships display_name", "primary_topic display_name",
                            "primary_topic subfield display_name",
                            "primary_topic field display_name",
                            "primary_topic domain display_name",
                            "cited_by_count","core","type","venue"]

        def extract_fields(data, available_fields):
            field_mapping = {
                "id": data.get("id", "") or "",
                "doi": data.get("doi", "") or "",
                "title": data.get("title", "") or "",
                "publication_year": data.get("publication_year", "") or "",
                "language": data.get("language", "") or "",
                "type": data.get("type", "") or "",
                "authorships display_name": ", ".join(
                    [authorship.get("author", {}).get("display_name", "") for authorship in data.get("authorships", [])
                     if isinstance(authorship, dict)]),

                "primary_topic display_name": (data.get("primary_topic") or {}).get("display_name", ""),
                "primary_topic subfield display_name": (data.get("primary_topic") or {}).get("subfield", {}).get("display_name", ""),
                "primary_topic field display_name": (data.get("primary_topic") or {}).get("field", {}).get("display_name", ""),
                "primary_topic domain display_name": (data.get("primary_topic") or {}).get("domain", {}).get("display_name", ""),

                "cited_by_count": data.get("cited_by_count", "") or "",
                "core": data.get("core", "") or "False",
                "venue": (data.get("primary_location") or {}).get("display_name", "") or "False",
            }

            extracted_data = {}
            for field in available_fields:
                extracted_data[field] = field_mapping.get(field, None)
            return extracted_data

    else:
        type_of_graph = "undirected co-authorship"
        available_metrics = {
            "betweenness_centrality": nx.betweenness_centrality,
            "closeness_centrality": nx.closeness_centrality,
            "eigenvector_centrality": nx.eigenvector_centrality,
            "page_rank": nx.pagerank,
            "degree": nx.degree_centrality,
        }
        available_fields = ["author id", "author orcid", "author display_name", "institutions display_name", "institutions type", "countries"]

        def extract_fields(data, available_fields):

            field_mapping = {
                "author id": (data.get("author", {})  or {}).get("id",""),
                "author orcid": (data.get("author", {}) or {}).get("orcid",""),
                "author display_name": (data.get("author", {}) or {}).get("display_name",""),
                "institutions display_name": " ".join([inst["display_name"] for inst in (data.get("institutions", []) or [])]),
                "institutions type": " ".join([inst["type"] for inst in (data.get("institutions", []) or [])]),
                "countries": " ".join((data.get("countries", []) or []))
            }

            extracted_data = {}
            for field in available_fields:
                extracted_data[field] = field_mapping.get(field, None)

            return extracted_data

    return type_of_graph,available_metrics,available_fields,extract_fields

def extract_metrics_to_csv(G: nx.DiGraph|nx.Graph, metrics: List[str], fields: List[str], csv_file_path: str):
    """
    Extracts the specified metrics and fields from the graph and saves them to a CSV file.

    :param G: A NetworkX directed or undirected graph.
    :param metrics: A list of metrics to compute (available: 'betweenness_centrality', 'closeness_centrality',
                    'page_rank', 'in_degree', 'out_degree', 'degree').
                    'in_degree', 'out_degree' are available only for directed graphs
                    'degree' is available for undirected graphs

    :param fields: A list of fields to include in the CSV
                   for articles the available fields are: "id", "doi", "title", "publication_year", "language", "type",
                            "authorships display_name", "primary_topic display_name",
                            "primary_topic subfield display_name",
                            "primary_topic field display_name",
                            "primary_topic domain display_name",
                            "cited_by_count","core";
                   for authors the available fields are:  "author id", "author orcid", "author display_name", "institutions display_name", "institutions type", "countries")
    :param csv_file_path: The path to the CSV file to save the results.
    """

    type_of_graph, available_metrics, available_fields, extract_fields = _customize_for_network_type(G)

    if fields is None:
        fields = available_fields
    if metrics is None:
        metrics = list(available_metrics.keys())
    if len(fields)==0:
        raise ValueError(f"Choose at least a field from: {', '.join(available_fields)}")
    if len(metrics)==0:
        raise ValueError(f"Choose at least a metric from: {', '.join(available_metrics)}")

    # Check for invalid metrics and fields
    invalid_metrics = [metric for metric in metrics if metric not in available_metrics]
    if invalid_metrics:
        raise ValueError(f"The following metrics are not available in a {type_of_graph} graph: {', '.join(invalid_metrics)}")

    invalid_fields = [field for field in fields if field not in available_fields]

    if invalid_fields:
        raise ValueError(f"The following fields are not available in a {type_of_graph} graph: {', '.join(invalid_fields)}")

    # Compute the metrics
    metric_results = {
        metric: available_metrics[metric](G) if callable(available_metrics[metric]) else dict(available_metrics[metric])
        for metric in metrics}

    # Write to CSV
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = fields + metrics
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for node in G.nodes(data=True):
            node_info=json.loads(node[1]['info'])
            row = extract_fields(node_info,fields)
            for metric in metrics:
                row[metric] = metric_results[metric].get(node[0], '')
            writer.writerow(row)

def extract_clusters_to_csv(G: nx.DiGraph|nx.Graph, fields: List[str], csv_file_path: str):
    """
    Extracts the cluster and selected fields from the graph and saves them to a CSV file.

    :param G: A NetworkX directed or undirected graph.

    :param fields: A list of fields to include in the CSV
                   for articles the available fields are: "id", "doi", "title", "publication_year", "language", "type",
                            "authorships display_name", "primary_topic display_name",
                            "primary_topic subfield display_name",
                            "primary_topic field display_name",
                            "primary_topic domain display_name",
                            "cited_by_count","core";
                   for authors the available fields are:  "author id", "author orcid", "author display_name", "institutions display_name", "institutions type", "countries")
    :param csv_file_path: The path to the CSV file to save the results.
    """



    type_of_graph, _, available_fields, extract_fields = _customize_for_network_type(G)
    if fields is None:
        fields = available_fields
    if len(fields)==0:
        raise ValueError(f"Choose at least a field from: {', '.join(available_fields)}")
    print(fields)
    invalid_fields = [field for field in fields if field not in available_fields]
    print(invalid_fields)

    if invalid_fields:
        raise ValueError(f"The following fields are not available in a {type_of_graph} graph: {', '.join(invalid_fields)}")

    # Write to CSV
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = fields + ["cluster"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for node in G.nodes(data=True):
            node_info = json.loads(node[1]['info'])
            print(node_info)

            row = extract_fields(node_info, fields)
            print(row)
            row["cluster"] = node[1]["cluster"]
            writer.writerow(row)

def cluster_graph(G: nx.DiGraph|nx.Graph, algorithm: str = 'louvain', num_clusters=5) -> nx.DiGraph|nx.Graph:
    """
    Clusters the nodes in the graph using the specified clustering algorithm and returns a new graph with cluster numbers as node attributes.

    :param G: A NetworkX directed or undirected graph.
    :param algorithm: The clustering algorithm to use ('louvain', 'girvan_newman','infomap', 'spectral_clustering', 'stochastic_block_model').
    :param num_cluster: The number of clusters to create when Spectral Clustering is used. Default is 5.
    :return: A NetworkX graph with cluster numbers as a new node attribute.


    """
    if algorithm == 'louvain':
        partition = community_louvain.best_partition(G.to_undirected(),random_state=87)
        for node, cluster in partition.items():
            G.nodes[node]['cluster'] = cluster
    elif algorithm == 'girvan_newman':
        comp = girvan_newman(G)
        clusters = tuple(sorted(c) for c in next(comp))
        for cluster_num, cluster_nodes in enumerate(clusters):
            for node in cluster_nodes:
                G.nodes[node]['cluster'] = cluster_num
    elif algorithm == 'infomap':

        mapping_realid2fakeid = {}
        mapping_fakeid2realid = {}

        for i, node in enumerate(G.nodes()):
            mapping_realid2fakeid[node] = i
            mapping_fakeid2realid[i] = node

        H = nx.relabel_nodes(G, mapping_realid2fakeid, copy=True)
        im = Infomap(silent=False)
        im.add_networkx_graph(H)
        im.run()

        nodes = im.get_nodes(depth_level=1)
        for node in nodes:
            G.nodes[mapping_fakeid2realid[node.node_id]]['cluster'] = node.module_id
    elif algorithm == 'spectral_clustering':
        adj_matrix = nx.to_numpy_array(G)
        spectral = SpectralClustering(n_clusters=min(num_clusters, len(G)), affinity='precomputed', random_state=87)
        labels = spectral.fit_predict(adj_matrix)

        for node, cluster in zip(G.nodes(), labels):
            G.nodes[node]['cluster'] = cluster

    elif algorithm == 'stochastic_block_model':
        if G.is_directed():
            mode="directed"
        else:
            mode="undirected"

        adjacency_matrix = nx.to_numpy_array(G)
        g = ig.Graph.Weighted_Adjacency(adjacency_matrix.tolist(), mode=mode)
        sbm = g.community_infomap()  # SBM with Infomap
        for cluster_num, cluster_nodes in enumerate(sbm):
            for node in cluster_nodes:
                G.nodes[list(G.nodes())[node]]['cluster'] = cluster_num
    else:
        raise ValueError("Unsupported algorithm. Choose from 'louvain' or 'girvan_newman'.")


    return G

