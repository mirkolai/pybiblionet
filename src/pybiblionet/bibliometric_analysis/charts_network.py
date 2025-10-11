import json
from typing import Optional, List, Tuple, Dict, Union
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch
import textwrap

def _get_centeroid(arr: List[Tuple[float, float]], position_scale_factor: int =1) -> Tuple[float, float]:
    """
    Calculate the centroid of a list of points.

    :param arr: List of tuples, where each tuple contains (x, y) coordinates of a point.
    :param position_scale_factor: Scaling factor for the position coordinates. Defaults to 1.

    :return: A tuple representing the centroid (x, y).
    """
    length = len(arr)

    sum_x = np.sum([z[0] for z in arr]) * position_scale_factor
    sum_y = np.sum([z[1] for z in arr]) * position_scale_factor
    return sum_x / length, sum_y / length

def _get_limits(arr: List[Tuple[float, float]], limit_border:int = 1) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Calculate the limits (min and max) for x and y coordinates.

    :param arr: List of tuples, where each tuple contains (x, y) coordinates of a point.
    :param limit_border: Scaling factor for the limits. Defaults to 1.
    :return: A tuple of tuples representing the x and y limits ((x_min, x_max), (y_min, y_max)).
    """
    x_max = np.max([z[0] for z in arr]) * limit_border
    x_min = np.min([z[0] for z in arr]) * limit_border
    y_max = np.max([z[1] for z in arr]) * limit_border
    y_min = np.min([z[1] for z in arr]) * limit_border
    return (x_min, x_max), (y_min, y_max)




def _get_entities_in_clusters_citation_graph(G: nx.DiGraph, topics_level: str = "domain") -> Dict[int, Dict[str, int]]:
    """
    Extracts entities from graph nodes and groups them by their clusters, counting occurrences of each entity.

    Args:
        G (Any): A graph object with nodes containing 'cluster' and 'info' attributes.
                 Each node's 'info' attribute should be a JSON string containing either 'concepts' or 'topics'.
        topics_level (Union[int, None]): The level of topics to filter by ("field" or "domain").

    Returns:
        Dict[int, Dict[str, int]]: A dictionary where each key is a cluster identifier (int) and each value is a
                                   dictionary of entity names (str) with their respective counts (int) within that cluster.
                                   Example:
                                   {
                                       0: {'Social Sciences': 50, 'Medicine': 24, ...},
                                       1: {'Computer Science': 17, 'Engineering': 82, ...},
                                       ...
                                   }
    """
    entries_in_clusters = {}
    for node_id, node_data in G.nodes(data=True):

        if node_data["cluster"] not in entries_in_clusters:
            entries_in_clusters[node_data["cluster"]] = {}


        if topics_level is not None:
            node_info=json.loads(node_data["info"])
            if node_info["primary_topic"] is None:
                continue
            primary_topic=node_info["primary_topic"].get(topics_level,{}).get('display_name',"None")
            if primary_topic not in entries_in_clusters[node_data["cluster"]]:
                entries_in_clusters[node_data["cluster"]][primary_topic] = 0
            entries_in_clusters[node_data["cluster"]][primary_topic] += 1

    return entries_in_clusters


def _get_entities_in_clusters_coauthorship_graph(G: nx.Graph) -> Dict[int, Dict[str, int]]:
    """
    Extracts entities from graph nodes and groups them by their clusters, counting occurrences of each entity.

    Args:
        G (Any): A graph object with nodes containing 'cluster' and 'info' attributes.

    Returns:
        Dict[int, Dict[str, int]]: A dictionary where each key is a cluster identifier (int) and each value is a
                                   dictionary of entity names (str) with their respective counts (int) within that cluster.
                                   Example:
                                   {
                                       0: {'IT': 50, 'FR': 24, ...},
                                       1: {'ES': 17, 'US': 82, ...},
                                       ...
                                   }
    """
    entries_in_clusters = {}
    for node_id, node_data in G.nodes(data=True):

        if node_data["cluster"] not in entries_in_clusters:
            entries_in_clusters[node_data["cluster"]] = {}


        node_info=json.loads(node_data["info"])
        if node_info["countries"] is None or len(node_info["countries"])==0:
            continue

        for country in node_info["countries"]:
            if country not in entries_in_clusters[node_data["cluster"]]:
                entries_in_clusters[node_data["cluster"]][country] = 0
            entries_in_clusters[node_data["cluster"]][country] += 1

    return entries_in_clusters



def _get_top_m_entries(clusters: Dict[int, Dict[str, int]], n: int,top_n_clusters) -> List[str]:
    """
    Aggregates counts of different entries across multiple clusters and returns the top m entries.

    Args:
        clusters (dict): A dictionary where each key is a cluster identifier and each value is a dictionary
                         of entries with their respective counts within that cluster.
                         Example:
                         {
                             0: {'Social Sciences': 507, 'Medicine': 24, ...},
                             1: {'Computer Science': 17, 'Engineering': 82, ...},
                             ...
                         }
        n (int): The number of top entries to return based on their aggregated counts across all clusters.

    Returns:
        list: A list of the top n entries sorted by their aggregated counts in descending order.
              Example: ['Social Sciences', 'Engineering', 'Environmental Science', ...]
    """
    total_counts = {}

    for cluster_id, entries_in_cluster in clusters.items():
        if cluster_id not in top_n_clusters:
            continue
        for entry, count in entries_in_cluster.items():
            if entry not in total_counts:
                total_counts[entry] = 0
            total_counts[entry] += count
    sorted_entries = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)

    return [x[0] for x in sorted_entries[:n]]

def _get_top_n_clusters(G: nx.DiGraph|nx.Graph, n_clusters: int) -> Dict[int, int]:
    """
    Identifies the top N clusters based on the number of nodes in each cluster.

    Args:
        G (Any): A graph-like object containing nodes with cluster information.
        n_clusters (int): The number of top clusters to return.

    Returns:
        Dict[int, int]: A dictionary where keys are cluster identifiers and values are the counts of nodes in each cluster,
                        sorted by the count in descending order.
    """
    top_n_clusters = {}
    for node_id, node_data in G.nodes(data=True):
        if node_data['cluster'] in top_n_clusters:
            top_n_clusters[node_data['cluster']] += 1
        else:
            top_n_clusters[node_data['cluster']] = 1


    top_n_clusters = sorted(top_n_clusters.items(), key=lambda x: x[1], reverse=True)[:n_clusters]

    return dict(top_n_clusters)


def _get_cluster_position(node_sizes, edges_weights, max_node_radius):
    G = nx.DiGraph()

    for node, size in node_sizes.items():
        G.add_node(node, size=size)

    for edge, weight in edges_weights.items():
        source, target = map(int, edge.split())
        G.add_edge(source, target, weight=weight)

    pos = nx.spring_layout(G, k=max_node_radius, iterations=50, weight='weight')
    pos = _prevent_overlap(pos, max_node_radius)

    return _center_graph_positions(pos)


def _prevent_overlap(positions: Dict[int, np.ndarray], max_radius: float) -> Dict[
    int, np.ndarray]:
    """
    Adjusts node positions to prevent overlap by ensuring no nodes are within `2 * max_radius` distance.
    Expands the entire graph outward if any overlap is detected.

    Parameters:
    - positions: A dictionary of node positions {node_id: position_array}.
    - max_radius: The maximum radius of a node, used to set the minimum distance between nodes.

    Returns:
    - adjusted_positions: A dictionary with adjusted positions to avoid overlap.
    """
    adjusted_positions = positions.copy()
    overlap_detected = True

    # Check each pair of nodes for overlap
    while overlap_detected:
        overlap_detected=False
        positions = adjusted_positions.copy()
        for node_a, pos_a in positions.items():
            for node_b, pos_b in positions.items():
                if node_a < node_b:  # Avoid redundant checks
                    dist = np.linalg.norm(pos_a - pos_b)
                    # If nodes are too close, push them apart
                    if dist <= 2.5 * max_radius:
                        overlap_detected = True
                        # If overlap was detected, expand the entire graph outward along each axis
                        if overlap_detected:
                            for node in adjusted_positions:
                                adjusted_positions[node] *= 1.1
                            break

    return adjusted_positions

def _center_graph_positions(cluster_positions: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """
        Centers the positions of clusters in a graph layout.

        This function calculates the centroid (average position) of all clusters
        and then shifts each cluster's position so that the overall layout is
        centered around the origin.

        Args:
            cluster_positions (Dict[int, np.ndarray]): A dictionary mapping cluster IDs
                to their position vectors (e.g., 2D or 3D coordinates).

        Returns:
            Dict[int, np.ndarray]: A new dictionary with the same keys but with
            position vectors translated to center the entire layout.
    """
    centroid = np.mean(np.array(list(cluster_positions.values())), axis=0)

    centered_positions = {cluster_id: pos - centroid for cluster_id, pos in cluster_positions.items()}

    return centered_positions


def _normalize_degree(data, min_target=1, max_target=10):
    """
    Normalizes degree values to a target range using min-max scaling.

    This function rescales all values in the input dictionary so that they lie
    between `min_target` and `max_target`. If all values are equal, all outputs
    will be set to `max_target`.

    Args:
        data (dict): A dictionary of numerical values to normalize.
        min_target (float, optional): The minimum value in the target range. Defaults to 1.
        max_target (float, optional): The maximum value in the target range. Defaults to 10.

    Returns:
        dict: A new dictionary with the same keys and normalized values.
    """
    min_val = min(data.values())
    max_val = max(data.values())

    if max_val == min_val:
        return {k: max_target for k in data}

    return {k: min_target + (max_target - min_target) * (v - min_val) / (max_val - min_val)
            for k, v in data.items()}

def show_clustered_graph(G,
                         n_clusters: int = 5,
                         m_entries: int = 5, verbose = False,
                         n_cluster_colors: Optional[List[str]] = None,
                         m_entry_colors: Optional[List[str]] = None,
                         min_node_radius: float = 3,
                         max_node_radius: float = 5,
                         min_pie_radius: float = 2,
                         max_pie_radius : float = 3,
                         size_legend_marker=2,
                         size_legend_font=15,
                         size_node_font=13,
                         min_edge_width :int = 1,
                         max_edge_width :int = 200,
                         edge_color: str = '#add8e6',
                         hatch: Optional[str] = '',
                         top_m_entries: Optional[List[str]] = None,
                         bbox_to_anchor_legend_entries: Tuple[float, float, float, float] = (1, 1),
                         bbox_to_anchor_legend_clusters: Tuple[float, float, float, float] = (0, 1),
                         image_size : Tuple[int,int] = (947, 1061),
                         topics_level: Optional[str] = "domain",
                         export_path_png: Optional[str] = None) -> None:
    """
        Visualizes a clustered bibliometric graph with pie charts and legends.

        This function generates a high-level visualization of clusters within a graph `G`,
        where each cluster is represented by a circle, and the composition of entities
        (e.g., fields or domains) is shown using pie charts.

        Parameters:
            G (nx.Graph or nx.DiGraph): The graph representing bibliometric data.
            n_clusters (int): Number of top clusters to display.
            m_entries (int): Number of top entities (e.g., disciplines) to include in pies.
            verbose (bool): If True, prints debug info.
            n_cluster_colors (Optional[List[str]]): Colors for each cluster.
            m_entry_colors (Optional[List[str]]): Colors for each entity in pies.
            min_node_radius (float): Minimum cluster circle size.
            max_node_radius (float): Maximum cluster circle size.
            min_pie_radius (float): Minimum pie chart radius around clusters.
            max_pie_radius (float): Maximum pie chart radius.
            size_legend_marker (int): Marker size in legend for clusters.
            size_legend_font (int): Font size in legend for entries.
            size_node_font (int): Font size for cluster labels.
            min_edge_width (int): Minimum width for inter-cluster edges.
            max_edge_width (int): Maximum width for inter-cluster edges.
            edge_color (str): Color used for edges between clusters.
            hatch (Optional[str]): Pattern to fill the doughnut segments. Default is '', e.g., '/', 'x', or 'o'.
            top_m_entries (Optional[List[str]]): Specific top entries to include in pies.
            bbox_to_anchor_legend_entries (Tuple[float]): Positioning of the entries legend.
            bbox_to_anchor_legend_clusters (Tuple[float]): Positioning of the cluster legend.
            image_size (Tuple[int, int]): Output image size in pixels.
            topics_level (Optional[str]): Taxonomy level ("field" or "domain") for entity clustering.
            export_path_png (Optional[str]): If specified, saves the figure to this path.

        Raises:
            ValueError: If inputs are inconsistent or invalid (e.g., length mismatch of color lists).

        Returns:
            None: Displays or saves the graph visualization.

    """
    if topics_level is not None:
        if topics_level not in ["field","domain"]:
            raise ValueError("topics_level must be either 'field' or 'domain'.")

    if top_m_entries is not None:
        if len(top_m_entries) != m_entries:
            raise ValueError("top entries must be the same size of m_entries.")
    if n_cluster_colors is not None:
        if len(n_cluster_colors) != n_clusters:
            raise ValueError("top n_cluster_colors must be the same size of n_clusters.")
    if m_entry_colors is not None:
        if len(m_entry_colors) != m_entries+1:
            raise ValueError("top m_entry_colors must be the same size of m_entries (one color for other).")

    if top_m_entries != None:
        if len(top_m_entries) != m_entries:
            raise ValueError("top entries must be m_entries.")

    dpi = plt.rcParams['figure.dpi']
    fig_size = (image_size[0]/dpi, image_size[1]/dpi)

    fig = plt.figure(figsize=fig_size)
    # Definisci la griglia di sottografici
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 5])

    # Sottografici per le leggende
    ax_legend_left = fig.add_subplot(gs[0, 0])
    ax_legend_right = fig.add_subplot(gs[0, 1])

    # Sottografico per il grafo
    ax_graph = fig.add_subplot(gs[1, :])

    # Disattivare gli assi per le leggende
    ax_legend_left.axis('off')
    ax_legend_right.axis('off')


    if n_cluster_colors is None:

        n_cluster_colors = n_clusters*["#FFFFFF"]

    if m_entry_colors is None:
        cmap = plt.get_cmap('tab10')
        m_entry_colors = [cmap(i) for i in range(n_clusters+1)] #+1 for Other


    top_n_clusters=_get_top_n_clusters(G, n_clusters)
    top_n_clusters=_normalize_degree(top_n_clusters)


    if G.is_directed():
        entries_in_clusters = _get_entities_in_clusters_citation_graph(G, topics_level)
    else:
        entries_in_clusters = _get_entities_in_clusters_coauthorship_graph(G)
    if verbose:
        print("entries_in_clusters",entries_in_clusters)

    if top_m_entries is None:
        top_m_entries = _get_top_m_entries(entries_in_clusters, m_entries,top_n_clusters)
    map_color_to_entry={}
    for i, entry in enumerate(top_m_entries+["Other"]):
        map_color_to_entry[entry]=m_entry_colors[i]

    cluster_edge_map={}
    for edge in G.edges():
        if G.nodes[edge[0]]['cluster'] not in top_n_clusters.keys() or G.nodes[edge[1]]['cluster'] not in top_n_clusters.keys():
            continue
        if str(G.nodes[edge[0]]['cluster'])+" "+str(G.nodes[edge[1]]['cluster']) not in cluster_edge_map:
            cluster_edge_map[str(G.nodes[edge[0]]['cluster'])+" "+str(G.nodes[edge[1]]['cluster'])]=0
        cluster_edge_map[str(G.nodes[edge[0]]['cluster'])+" "+str(G.nodes[edge[1]]['cluster'])]+=1


    cluster_position_map = _get_cluster_position(top_n_clusters, cluster_edge_map, max_node_radius + max_pie_radius)

    max_radius=0
    ##########################################################################################
    #draw circles

    for i, cluster_id in enumerate(top_n_clusters.keys()):
        radius = min_node_radius +(max_node_radius-min_node_radius) * (top_n_clusters[cluster_id]/max(top_n_clusters.values()))
        x, y = cluster_position_map[cluster_id]
        circle1 = plt.Circle((x, y),
                             radius,
                             ec='white',
                             linewidth=2,
                             color=n_cluster_colors[i],
                             zorder=4,
                             )
        ax_graph.add_patch(circle1)
        ax_graph.annotate(str(int(cluster_id)),
                    fontsize=size_node_font,
                    xy=(x, y),
                    ha='center',
                    va='center',
                    zorder=5,
                    color='black')
        if max_radius<radius:
            max_radius=radius



    ##########################################################################################
    #draw edges
    def draw_curved_edge(ax, x0, y0, x1, y1, weight, max_weight, min_edge_width=0.5, max_edge_width=5, color='gray',
                         clockwise=True):
        # Calcola un offset per la curvatura
        dx = x1 - x0
        dy = y1 - y0
        dist = np.sqrt(dx ** 2 + dy ** 2)

        # Curvature: positiva = anti-orario, negativa = orario
        curvature = 0.2  # puoi modulare in base alla distanza o al peso
        if clockwise:
            curvature = -curvature

        # Usa FancyArrowPatch per arco curvo
        arrow = FancyArrowPatch((x0, y0), (x1, y1),
                                connectionstyle=f"arc3,rad={curvature}",
                                linewidth=min_edge_width + ((max_edge_width - min_edge_width) * (weight / max_weight)),
                                color=color,
                                alpha=0.3,
                                zorder=-100)
        ax.add_patch(arrow)

    max_weight = max(cluster_edge_map.values())
    for community_pair, weight in cluster_edge_map.items():
        source, target = map(int, community_pair.split(" "))
        if source not in cluster_position_map or target not in cluster_position_map:
            continue

        x0, y0 = cluster_position_map[source]
        x1, y1 = cluster_position_map[target]

        # Determina la direzione
        # ad esempio: entrante (target) = orario, uscente (source) = antiorario
        clockwise = True  # True = curvatura in senso orario, False = anti-orario
        # Puoi anche decidere basandoti su qualche proprietà del nodo

        draw_curved_edge(ax_graph, x0, y0, x1, y1, weight, max_weight, min_edge_width=min_edge_width, max_edge_width=max_edge_width,
                         color=edge_color, clockwise=clockwise)





    ###################################################
    ##draw PIES

    handles = {}
    size_legend_markers=[]
    for i, cluster_id in enumerate(cluster_position_map.keys()):
        radius = min_node_radius +(max_node_radius-min_node_radius) * (top_n_clusters[cluster_id]/max(top_n_clusters.values()))

        radius=radius+min_pie_radius + max_pie_radius *( top_n_clusters[cluster_id]/max(top_n_clusters.values()))
        size_legend_markers.append(radius)
        filtered_enties={ "Other" : 0.1}
        for discipline,occurencies in entries_in_clusters[cluster_id].items():
            if discipline in top_m_entries:
                filtered_enties[discipline]=occurencies
            else:
                filtered_enties["Other"]+=occurencies

        x, y = cluster_position_map[cluster_id]

        patches = ax_graph.pie([occurency for occurency in filtered_enties.values()],
                         explode=[0] * len(filtered_enties),
                         startangle=90,
                         center=(x, y),
                         colors=[ map_color_to_entry[entry] for entry in filtered_enties.keys()],
                         counterclock=False,
                         radius=radius)

        if max_radius<radius:
            max_radius=radius
        for i in range(len(patches[0])):
            if list(filtered_enties.values())[i]==0:
                continue
            patches[0][i].set(fill=True,
                              hatch=hatch,
                              edgecolor='#C0C0C0')



    #############################
    ### draw legends


    for entry in top_m_entries+["Other"]:

        handles[entry] = \
            mpatches.Patch(facecolor=map_color_to_entry[entry],
                           label=entry,
                           fill=True,
                           hatch=hatch,
                           edgecolor='#C0C0C0')

    legend_entries = ax_legend_left.legend(handles=[value for x, value in handles.items()],
                                loc='center right',
                                bbox_to_anchor=bbox_to_anchor_legend_entries,
                                fontsize=size_legend_font,
                                frameon=False)



    legend_elements=[]
    for i, cluster_id in enumerate(cluster_position_map.keys()):
        legend_elements.append(Line2D([], [],
                                      markeredgecolor='black',
                                      markerfacecolor=n_cluster_colors[i],
                                      marker='o',
                                      linestyle='None',
                                      markersize=size_legend_markers[i]*size_legend_marker,
                                      label="CLUSTER "+str(cluster_id)))
    legend_clusters=ax_legend_right.legend(handles=legend_elements,
                               loc='center left',
                               bbox_to_anchor=bbox_to_anchor_legend_clusters,
                               fontsize = 16,
                               frameon=False)






    ##############################################
    (x_min, x_max), (y_min, y_max) =_get_limits(cluster_position_map.values())
    plt.axis((x_min,x_max,y_min,y_max))
    plt.xlim((x_min - max_radius*2, x_max +  max_radius*2))
    plt.ylim((y_min - max_radius*2, y_max +  max_radius*2))
    if export_path_png is not None:
        plt.savefig(export_path_png)

    plt.tight_layout()
    plt.show()
    plt.close()


import pandas as pd
import matplotlib.pyplot as plt




def show_cluster_statistics(
        csv_file_path,
        n_cluster_colors,
        n_clusters=5):
    """
    Generate cluster-level statistics and a LaTeX table summarizing the most frequent categories
    (disciplines or countries) for each cluster.

    Parameters
    ----------
    csv_file_path : str
        Path to the CSV file containing the clustering data.
    n_cluster_colors : list of str, optional
        List of HEX color codes to represent clusters. Default is a predefined color palette.
    n_clusters : int, optional
        Number of clusters to display (and number of colors expected).

    Returns
    -------
    stats_df : pandas.DataFrame
        DataFrame containing the computed cluster statistics.
    latex_output : str
        String with the LaTeX table code representing cluster composition.

    Notes
    -----
    The CSV file must contain a 'cluster' column and either:
      - 'primary_topic field display_name' (for citation graph analysis), or
      - 'countries' (for co-authorship graph analysis).
    """

    # --- Check colors ---
    if n_cluster_colors is None:
        n_cluster_colors = n_clusters * ["#FFFFFF"]

    if len(n_cluster_colors) != n_clusters:
        raise ValueError(
            f"The number of colors ({len(n_cluster_colors)}) must match n_clusters ({n_clusters})."
        )

    # --- Read CSV file ---
    df = pd.read_csv(csv_file_path)

    if 'cluster' not in df.columns:
        raise ValueError("The CSV file must contain a 'cluster' column.")
    if ('primary_topic field display_name' not in df.columns) and ('countries' not in df.columns):
        raise ValueError(
            "The CSV file must contain 'primary_topic field display_name' (for citation graphs) "
            "or 'countries' (for co-authorship graphs)."
        )

    # --- Compute statistics per cluster ---
    cluster_stats = []
    for cluster_id, group in df.groupby('cluster'):
        num_entities = len(group)
        if 'primary_topic field display_name' in df.columns:
            type_class = "disciplines"
            entities = "papers"
        else:
            type_class = "countries"
            entities = "authors"

        # Rename to a common 'class' column for easier processing
        group = group.rename(columns={'primary_topic field display_name': 'class', 'countries': 'class'})
        most_common = group['class'].value_counts(normalize=True).head(3)
        entries = list(most_common.index)
        percentages = list((most_common * 100).round(0).astype(int))

        row = {
            'Id': int(cluster_id),
            f'Num of {entities}': num_entities,
        }

        # Fill top 3 most common categories
        for i in range(3):
            if i + 1 == 1:
                post = "st"
            elif i + 1 == 2:
                post = "nd"
            else:
                post = "rd"

            if i < len(entries):
                row[f'{i + 1}{post}'] = f"{entries[i]} ({percentages[i]}\\%)"  # \% for LaTeX
            else:
                row[f'{i + 1}{post}'] = ""

        cluster_stats.append(row)

    stats_df = pd.DataFrame(cluster_stats)

    # Sort clusters by size (descending), then by ID (ascending)
    stats_filtered_df = (
        stats_df
        .sort_values(by=[f'Num of {entities}', 'Id'], ascending=[False, True])
        .head(n_clusters)
        .reset_index(drop=True)
    )

    # --- Generate LaTeX table code ---
    latex_lines = []
    latex_lines.append("\\begin{table}[h!]")
    latex_lines.append("\\centering")
    latex_lines.append("\\tiny")
    latex_lines.append("\\begin{tabular}{lllll}")
    latex_lines.append("\\toprule")
    latex_lines.append(
        f"\\multicolumn{{2}}{{l}}{{\\textbf{{Cluster}}}} & "
        f"\\multicolumn{{3}}{{l}}{{\\textbf{{Most frequent {type_class}}}}}\\\\"
    )
    latex_lines.append("\\cmidrule(lr){1-2} \\cmidrule(lr){3-5}")
    latex_lines.append("Id & Num of papers & 1st & 2nd & 3rd \\\\")
    latex_lines.append("\\midrule")

    # Add table rows with color-coded cluster markers
    for i, row in stats_filtered_df.iterrows():
        color = n_cluster_colors[i]
        latex_color_name = f"color{i}"

        # Convert HEX color to LaTeX RGB format
        r = int(color[1:3], 16) / 255
        g = int(color[3:5], 16) / 255
        b = int(color[5:7], 16) / 255

        latex_lines.append(f"\\definecolor{{{latex_color_name}}}{{rgb}}{{{r:.2f},{g:.2f},{b:.2f}}}")
        latex_lines.append(
            f"\\textcolor{{{latex_color_name}}}{{  \\scalebox{{1.5}}{{$\\bullet$}}  }}~{row['Id']} & "
            f"{row[f'Num of {entities}']} & {row['1st']} & {row['2nd']} & {row['3rd']} \\\\"
        )

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append(f"\\caption{{Cluster composition and most frequent {type_class}.}}")
    latex_lines.append(f"\\label{{tab:cluster_{entities}}}")
    latex_lines.append("\\end{table}")

    latex_output = "\n".join(latex_lines)

    return stats_df, latex_output



def show_graph_statistics(G,csv_file_path,header_color="#a5b8d7"):
    """
    Displays structural and centrality metric statistics for a given graph using data from a CSV file.

    This function generates the following visualizations:
        - A summary table showing the number of nodes, edges, and graph density.
        - A table with the mean and variance for each available centrality metric.
        - Histograms showing the distribution of each centrality metric.
        - A correlation heatmap between centrality measures.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        The graph object whose statistics are to be analyzed.

    csv_file_path : str
        Path to the CSV file containing precomputed metrics for each node.
        The file may include one or more of the following columns:
        'betweenness_centrality', 'closeness_centrality', 'page_rank',
        'in_degree', 'out_degree', 'degree'.

    header_color : str, optional
        Hex color code used for the headers in the generated tables (default: '#a5b8d7').

    Returns
    -------
    None
        This function does not return any value. It displays plots and tables as output.

    """
    df = pd.read_csv(csv_file_path)

    all_available_metrics = ['betweenness_centrality', 'closeness_centrality', 'page_rank', 'in_degree', 'out_degree', 'degree']
    df = df[[col for col in all_available_metrics if col in df.columns]]
    def to_camel_case(text: str) -> str:
        text = text.replace("_", " ")
        words = text.split()
        return ' '.join(word.capitalize() for word in words)

    df.rename(columns=lambda col: to_camel_case(col), inplace=True)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = f"{nx.density(G):.2e}"
    statistics = {}

    for colum in df.columns:
        statistics[colum] = {'mean': df[colum].mean(),
                            'variance': df[colum].var()}
    statistics_table = {
        'Metric':  [f'{col}' for col in statistics],
        'Mean': [f"{stats['mean']:.2e}" for stats in statistics.values()],
        'Variance': [f"{stats['variance']:.2e}" for stats in statistics.values()]
    }


    fig, ax = plt.subplots(figsize=(8, 4))

    ax.axis('tight')
    ax.axis('off')
    y_min=0.9
    vertical_width=0.1
    header = plt.table(cellText=[['Statistic', 'Value']],
                       fontsize=20,
                       cellColours=[[header_color, header_color]],
                       cellLoc='center',
                       bbox=[0, y_min, 1, vertical_width],
                       )
    for cell in header.get_celld().values():
        cell.get_text().set_fontweight('bold')

    for metric,value in zip(['Number of Nodes', 'Number of Edges', 'Density'],[num_nodes,num_edges,density]):
        y_min=y_min-vertical_width
        plt.table(cellText=[[metric, f"{value}"]],
                  cellLoc='center',
                  bbox=[0, y_min, 1, vertical_width]
                  )
    y_min=y_min-vertical_width
    header = plt.table(cellText=[['Statistic', 'Mean', 'Variance']],
                       cellColours=[[header_color, header_color, header_color]],

                       cellLoc='center',
                       bbox=[0,y_min, 1, vertical_width]
                       )
    for cell in header.get_celld().values():
        cell.get_text().set_fontweight('bold')

    for i in range(0,len(statistics)):
        y_min=y_min-vertical_width
        plt.table(cellText=[[statistics_table["Metric"][i],statistics_table["Mean"][i],statistics_table["Variance"][i]]],
                  cellLoc='center',
                  bbox=[0, y_min, 1, vertical_width]
                  )


    plt.show()
    plt.close()

    correlation_matrix = df.corr()

    sns.set_theme(style="white")



    for colum in df.columns:

        plt.figure(figsize=(10, 6))
        ax = sns.histplot(df[colum], bins=30, kde=True, color='b', log_scale=True)
        plt.title(f'Distribution of {colum}', fontsize=20, fontweight='bold')
        plt.xlabel(colum)
        plt.ylabel('Frequency')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.show()
        plt.close()

    plt.figure(figsize=(8, 6))
    mask = np.tril(np.ones_like(correlation_matrix, dtype=bool))
    np.fill_diagonal(mask, True)

    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, annot_kws={"color": "black"})
    plt.title('Correlation Heatmap of Centrality Measures', fontsize=20, fontweight='bold',pad=30)
    for i, label in enumerate(df.columns):
        wrapped_label = "\n".join(textwrap.wrap(label, width=15))
        if i>0:
            rotation=0
        else:
            rotation=0
        plt.text(0.9 + i, i + 0.5, wrapped_label, ha='right', va='center',  fontsize=13, color="black",rotation=rotation)

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.subplots_adjust(left=0.2)
    plt.show()