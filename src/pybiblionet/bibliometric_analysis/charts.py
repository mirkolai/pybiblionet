import json
import os.path
import pickle

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

import numpy as np
from datetime import datetime
from typing import Dict, Optional
from collections import Counter, defaultdict
from keybert import KeyBERT
from tqdm import tqdm
from matplotlib.lines import Line2D
from matplotlib.colors import to_hex
from adjustText import adjust_text
import platform
if platform.system() == 'Linux':
    plt.switch_backend('TkAgg') #on linux run sudo apt install python3-tk
def _parse_publication_date(date_str: str) -> Optional[datetime]:
    """
    Parses the publication date string and returns a corresponding datetime object.

    Parameters:
    date_str (str): A string representing the publication date in the format "YYYY-MM-DD".

    Returns:
    Optional[datetime]: A datetime object corresponding to the parsed date if successful,
                         or None if the date string is not in the expected format.

    Example:
    >>> _parse_publication_date("2025-01-20")
    datetime.datetime(2025, 1, 20, 0, 0)

    >>> _parse_publication_date("invalid-date")
    None
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except:
        return None


def _aggregate_articles_by_interval(
        data: Dict[str, Dict],
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        interval: str = "month"
) -> pd.DataFrame:
    """
    Aggregates articles by the specified interval (month, quarter, or year).

    This function processes a dictionary of articles and their associated citations,
    aggregates them based on the specified time interval, and returns the count of core
    and non-core articles within each interval.

    Args:
        data (Dict[str, Dict]): A dictionary where the keys are article IDs and the values
                                 are dictionaries containing article details, including
                                 "publication_date", "incoming_citations", and "outgoing_citations".
        date_from (Optional[datetime], optional): The starting date for the aggregation range. Defaults to None.
        date_to (Optional[datetime], optional): The ending date for the aggregation range. Defaults to None.
        interval (str, optional): The time interval for aggregation.
                                  Can be "month", "quarter", or "year". Defaults to "month".

    Returns:
        pd.DataFrame: A DataFrame with the aggregated counts of core and non-core articles for each time interval.

    Raises:
        ValueError: If an unsupported interval is provided.

    Example:
        data = {
            "article1": {
                "publication_date": "2023-01-15",
                "incoming_citations": [{"id": "article2", "publication_date": "2023-02-01"}],
                "outgoing_citations": []
            },
            "article2": {
                "publication_date": "2023-02-01",
                "incoming_citations": [],
                "outgoing_citations": []
            }
        }

        aggregated_data = _aggregate_articles_by_interval(data, date_from=datetime(2023, 1, 1), interval="month")
        print(aggregated_data)
    """
    all_articles = []
    core_ids = set(data.keys())

    for article_id, article_data in data.items():

        # Add the core article
        pub_date = _parse_publication_date(article_data.get("publication_date"))
        core_id=set()
        if pub_date:
            all_articles.append({
                "id": article_id,
                "publication_date": pub_date,
                "is_core": True
            })
            core_id.add(article_id)

        # Process incoming and outgoing citations
        for citation_key in ["incoming_citations", "outgoing_citations"]:
            for cited_article in article_data.get(citation_key, []):
                cited_id=cited_article["id"]
                if cited_id not in core_ids:
                    pub_date = _parse_publication_date(cited_article.get("publication_date"))
                    if pub_date:
                        all_articles.append({
                            "id": cited_id,
                            "publication_date": pub_date,
                            "is_core": True if cited_id in core_id else False
                        })

    # Create a DataFrame
    df = pd.DataFrame(all_articles).drop_duplicates(subset=["id"])
    # Filter by date range
    if date_from is not None:
        df = df[df["publication_date"] >= date_from]
    if date_to  is not None:
        df = df[df["publication_date"] <= date_to]

    # Add time intervals
    if interval == "month":
        df["interval"] = df["publication_date"].dt.to_period("M")
    elif interval == "quarter":
        df["interval"] = df["publication_date"].dt.to_period("Q")
    elif interval == "year":
        df["interval"] = df["publication_date"].dt.to_period("Y")
    else:
        raise ValueError("Unsupported interval. Use 'month', 'quarter', or 'year'.")

    # Aggregate counts by interval and is_core
    aggregated = df.groupby(["interval", "is_core"]).size().unstack(fill_value=0)
    aggregated.columns = ["non-core" if not col else "core" for col in aggregated.columns]
    for col in ["core", "non-core"]:
        if col not in aggregated.columns:
            aggregated[col] = 0

    aggregated = aggregated[["core", "non-core"]]

    aggregated = aggregated.reset_index()

    return aggregated


def plot_article_trends(
        articles: dict,
        interval: str,
        color_core: str = "#1f78b4",
        color_periphery: str = "#a6cee3",
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        num_ticks: int = 10
) -> None:
    """
    Plots a stacked area chart based on the aggregated article data.

    This function generates a stacked area chart that visualizes the number of core and
    non-core articles over time, aggregated by the specified interval (month, quarter, year).

    Args:
        articles (dict): A dictionary where the keys are article IDs and the values are
                          dictionaries containing article details (including "publication_date",
                          "incoming_citations", and "outgoing_citations").
        interval (str): Time interval for the x-axis labels. Can be "month", "quarter", or "year".
        color_core (str, optional): Color for the "core" articles in the plot. Defaults to "orange".
        color_periphery (str, optional): Color for the "periphery" (non-core) articles in the plot.
                                         Defaults to "skyblue".
        date_from (Optional[pd.Timestamp], optional): Start date for the aggregation range. Defaults to None.
        date_to (Optional[pd.Timestamp], optional): End date for the aggregation range. Defaults to None.
        num_ticks (int, optional): The maximum number of x-axis ticks to display. Defaults to 10.

    Returns:
        None: The function generates and displays the stacked area chart.

    Raises:
        ValueError: If the `interval` is not one of "month", "quarter", or "year".

    Example:
        data = {
            "article1": {
                "publication_date": "2023-01-15",
                "incoming_citations": [{"id": "article2", "publication_date": "2023-02-01"}],
                "outgoing_citations": []
            },
            "article2": {
                "publication_date": "2023-02-01",
                "incoming_citations": [],
                "outgoing_citations": []
            }
        }

        plot_stacked_area_chart(data, interval="month", date_from=pd.Timestamp("2023-01-01"))
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    aggregated_data = _aggregate_articles_by_interval(articles, date_from=date_from, date_to=date_to, interval=interval)

    # Prepare data for plotting
    x = aggregated_data["interval"].astype(str)
    y_core = aggregated_data["core"]
    y_non_core = aggregated_data["non-core"]

    ax.fill_between(x, 0, y_core, label="Core", color=color_core, alpha=0.7)
    ax.fill_between(x, y_core, y_core + y_non_core, label="Periphery", color=color_periphery, alpha=0.7)

    # Configure chart
    ax.set_title(f"Number of Articles by {interval}", fontsize=26, fontweight='bold')
    ax.set_xlabel(f"Publication {interval.title()}", fontsize=22)
    ax.set_ylabel("Number of Articles", fontsize=22)
    ax.legend(fontsize=20)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Adjust x-axis ticks for better readability
    xticks = x[::max(1, len(x) // num_ticks)]  # Reduce the number of ticks to at most num_ticks
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, rotation=45, fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    plt.tight_layout()
    plt.show()



def _extract_topic(
    data: Dict[str, Dict],
    field_key: Optional[str] = "domain",
) -> pd.DataFrame:
    """
    Extracts topic or concept data from articles for aggregation.

    This function processes the articles dataset, extracting either topics or concepts based
    on the provided key, and returns a DataFrame containing the extracted information.
    Optionally filters by field or domain for topics, or by level for concepts.

    Args:
        data (Dict[str, Dict]): A dictionary where the keys are article IDs and the values are
                                 dictionaries containing article details such as "publication_date",
                                 "topics", "concepts", "incoming_citations", and "outgoing_citations".
        field_key (Optional[str], optional): Must be one of "field" or "domain". Defaults to "domain".

    Returns:
        pd.DataFrame: A DataFrame containing the extracted information, including article ID,
                      publication date, whether the article is core or periphery, and the name of the
                      topic.

    Example:
        data = {
            "article1": {
                "publication_date": "2023-01-15",
                "topics": [{"display_name": "AI", "field": {"display_name": "Computer Science"}}],
                "incoming_citations": [],
                "outgoing_citations": []
            },
            "article2": {
                "publication_date": "2023-02-01",
                "topics": [{"display_name": "Machine Learning", "field": {"display_name": "AI"}}],
                "incoming_citations": [{"id": "article1", "publication_date": "2023-01-10"}],
                "outgoing_citations": []
            }
        }

        extracted_data = _extract_topic_or_concept_data(data, "domain")
        print(extracted_data)
    """
    key="topics"
    extracted_data = []
    core_ids = set(data.keys())

    for article_id, article_data in data.items():

        pub_date = _parse_publication_date(article_data.get("publication_date"))
        is_core = True

        # Process core article topics/concepts
        score=0
        field_name = "None"
        for item in article_data.get(key, []):
            if  field_key and field_key in item:
                field_name = item[field_key]["display_name"]
            else:
                if score < item.get("score"):
                    field_name = item.get("display_name")
                    score = item.get("score")
        extracted_data.append({
            "id": article_id,
            "publication_date": pub_date,
            "is_core": is_core,
            "name": field_name
        })

        # Process citations
        for citation_key in ["incoming_citations", "outgoing_citations"]:
            for cited_article in article_data.get(citation_key, []):
                cited_id=cited_article["id"]
                if cited_id not in core_ids:
                    pub_date = _parse_publication_date(cited_article.get("publication_date"))
                    for item in cited_article.get(key, []):
                        if field_key and field_key in item:
                            field_name = item[field_key]["display_name"]
                        else:
                            field_name = item.get("display_name")

                        extracted_data.append({
                            "id": cited_id,
                            "publication_date": pub_date,
                            "is_core": False,
                            "name": field_name
                        })

    return pd.DataFrame(extracted_data)

def _aggregate_topics(
    data: pd.DataFrame,
    interval: str = "month",
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Aggregates topics or concepts data by the specified time interval.

    This function filters and aggregates data based on the time interval, and optionally
    filters by date range. It also returns the top N most frequent topics or concepts.

    Args:
        data (pd.DataFrame): DataFrame containing extracted topic or concept data.
        interval (str, optional): Time interval for aggregation ("month", "quarter", "year").
                                  Defaults to "month".
        date_from (Optional[datetime], optional): Start date for filtering the data. Defaults to None.
        date_to (Optional[datetime], optional): End date for filtering the data. Defaults to None.
        top_n (int, optional): The number of top topics or concepts to display. Defaults to 10.

    Returns:
        pd.DataFrame: Aggregated data for plotting, including counts of topics or concepts
                      per time interval, and whether the article is core or peripheral.

    Raises:
        ValueError: If the `interval` is not one of "month", "quarter", or "year".

    Example:
        # Assuming `extracted_data` is a DataFrame containing extracted topic or concept data
        aggregated_data = _aggregate_topic_or_concept_data(extracted_data, interval="month", top_n=5)
        print(aggregated_data)
    """
    if date_from:
        data = data[data["publication_date"] >= date_from]
    if date_to:
        data = data[data["publication_date"] <= date_to]

    # Add time intervals
    if interval == "month":
        data["interval"] = data["publication_date"].dt.to_period("M")
    elif interval == "quarter":
        data["interval"] = data["publication_date"].dt.to_period("Q")
    elif interval == "year":
        data["interval"] = data["publication_date"].dt.to_period("Y")
    else:
        raise ValueError("Unsupported interval. Use 'month', 'quarter', or 'year'.")

    # Get top N topics/concepts
    top_items = data["name"].value_counts().head(top_n).index
    data = data[data["name"].isin(top_items)]

    # Aggregate counts by interval, is_core, and name
    aggregated = data.groupby(["interval", "is_core", "name"]).size().unstack(fill_value=0)
    return aggregated

def plot_topic_trends(
    articles: Dict,
    interval: str = "month",
    top_n_colors: Optional[list] = None,
    show_periphery: bool = True,
    show_core: bool = False,
    field_key: Optional[str] = "domain",
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    top_n: int = 10,
    num_ticks: int = 10
) -> None:
    """
    Plots a grouped stacked bar chart for aggregated data based on topics or concepts.

    Args:
        articles (dict): JSON-like dataset containing articles.
        interval (str): Time interval for aggregation ("month", "quarter", "year").
        top_n_colors (Optional[list], optional): List of n_colors to use for the bars.
        show_periphery (bool, optional): Whether to display non-core articles.
        show_core (bool, optional): Whether to display core articles.
        field_key (Optional[str], optional): Key to filter by for topics ('field' or 'domain'). Default 'domain'.
        date_from (Optional[datetime], optional): Start date for filtering.
        date_to (Optional[datetime], optional): End date for filtering.
        top_n (int, optional): Number of top topics to display.
        num_ticks (int, optional): Maximum number of ticks on the x-axis.

    Raises:
        ValueError: If invalid parameters are passed, such as unsupported intervals, keys, or field keys.

    Example:
        plot_multiple_stacked_bar_chart(articles, interval="month", top_n=5, show_core=True, key="topics")
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    if not  show_periphery and not show_core:
        raise ValueError("You must display at least one of the following: core or periphery (show_periphery, show_core).")
    if interval not in ["month","quarter","year"]:
        raise ValueError("Unsupported interval. Use 'month', 'quarter', or 'year'.")
    if field_key not in ["field", "domain"]:
        raise ValueError("Unsupported 'field_key'.You must use 'field' or 'domain'.")
    key = field_key

    if top_n_colors is None:
        colormap = plt.get_cmap('tab20')
        top_n_colors = [colors.to_hex(c) for c in colormap(np.linspace(0, 1, top_n + 1))]

        # Extract and aggregate data
    data = _extract_topic(articles, key)
    aggregated_data = _aggregate_topics(data, interval=interval, date_from=date_from, date_to=date_to, top_n=top_n)
    if aggregated_data.empty:
        print("No data available for the selected parameters.")
        return
    # Prepare data for plotting
    unique_intervals = aggregated_data.index.get_level_values('interval').unique()
    x = range(len(unique_intervals))  # Numerical indices for x-axis
    interval_labels = [str(interval) for interval in unique_intervals]  # Labels for x-axis

    aggregated_data=aggregated_data.reset_index()
    aggregated_data = aggregated_data.set_index('interval')
    if show_core:
        # Ensure all columns and intervals are aligned
        core_data = (
            aggregated_data[aggregated_data['is_core'] == True]
            .reindex(unique_intervals, fill_value=0)
            .sort_index()
        )
        core_data=core_data.drop('is_core',axis=1)
        if not show_periphery:
            core_data = core_data.loc[:, (core_data != 0).any(axis=0)]

    if show_periphery:

        non_core_data = (
            aggregated_data[aggregated_data['is_core'] == False]
            .reindex(unique_intervals, fill_value=0)
            .sort_index()
        )
        non_core_data=non_core_data.drop('is_core',axis=1)
        if not show_periphery:
            non_core_data = non_core_data.loc[:, (non_core_data != 0).any(axis=0)]

    # Offsets for grouped bars
    bar_width = 0.4  # Width of each group
    x_core = [val - bar_width / 2 for val in x]  # Left bar positions
    x_non_core = [val + bar_width / 2 for val in x]  # Right bar positions
    # Plot core data
    bottom_core = pd.Series(0, index=unique_intervals)

    if show_core:

        for column in core_data.columns:
            ax.bar(
                x_core, core_data[column].values,
                label=f"{column}",
                bottom=bottom_core.values,
                color=top_n_colors[core_data.columns.get_loc(column) % top_n],  # Cycle n_colors
                alpha=0.8, width=bar_width
            )
            bottom_core += core_data[column]

        if show_periphery and show_core:
            # Add symbols on top of bars
            ax.scatter(x_core, bottom_core.values + core_data[column].values, marker=7, color="black", s=30)
    # Plot non-core data
    bottom_non_core = pd.Series(0, index=unique_intervals)
    if show_periphery:
        for column in non_core_data.columns:
            ax.bar(
                x_non_core, non_core_data[column].values,
                bottom=bottom_non_core.values,
                color=top_n_colors[core_data.columns.get_loc(column) % 10],  # Cycle n_colors
                alpha=0.8, width=bar_width
            )
            bottom_non_core += non_core_data[column]
        if show_periphery and show_core:
            # Add symbols on top of bars
            ax.scatter(x_non_core, bottom_non_core.values + core_data[column].values, marker=10, color="black", s=30)

    # Configure chart
    ax.set_title(f"Top {top_n} {key} by {interval}", fontsize=26, fontweight='bold')
    ax.set_xlabel(f"Publication {interval.title()}", fontsize=22)
    ax.set_ylabel("Number of Articles", fontsize=22)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Adjust x-axis ticks for better readability
    xticks = list(x)[::max(1, len(x) // num_ticks)]
    interval_labels = list(interval_labels)[::max(1, len(interval_labels) // num_ticks)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(interval_labels, rotation=45,fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    markers=[]
    markers_labels=[]
    if show_core and show_periphery:
        core_marker = Line2D([0], [0], marker=7, color='black', markersize=6, linestyle='None', label='Core Articles')
        markers.append(core_marker)
        markers_labels.append('Core Articles')
        periphery_marker = Line2D([0], [0], marker=10, color='black', markersize=6, linestyle='None',
                                  label='Periphery Articles')
        markers.append(periphery_marker)
        markers_labels.append("Periphery Articles")

    # Get existing legend elements
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Remove duplicate labels

    # Add custom markers to legend
    ax.legend(
        (list(by_label.values()) + markers)[::-1],
        (list(by_label.keys()) + markers_labels)[::-1],
              fontsize=20, loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.show()




def plot_top_authors(
    articles, field_key="domain",
    date_from=None, date_to=None, num_authors=5, by_citations=False,
    show_periphery=True, show_core=True, n_colors=None):
    """
    Plots a stacked bar chart showing the top authors and their articles or citations split by topics or concepts.

    Args:
        articles (dict): JSON-like dataset containing articles with metadata.
        field_key (str, optional): Field key to filter topics by ('field' or 'domain').
        date_from (datetime, optional): Start date for filtering articles by publication date.
        date_to (datetime, optional): End date for filtering articles by publication date.
        num_authors (int): Number of top authors to display on the chart.
        by_citations (bool): If True, ranks authors by the number of citations instead of the number of publications.
        show_periphery (bool): If True, includes non-core articles in the analysis.
        show_core (bool): If True, includes core articles in the analysis.
        n_colors (list, optional): List of hex color codes for the chart bars. Defaults to a predefined colormap.

    Raises:
        ValueError: If neither `show_periphery` nor `show_core` is True.
        ValueError: If `field_key` is invalid.

    Returns:
        None: The function directly displays the chart using matplotlib.

    Example Usage:
        plot_author_stacked_bar_chart(
            articles=my_articles,
            date_from=datetime(2020, 1, 1),
            date_to=datetime(2022, 12, 31),
            num_authors=10,
            by_citations=False,
            n_colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        )
    """

    if not  show_periphery and not show_core:
        raise ValueError("You must display at least one of the following: core or periphery (show_periphery, show_core).")

    if field_key not in ["field", "domain"]:
            raise ValueError("Unsupported 'field_key'. When using key == topics, you must use 'field' or 'domain'.")
    key=field_key
    if n_colors is None:
        colormap = plt.get_cmap('tab10')
        n_colors = [colors.to_hex(c) for c in colormap(np.linspace(0, 1, 10 + 1))]

    author_data = []
    done=set()
    for article_id, article_data in articles.items():
        done.add(article_id)
        pub_date = _parse_publication_date(article_data.get("publication_date"))
        if date_from and (not pub_date or pub_date < date_from):
            continue
        if date_to and (not pub_date or pub_date > date_to):
            continue

        # Find the dominant topic
        field_name = "None"
        if article_data["primary_topic"] is not None:
                field_name = article_data["primary_topic"][field_key]["display_name"]


        # Process authorships
        for authorship in article_data.get("authorships", []):
            author = authorship.get("author", {})
            author_name = author.get("display_name", "Unknown Author")
            if by_citations:
                count = article_data.get("cited_by_count", 0)  # Count citations
            else:
                count = 1  # Count publications
            author_data.append({
                "author": author_name,
                key: field_name,
                "count": count
            })
        if show_periphery:
            for citation_key in ["incoming_citations", "outgoing_citations"]:
                for related_article in article_data.get(citation_key, []):
                    related_article_id = related_article["id"]
                    if related_article_id not in done:
                        done.add(related_article_id)
                        for authorship in related_article.get("authorships", []):
                            author = authorship.get("author", {})
                            author_name = author.get("display_name", "Unknown Author")
                            if by_citations:
                                count = related_article.get("cited_by_count", 0)  # Count citations
                            else:
                                count = 1  # Count publications
                            author_data.append({
                                "author": author_name,
                                key: field_name,
                                "count": count
                            })



    # Create a DataFrame
    df = pd.DataFrame(author_data)
    if df.empty:
        raise Exception("No data using this filter")
    # Aggregate by author and topic
    aggregated = df.groupby(["author", key])["count"].sum().unstack(fill_value=0)

    # Rank authors by total count
    top_authors = aggregated.sum(axis=1).nlargest(num_authors).index
    aggregated = aggregated.loc[top_authors]

    # Plot stacked bar chart
    visible_topics = aggregated.loc[:, (aggregated != 0).any(axis=0)]

    fig, ax = plt.subplots(figsize=(14, 8))
    bottom = pd.Series(0, index=aggregated.index)

    for topic in visible_topics.columns:
        ax.bar(
            aggregated.index, visible_topics[topic],
            bottom=bottom,
            label=topic,
            color=n_colors[visible_topics.columns.get_loc(topic) % len(n_colors)]
        )
        bottom += visible_topics[topic]

    label="number of published articles"
    if by_citations:
        label = "number of citations"

    ax.legend(
        title=f"{key}",
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=22
    )


    # Configure chart
    ax.set_title(f"Top {num_authors} authors by {label}", fontsize=26, fontweight='bold')
    ax.set_xlabel("Authors", fontsize=22)
    ax.set_ylabel("Number of Articles" if not by_citations else "Number of Citations", fontsize=14)
    ax.legend(title=key, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16, title_fontsize=22)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=45, ha="right",fontsize=20)
    ax.tick_params(axis='y', labelsize=20)

    plt.tight_layout()
    plt.show()




def plot_top_keywords_from_abstracts(
    articles, date_from=None, date_to=None, show_core=True,
    show_periphery=False, top_n=10, ngram_range=(1, 2),color="#abd9e9"
):
    """
    Plots the most frequent keywords extracted from abstracts of articles,
    including citations when `show_periphery=True`.

    Args:
        articles (dict): JSON-like dataset containing articles with metadata.
        date_from (datetime, optional): Start date for filtering articles by publication date.
        date_to (datetime, optional): End date for filtering articles by publication date.
        show_core (bool): If True, include core articles in the analysis.
        show_periphery (bool): If True, include non-core articles and their citations.
        top_n (int): Number of top keywords to display.
        ngram_range (tuple): N-gram range for extracting keywords with KeyBERT.
        color (string): hexadecimal color of bars i.e. ,"#abd9e9"

    Raises:
        ValueError: If neither `show_core` nor `show_periphery` is True.

    Returns:
        None: The function directly displays the chart using matplotlib.
    """
    if not show_core and not show_periphery:
        raise ValueError("You must display at least one of the following: core or periphery (show_core, show_periphery).")

    kw_model = KeyBERT()
    keyword_counts = Counter()



    # Process articles
    core_article_id=set(articles.keys())

    for article_id, article_data in tqdm(articles.items()):

        pub_date = _parse_publication_date(article_data.get("publication_date"))
        if date_from and (not pub_date or pub_date < date_from):
            continue
        if date_to and (not pub_date or pub_date > date_to):
            continue


        # Extract abstracts for the main article and its periphery
        abstracts = []

        # Main article abstract
        main_abstract = article_data.get("abstract", "")
        if main_abstract:
            abstracts.append(main_abstract)

        # Periphery: citations and cited articles
        if show_periphery:
            for citation_key in ["incoming_citations", "outgoing_citations"]:
                for related_article in article_data.get(citation_key, []):
                    related_article_id=related_article["id"]
                    pub_date = _parse_publication_date(related_article.get("publication_date"))
                    if date_from and (not pub_date or pub_date < date_from):
                        continue
                    if date_to and (not pub_date or pub_date > date_to):
                        continue
                    if related_article_id in core_article_id:
                        continue
                    cited_abstract = related_article.get("abstract", "")
                    if cited_abstract:
                        abstracts.append(cited_abstract)
                    core_article_id.add(related_article_id)



        # Extract keywords from abstracts
        for abstract in abstracts:
            keywords = kw_model.extract_keywords(
                abstract, keyphrase_ngram_range=ngram_range
            )
            # Add keywords to the counter (ensuring uniqueness per article)
            keyword_counts.update(set(kw for kw, _ in keywords))

    # Get the top `top_n` keywords
    most_common_keywords = keyword_counts.most_common(top_n)

    # Separate keywords and their counts for plotting
    keywords, counts = zip(*most_common_keywords)

    # Plot the bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(keywords, counts, color=color, edgecolor='black')

    ax.set_title(f"Top {top_n} Keywords from Abstracts", fontsize=26, fontweight='bold')
    ax.set_xlabel("Keywords", fontsize=22)
    ax.set_ylabel("Frequency", fontsize=22)
    plt.xticks(rotation=45, ha="right",fontsize=20)
    ax.tick_params(axis='y', labelsize=20)

    plt.tight_layout()
    plt.show()



def plot_keyword_trends(
    articles, date_from=None, date_to=None, show_core=True,
    show_periphery=False, top_n=10, ngram_range=(1, 2), interval="month", n_colors=None
):
    """
    Plots a line chart showing the trends of top keywords over time.

    Args:
        articles (dict): JSON-like dataset containing articles with metadata.
        date_from (datetime, optional): Start date for filtering articles by publication date.
        date_to (datetime, optional): End date for filtering articles by publication date.
        show_core (bool): If True, include core articles in the analysis.
        show_periphery (bool): If True, include non-core articles and their citations.
        top_n (int): Number of top keywords to display.
        ngram_range (tuple): N-gram range for extracting keywords with KeyBERT.
        interval (str): Time interval for grouping ('month', 'quarter', 'year').
        n_colors (list, optional): List of hex colors for the keyword lines. Repeats if shorter than top_n.
    Raises:
        ValueError: If neither `show_core` nor `show_periphery` is True.

    Returns:
        None: The function directly displays the chart using matplotlib.
    """
    if not show_core and not show_periphery:
        raise ValueError("You must display at least one of the following: core or periphery (show_core, show_periphery).")

    if interval not in ["month", "quarter", "year"]:
        raise ValueError("Invalid interval. Choose from 'month', 'quarter', or 'year'.")


    kw_model = KeyBERT()
    keyword_counts_by_time = defaultdict(Counter)

    core_article_id = set(articles.keys())

    for article_id, article_data in tqdm(articles.items()):
        pub_date = _parse_publication_date(article_data.get("publication_date"))


        # Determine time period based on interval
        if interval == "month":
            time_period = pub_date.strftime("%Y-%m")
        elif interval == "quarter":
            quarter = (pub_date.month - 1) // 3 + 1
            time_period = f"{pub_date.year}-Q{quarter}"
        elif interval == "year":
            time_period = pub_date.strftime("%Y")

        # Main article abstract

        if not (date_from and (not pub_date or pub_date < date_from) or date_to and (not pub_date or pub_date > date_to)):
            abstract = article_data.get("abstract", "")
            keywords = kw_model.extract_keywords(
                abstract, keyphrase_ngram_range=ngram_range
            )
            keyword_counts_by_time[time_period].update(
                set(kw for kw, _ in keywords)
            )

        # Periphery: citations and cited articles
        if show_periphery:
            for citation_key in ["incoming_citations", "outgoing_citations"]:
                for related_article in article_data.get(citation_key, []):
                    pub_date = _parse_publication_date(related_article.get("publication_date"))
                    if date_from and (not pub_date or pub_date < date_from):
                        continue
                    if date_to and (not pub_date or pub_date > date_to):
                        continue
                    related_article_id = related_article["id"]
                    if related_article_id in core_article_id:
                        continue
                    cited_abstract = related_article.get("abstract", "")
                    if not cited_abstract:
                        continue
                    core_article_id.add(related_article_id)

                    # Determine time period based on interval
                    if interval == "month":
                        time_period = pub_date.strftime("%Y-%m")
                    elif interval == "quarter":
                        quarter = (pub_date.month - 1) // 3 + 1
                        time_period = f"{pub_date.year}-Q{quarter}"
                    elif interval == "year":
                        time_period = pub_date.strftime("%Y")
                    keywords = kw_model.extract_keywords(
                        cited_abstract, keyphrase_ngram_range=ngram_range
                    )
                    keyword_counts_by_time[time_period].update(
                        set(kw for kw, _ in keywords)
                    )

    # Aggregate data for top keywords
    aggregated_counts = defaultdict(lambda: defaultdict(int))
    for time_period, counts in keyword_counts_by_time.items():
        for keyword, count in counts.items():
            aggregated_counts[keyword][time_period] += count

    # Get top `top_n` keywords overall
    overall_counts = Counter()
    for keyword, time_data in aggregated_counts.items():
        overall_counts[keyword] += sum(time_data.values())
    top_keywords = [kw for kw, _ in overall_counts.most_common(top_n)]

    # Prepare data for plotting
    df = pd.DataFrame.from_dict(aggregated_counts, orient="index").fillna(0)
    df = df.loc[top_keywords]  # Keep only top keywords
    df = df.T.sort_index()  # Transpose and sort by time

    # Plot line chart

    fig, ax = plt.subplots(figsize=(14, 8))

    texts = []
    if n_colors is None:
        colormap = plt.get_cmap('tab10')
        n_colors = [to_hex(c) for c in colormap(np.linspace(0, 1, 10 + 1))]
    for i,keyword in enumerate(top_keywords):
        y_values = df[keyword]

        color = n_colors[i % len(n_colors)] if n_colors else None
        ax.plot(df.index, y_values, marker="o", linestyle="--", label=keyword, alpha=0.7, color=color)


        num_points = len(df.index)
        i=(i%num_points)+1
        x_pos = df.index[-i]
        y_pos = df.loc[x_pos, keyword]

        text = ax.text(x_pos, y_pos, keyword, fontsize=20, verticalalignment="bottom",
                       bbox=dict(facecolor='white', alpha=0.3, edgecolor='white'))
        texts.append(text)
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=2.5))

    ax.set_title("Keyword Trends Over Time", fontsize=26, fontweight='bold')
    ax.set_xlabel(f"Time Period ({interval})", fontsize=22)
    ax.set_ylabel("Frequency", fontsize=22)
    ax.legend(title="Keywords", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=20)
    plt.xticks(rotation=45, ha="right",fontsize=20)
    ax.tick_params(axis='y', labelsize=20)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()