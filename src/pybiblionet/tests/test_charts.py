from pybiblionet.bibliometric_analysis.charts import plot_topic_trends, plot_article_trends, \
    plot_top_authors, plot_top_keywords_from_abstracts, plot_keyword_trends
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import json
# This script performs various bibliometric analyses on a set of academic articles stored in a JSON file (recovered using test_openalex.py).
# It generates several visualizations:
# 1. Article publication trends over time (by month, quarter, or year).
# 2. Topic trends over time, highlighting core and periphery topics.
# 3. Top authors based on citation count, focusing on the most cited authors.
# 4. Top keywords extracted from article abstracts.
# 5. Keyword trends over time, showing the frequency of keywords and their evolution.
# The script uses `matplotlib` for plotting and supports customization for the time intervals, number of authors,
# and other aspects of the analysis.
if __name__ == "__main__" :


    json_file_path= "query_results/query_result_1ea020320b230de5a973a39682eaa53dce89a9bb026b441a5f825232.json"
    articles = json.load(open(json_file_path))



    plot_article_trends(articles,
                        date_from=datetime(2019, 1, 1),
                        interval="month",  # Change to "month" or "quarter" or "year" as needed
                        date_to=datetime(2023, 12, 31),
                        num_ticks=20)
    plot_topic_trends(
        articles,
        top_n_colors= ["#a6cee3","#1f78b4","#b2df8a","#33a02c","#d7191c"],
        show_core=True,
        show_periphery=True,
        interval="quarter", # Change to "quarter" or "year"
        date_from=datetime(2019, 1, 1),
        date_to=datetime(2022, 12, 31),
        top_n=5,
        )
    plot_top_authors(
        articles,
        date_from=datetime(2019, 1, 1),
        date_to=datetime(2022, 12, 31),
        num_authors=10,
        by_citations=True,
        show_periphery=False,
        n_colors=["#a6cee3","#1f78b4","#b2df8a","#33a02c","#d7191c","#fdae61"]
    )

    plot_top_keywords_from_abstracts(articles)


    plot_keyword_trends(
        articles=articles,
        date_from=datetime(2020, 1, 1),
        date_to=datetime(2023, 1, 1),
        show_core=True,
        show_periphery=True,
        top_n=10,
        ngram_range=(1, 2),
        interval="quarter",
        n_colors=["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#d7191c", "#fdae61"]
    )














