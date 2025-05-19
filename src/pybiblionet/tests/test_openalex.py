import json
from pathlib import Path

from pybiblionet.openalex.core import string_generator_from_lite_regex, retrieve_articles, create_citation_graph, \
    create_coauthorship_graph, export_articles_to_csv, export_authors_to_csv, export_venues_to_csv, \
    export_institutions_to_csv, export_articles_to_scopus

# This script retrieves academic articles from OpenAlex related to the concept of "15-minute city"
# using a regex-based query. It exports the retrieved articles and their authors to CSV files,
# then generates and saves two graph representations: a citation graph and a co-authorship graph.
# Finally, it prints the number of nodes and edges in each graph to give an overview of their structure.
if __name__ == "__main__" :


    queries = string_generator_from_lite_regex("(15)( )(minute|min)( )(city)")

    mail = "youremail@example.com"
    from_publication_date = "2019-01-01"
    to_publication_date = None
    json_file_path = retrieve_articles(queries, mail, from_publication_date, to_publication_date)
    json_file_path= "query_results/query_result_1ea020320b230de5a973a39682eaa53dce89a9bb026b441a5f825232.json"

    export_institutions_to_csv(json_file_path,
                           fields_to_export=None,
                           export_path="15minute_institutions.csv"
                           )

    export_venues_to_csv(json_file_path,
                           fields_to_export=None,
                           export_path="15minute_venues.csv"
                           )

    export_articles_to_csv(json_file_path,
                           fields_to_export=None,
                           export_path="15minute_articles.csv"
                           )
    export_authors_to_csv(json_file_path,
                           fields_to_export=None,#["author id", "author orcid", "author display_name"],
                           export_path="15minute_authors.csv"
                           )

    articles = json.load(open(json_file_path))
    G_citation=create_citation_graph(articles,"15minute_citation_graph.GML",periphery=True)

    print(G_citation.number_of_nodes(),G_citation.number_of_edges())

    G_coauthorship=create_coauthorship_graph(articles,"15minute_coauthorship_graph.GML",periphery=True)

    print(G_coauthorship.number_of_nodes(),G_coauthorship.number_of_edges())

