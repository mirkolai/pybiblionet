import gzip
import hashlib
import re
from collections import Counter
from itertools import product
from typing import List, Optional, Dict

from time import  sleep
from urllib import parse
from os import path
import requests
import json
from pathlib import Path
import copy
import csv
import networkx as nx
from tqdm import tqdm
from pybiblionet.openalex.utils import check_date_format, abstract_inverted_index_to_abstract


def string_generator_from_lite_regex(queries_regex: str) -> List[str]:
    """
    Generates a list of query strings from a simplified regular expression pattern.

    This function takes a simplified regular expression pattern, extracts groups
    of alternatives, and produces all possible combinations of these alternatives
    as separate query strings. Each resulting query string is enclosed in double
    quotes.

    :param queries_regex: A string representing a simplified regular expression pattern.
                          The pattern should contain groups of alternatives enclosed in
                          parentheses and separated by the pipe character '|'.
                          Example: r"(1|2|3)(a|b|c)"

    :return: A list of strings, each representing a possible combination of the alternatives
             defined in the input pattern. Each string is enclosed in double quotes.

    Example:

    >>> queries_regex = r'"(1|3|5|10|15|20|30|45|one|three|five|ten|fifteen|twenty|thirty|fortyfive|x)( )(minute|minutes|min)( )(city|neighborhood|neighbourhood|urban|town)"'
    >>> result = string_generator_from_lite_regex(queries_regex)
    >>> print(result)
    ['"1 minute city"', '"1 minute neighborhood"', '"1 minute neighbourhood"', ...]

    """
    groups=[]
    for group in re.findall(r"\([ a-zA-Z0-9\-\|]{1,}\)",queries_regex):
        groups.append(group.replace("(","").replace(")","").split("|"))

    queries=[]
    for item in product(*groups):
        if len(item)==len(groups):
            queries.append("\""+''.join(item)+"\"")

    return queries

def query_OpenAlex(api_type: str, parameter: Optional[str], mail: str,
                   from_publication_date: Optional[str] = None,
                   to_publication_date: Optional[str] = None,
                   cache: bool = True) -> List[Dict[str, Dict]]:
    """
    Queries the OpenAlex API for works based on the type of query (search, cite, or cited_by).

    :param api_type: The type of API query ('search', 'cite', or 'cited_by').
    :param parameter: The article ID for 'cite' and 'cited_by' queries. Search string for 'search' queries.
    :param mail: The email address to be included in the API request for identification.
    :param from_publication_date: Optional start date for filtering results (format: 'YYYY-MM-DD').
    :param to_publication_date: Optional end date for filtering results (format: 'YYYY-MM-DD').
    :param cache: Boolean indicating whether to use caching for API responses (default True).
    :return: A list of results matching the query.
    """
    if api_type == 'search':
        cache_path = Path("cache") / "search"
        base_url = f'https://api.openalex.org/works?filter=title_and_abstract.search:{parse.quote_plus(parameter)}'
    elif api_type == 'cite':
        cache_path = Path("cache") / "cite"
        parameter = parameter.split("/")[-1]
        base_url = f'https://api.openalex.org/works?filter=cites:{parameter}'
    elif api_type == 'cited_by':
        cache_path = Path("cache") / "cite_by"
        parameter = parameter.split("/")[-1]
        base_url = f'https://api.openalex.org/works?filter=cited_by:{parameter}'
    else:
        raise ValueError("Invalid api_type. Must be 'search', 'cite', or 'cited_by'.")

    # Ensure the directory exists
    cache_path.mkdir(parents=True, exist_ok=True)

    if from_publication_date and check_date_format(from_publication_date):
        base_url += f',from_publication_date:{from_publication_date}'
    if to_publication_date and check_date_format(to_publication_date):
        base_url += f',to_publication_date:{to_publication_date}'

    base_url += f'&per-page=200&mailto={mail}'
    cursor = "*"
    results = []
    Path(cache_path).mkdir(parents=True, exist_ok=True)

    while cursor:
        try:
            this_url = f'{base_url}&cursor={cursor}'
            cursor = False
            print(this_url)
            def hash_url(url:str)->str:
                return hashlib.md5(url.encode('utf-8')).hexdigest()
            this_url_filename=this_url.split('?')[1]
            filename = cache_path / f"{hash_url(this_url_filename)}.json.gz"

            if cache and path.isfile(filename):
                with gzip.open(filename, 'rt', encoding='utf-8') as f:
                    result = json.load(f)
            else:
                response = requests.get(this_url)
                response.raise_for_status()
                txt = response.text

                if cache:
                    with gzip.open(filename, 'wt', encoding='utf-8') as f:
                        f.write(txt)

                result = json.loads(txt)
                sleep(3)  # to avoid hitting the rate limit

            for article in result['results']:
                if article['abstract_inverted_index'] is not None:
                    article['abstract'] = abstract_inverted_index_to_abstract(article['abstract_inverted_index'])

                del article['abstract_inverted_index']
                article["authorships_display_name"] = ", ".join(
                    [author["author"]["display_name"] for author in article.get("authorships", [])])
                results.append(article)


            print("total articles:",json.dumps(result['meta']["count"]), "currently retrieved:", len(results), "still to retreive:",  result["meta"]["count"] - len(results))
            if 'next_cursor' in result['meta'] and result["meta"]["count"] - len(results) > 0:
                cursor = result['meta']['next_cursor']
        except Exception as e:
            print(f"Error in query_OpenAlex_{api_type}:", e)
            sleep(3)
            break

    return results


def retrieve_articles(queries: list[str], mail: str,
                           from_publication_date: Optional[str] = None,
                           to_publication_date: Optional[str] = None)->str:
    """
    Retrieves core articles from the OpenAlex API based on a list of queries and other optional parameters.

    :param queries: A list of search queries to use for retrieving articles.
    :param mail: The email address to include in the API request for identification.
    :param from_publication_date: Optional start date for filtering results (format: 'YYYY-MM-DD').
    :param to_publication_date: Optional end date for filtering results (format: 'YYYY-MM-DD').

    This function performs the following steps:
    1. Iterates through the list of queries and retrieves articles from the OpenAlex search API .
    2. For each result, it retrieves the incoming and outgoing citations (by using the cite and cited_by APIs).
    3. Converts the abstract from an inverted index format to a readable one, if present.
    4. Stores the results in the `output` dictionary.
    5. Returns the file path of the json containing all the article for that query (the file is stored in the query_results folder).
    Example Usage:
    --------------
    queries = ["quantum computing", "artificial intelligence"]
    mail = "example@example.com"
    from_publication_date = "2020-01-01"
    to_publication_date = "2022-12-31"

    json_result_path = retrieve_articles(queries, mail, from_publication_date, to_publication_date)
    """
    output={}

    for query in queries:

        results=query_OpenAlex("search",query,  mail, from_publication_date, to_publication_date)
        result_ids = set()
        for result in results:
            result_ids.add(result['id'])

        for result in results:

            result["core"] = True
            result['incoming_citations'] =  query_OpenAlex("cite", result['id'], mail)
            result['outgoing_citations'] = query_OpenAlex("cited_by", result['id'], mail)

            output[result['id']] = copy.deepcopy(result)

    output_query_path=Path("query_results")
    Path(output_query_path).mkdir(parents=True, exist_ok=True)
    m = hashlib.sha224()
    m.update(str(queries).encode(encoding='UTF-8', errors='strict'))
    cache_key = m.hexdigest()
    output_query_path=Path("query_results")

    json_file_path = output_query_path / f"query_result_{cache_key}.json"
    json.dump(output,open(json_file_path,"w"))

    return str(json_file_path)


def export_articles_to_csv(json_file_path,fields_to_export: Optional[List[str]] = None,export_path=None):
    """
           Exports article information from a JSON file into a CSV file.

           The data to be exported is specified by the user through the `fields_to_export` parameter.
           Articles are categorized as either "core" or "periphery" based on their presence in the
           core articles set.

           Parameters:
           - json_file_path (str):
             The path to the JSON file that contains the articles data. The JSON should be structured
             such that it includes article information, including citations (both incoming and outgoing).

           - fields_to_export (Optional[List[str]]):
             A list of strings representing the fields to be included in the CSV export. If not specified,
             all available fields are exported. Available fields are:
               - "id"
               - "doi"
               - "title"
               - "abstract"
               - "publication_date"
               - "language"
               - "type"
               - "type_crossref"
               - "authorships_display_name"
               - "topics"
               - "is_retracted"
               - "cited_by_count"
               - "venue"

             If any fields in `fields_to_export` are not valid, a `ValueError` will be raised.

           - export_path (Optional[str]):
             The path where the CSV file will be saved. If not provided, the default path `"exports"`
             will be used. If a path is provided, the CSV file will be saved in that location.

           Returns:
           - None:
             This function does not return any value. It directly exports the articles data to a CSV file.

           Example usage:
           export_articles_to_csv("path/to/articles.json", fields_to_export=["id", "title", "doi"], export_path="output/articles.csv")
           """

    cache_path = Path("exports")


    field_mapping = lambda article_data: {
        "id": article_data.get("id", "") or "",
        "doi": article_data.get("doi", "") or "",
        "title": article_data.get("title", "") or "",
        "publication_year": article_data.get("publication_year", "") or "",
        "language": article_data.get("language", "") or "",
        "type": article_data.get("type", "") or "",
        "type_crossref": article_data.get("type_crossref", "") or "",
        "authorships display_name": ", ".join(
            [authorship.get("author", {}).get("display_name", "") for authorship in
             article_data.get("authorships", [])
             if isinstance(authorship, dict)]),

        "primary_topic display_name": (article_data.get("primary_topic") or {}).get("display_name", ""),
        "primary_topic subfield display_name": (article_data.get("primary_topic") or {}).get("subfield", {}).get(
            "display_name", ""),
        "primary_topic field display_name": (article_data.get("primary_topic") or {}).get("field", {}).get(
            "display_name", ""),
        "primary_topic domain display_name": (article_data.get("primary_topic") or {}).get("domain", {}).get(
            "display_name", ""),

        "cited_by_count": article_data.get("cited_by_count", "") or "",
        "core": article_data.get("core", "") or "False",
        "venue": (
                (
                    (article_data.get("primary_location") or {}).get("source")
                ) or {}
        ).get("display_name", ""),
        "publication_date": article_data.get("publication_date", ""),
    }

    available_fields = list(field_mapping({}).keys())

    if fields_to_export:
        invalid_fields = [field for field in fields_to_export if field not in available_fields]
        if invalid_fields:
            raise ValueError(f"The following fields are not available for exporting: {', '.join(invalid_fields)}. Available fields are : {', '.join(available_fields)}")
    else:
        fields_to_export=available_fields

    # Load the articles from the provided JSON file
    json_file_path = Path(json_file_path)  # Convert string to Path object
    if not json_file_path.is_file():
        raise ValueError(f"The input file does not exist: {json_file_path}")

    articles=json.load(open(json_file_path))

    core_articles=set()
    for article_id, article_data in articles.items():
        core_articles.add(article_id)
    # Determine the CSV file path
    if export_path is None:
        # Use pathlib to handle the export path and ensure cross-platform compatibility
        cache_key = Path(json_file_path).stem.replace(".json", "")
        csv_file_path = cache_path / f"articles_{cache_key}.csv"
    else:
        csv_file_path = Path(export_path)

    # Ensure the cache directory exists
    cache_path.mkdir(parents=True, exist_ok=True)


    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields_to_export+["scope"])
        writer.writeheader()
        already_save = set()
        for article_id, article_data in tqdm(articles.items(), desc="Exporting articles", unit="article"):

            if article_id not in already_save:
                row = {field: field_mapping(article_data)[field] for field in fields_to_export}
                row["scope"] = "core" if article_id in core_articles else "periphery"
                writer.writerow(row)
                already_save.add(article_id)

            for citation_type in ['incoming_citations','outgoing_citations']:
                for cite in article_data[citation_type]:
                    if cite["id"] not in already_save:

                        row={field: field_mapping(cite)[field] for field in fields_to_export}
                        row["scope"] = "core" if cite["id"] in core_articles else "periphery"
                        writer.writerow(row)
                        already_save.add(cite["id"])


    print(f"Export completed. The articles are saved in {csv_file_path}")



def export_articles_to_scopus(json_file_path,include_periphery: bool = True,export_path=None):
    """
           Exports article information from a JSON file into a  Scopus like CSV.



           Parameters:
           - json_file_path (str):
             The path to the JSON file that contains the articles' data. The JSON should be structured
             such that it includes article information, including citations (both incoming and outgoing).

           - include_periphery (Optional[bool]):
             If True, include periphery.

           - export_path (Optional[str]):
             The path where the CSV file will be saved. If not provided, the default path `"exports"`
             will be used. If a path is provided, the CSV file will be saved in that location.

           Returns:
           - None:
             This function does not return any value. It directly exports the articles data to a CSV file.

           Example usage:
           export_articles_to_scopus("path/to/articles.json", export_path="output/articles.csv")
           """

    cache_path = Path("exports")


    field_mapping = lambda article_data: {

        "Author": article_data.get("authorships_display_name", ""),
        "Authors full names": "; ".join([
            authorship.get("author", {}).get("display_name", "")+f"({authorship.get('id', '').split('/')[-1]})"
            for authorship in article_data.get("authorships", [])
            if isinstance(authorship, dict)
        ]),
        "Author(s) ID": "; ".join([
            (authorship.get("author", {}).get("id") or "").split("/")[-1]
            for authorship in article_data.get("authorships", [])
            if isinstance(authorship, dict)
        ]),
        "Title": article_data.get("title", ""),
        "Year": article_data.get("publication_year", ""),
        "Source title": (
            ((article_data.get("primary_location") or {}).get("source") or {}).get("display_name", "")
        ),
        "Volume": article_data.get("biblio", {}).get("volume", ""),
        "Issue": article_data.get("biblio", {}).get("issue", ""),
        "Art.No.": "",
        "Page start": article_data.get("biblio", {}).get("first_page", ""),
        "Page end": article_data.get("biblio", {}).get("last_page", ""),
        "Cited by": article_data.get("cited_by_count", ""),
        "DOI": (article_data.get("doi") or "").replace("https://doi.org/", ""),
        "Link": (article_data.get("primary_location") or {}).get("landing_page_url", ""),
        "Affiliations": "; ".join([
            aff
            for authorship in article_data.get("authorships", [])
            for aff in authorship.get("raw_affiliation_strings", [])
        ]),
        "Author(s) with affiliations": "; ".join([
            authorship.get("author", {}).get("display_name", "") + ", " +
            ", ".join(authorship.get("raw_affiliation_strings", []))
            for authorship in article_data.get("authorships", [])
            if isinstance(authorship, dict)
        ]),
        "Abstract": article_data.get("abstract", ""),
        "Author Keywords":"",
        "Index Keywords":"",
        "Molecular Sequence Numbers":"",
        "Chemicals/CAS":"",
        "Tradenames":"",
        "Manufacturers":"",
        "Funding Details": "",  # Non disponibile da OpenAlex
        "Funding Text": "",  # Non disponibile da OpenAlex


        "References": "",
        "Correspondence Address": "",
        "Editors": "",
        "Publisher": "",
        "Sponsors": "",
        "Conference name": "",
        "Conference date": "",
        "Conference location": "",
        "ISSN": ", ".join(
            ((article_data.get("primary_location") or {}).get("source") or {}).get("issn", []) or []
        ),
        "ISBN":"",
        "CODEN":"",
        "PubMed ID":"",
        "Language of Original Document":"",
        "Abbreviated Source Title":"",
        "Document Type": article_data.get("type_crossref", ""),
        "Publication Stage":"",
        "Open Access":"",
        "Source":"scopus",
        "EID": article_data.get("id", "").split("/")[-1],

        "Corresponding Author(s) ID": "; ".join([
            (authorship.get("author", {}).get("id") or "").split("/")[-1]
            for authorship in article_data.get("authorships", [])
            if authorship.get("is_corresponding", False)
        ]),
        "Correspondence Address": "; ".join([
            aff
            for authorship in article_data.get("authorships", [])
            if authorship.get("is_corresponding", False)
            for aff in authorship.get("raw_affiliation_strings", [])
        ]),
        "Publisher": (
            ((article_data.get("primary_location") or {}).get("source") or {}).get("host_organization_name", "")
        ),
        "Language of Original Document": article_data.get("language", ""),
        "Publication Stage": (article_data.get("primary_location") or {}).get("version", ""),
        "Open Access": str(article_data.get("open_access", {}).get("is_oa", False)),
        "OA URL": article_data.get("open_access", {}).get("oa_url", ""),
        "PubMed ID": "",  # Non disponibile da OpenAlex
        "Abbreviated Source Title": "",  # Non disponibile da OpenAlex
        "Source ID": "",  # Non disponibile da OpenAlex
        "Page count": "",  # Non disponibile direttamente
        "Sponsors": "",  # Non disponibile da OpenAlex
        "Publisher ID": "",  # Non disponibile da OpenAlex
        "CODEN": "",  # Non disponibile
        "PubMed Central ID": "",  # Non disponibile
        "Truncated Author(s)": "",  # Non disponibile
    }

    fields_to_export = list(field_mapping({}).keys())



    # Load the articles from the provided JSON file
    json_file_path = Path(json_file_path)  # Convert string to Path object
    if not json_file_path.is_file():
        raise ValueError(f"The input file does not exist: {json_file_path}")

    articles=json.load(open(json_file_path))

    core_articles=set()
    for article_id, article_data in articles.items():
        core_articles.add(article_id)
    # Determine the CSV file path
    if export_path is None:
        # Use pathlib to handle the export path and ensure cross-platform compatibility
        cache_key = Path(json_file_path).stem.replace(".json", "")
        csv_file_path = cache_path / f"articles_scopus_{cache_key}.csv"
    else:
        csv_file_path = Path(export_path)

    # Ensure the cache directory exists
    cache_path.mkdir(parents=True, exist_ok=True)


    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields_to_export+["scope"])
        writer.writeheader()
        already_save = set()
        for article_id, article_data in tqdm(articles.items(), desc="Exporting articles", unit="article"):

            if article_id not in already_save:
                row = {field: field_mapping(article_data)[field] for field in fields_to_export}
                row["scope"] = "core" if article_id in core_articles else "periphery"
                writer.writerow(row)
                already_save.add(article_id)

            for citation_type in ['incoming_citations','outgoing_citations']:
                for cite in article_data[citation_type]:
                    if not include_periphery and cite["id"] not in core_articles:
                        continue
                    if cite["id"] not in already_save:
                        row={field: field_mapping(cite)[field] for field in fields_to_export}
                        row["scope"] = "core" if cite["id"] in core_articles else "periphery"
                        writer.writerow(row)
                        already_save.add(cite["id"])


    print(f"Export completed. The articles are saved in {csv_file_path}")





def export_venues_to_csv(json_file_path,fields_to_export: Optional[List[str]] = None,export_path=None):
    """
           Exports venues information from a JSON file into a CSV file.

           The data to be exported is specified by the user through the `fields_to_export` parameter.
           Venues are aggregate as either "core" or "periphery" based on their presence in the
           core articles set.

           Parameters:
           - json_file_path (str):
             The path to the JSON file that contains the venues data. The JSON should be structured
             such that it includes venue information.

           - fields_to_export (Optional[List[str]]):
             A list of strings representing the fields to be included in the CSV export. If not specified,
             all available fields are exported. Available fields are:
               - primary_location source id
               - primary_location source display_name
               - primary_location source issn_l
               - primary_location source issn
               - primary_location source host_organization
               - primary_location source host_organization_name

             If any fields in `fields_to_export` are not valid, a `ValueError` will be raised.

           - export_path (Optional[str]):
             The path where the CSV file will be saved. If not provided, the default path `"exports"`
             will be used. If a path is provided, the CSV file will be saved in that location.

           Returns:
           - None:
             This function does not return any value. It directly exports the articles data to a CSV file.

           Example usage:
           export_venues_to_csv("path/to/articles.json", fields_to_export=["id", "title", "doi"], export_path="output/articles.csv")
           """
    """"""


    author_core=Counter()
    author_periphery=Counter()

    cache_path = Path("exports")
    field_mapping = lambda article_data: {
        "primary_location source id": ((article_data.get("primary_location", {}) or {}).get("source",{}) or {}).get("id", ""),
        "primary_location source display_name": ((article_data.get("primary_location", {}) or {}).get("source", {}) or {}).get("display_name", ""),
        "primary_location source issn_l": ((article_data.get("primary_location", {}) or {}).get("source", {}) or {}).get("issn_l", ""),
        "primary_location source issn": ", ".join(
            [ issn for issn in
              ((article_data.get("primary_location", {}) or {}).get("source", {}) or {}).get("issn") or [] ]),
        "primary_location source host_organization": ((article_data.get("primary_location", {}) or {}).get("source", {}) or {}).get("host_organization", ""),
        "primary_location source host_organization_name": ((article_data.get("primary_location", {}) or {}).get("source", {}) or {}).get("host_organization_name", ""),

    }

    available_fields = list(field_mapping({}).keys())

    if fields_to_export:
        invalid_fields = [field for field in fields_to_export if field not in available_fields]
        if invalid_fields:
            raise ValueError(f"The following fields are not available for exporting: {', '.join(invalid_fields)}. Available fields are : {', '.join(available_fields)}")
    else:
        fields_to_export=available_fields

    # Load the articles from the provided JSON file
    json_file_path = Path(json_file_path)  # Convert string to Path object
    if not json_file_path.is_file():
        raise ValueError(f"The input file does not exist: {json_file_path}")

    articles=json.load(open(json_file_path))

    core_articles = set(articles.keys())

    # Determine the CSV file path
    if export_path is None:
        # Use pathlib to handle the export path and ensure cross-platform compatibility
        cache_key = Path(json_file_path).stem.replace(".json", "")
        csv_file_path = cache_path / f"venues_{cache_key}.csv"
    else:
        csv_file_path = Path(export_path)

    # Ensure the cache directory exists
    cache_path.mkdir(parents=True, exist_ok=True)

    venues_to_save=[]
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields_to_export+["core","periphery"])
        writer.writeheader()
        already_done_articles = set()
        for article_id, article_data in tqdm(articles.items(), desc="Exporting articles", unit="article"):
            if article_id not in already_done_articles:

                row = {field: field_mapping(article_data)[field] for field in fields_to_export}
                venue_id=((article_data.get("primary_location", {}) or {}).get("source", {}) or {}).get("id", "")
                if article_id in core_articles:
                    author_core[venue_id] += 1
                else:
                    author_periphery[venue_id] += 1

                if venue_id not in [venue["primary_location source id"] for venue in venues_to_save]:
                    venues_to_save.append(row)
                already_done_articles.add(article_id)

            for citation_type in ['incoming_citations','outgoing_citations']:
                for cite in article_data[citation_type]:
                    if cite["id"] not in already_done_articles:
                        row={field: field_mapping(cite)[field] for field in fields_to_export}
                        venue_id = ((cite.get("primary_location", {}) or {}).get("source", {}) or {}).get("id","")
                        if cite["id"] in core_articles:
                            author_core[venue_id] += 1
                        else:
                            author_periphery[venue_id] += 1

                        print(row)
                        if venue_id not in [venue["primary_location source id"] for venue in venues_to_save]:
                            venues_to_save.append(row)
                        already_done_articles.add(cite["id"])

        for venue in venues_to_save:
            venue["core"]=author_core[venue['primary_location source id']]
            venue["periphery"]=author_periphery[venue['primary_location source id']]
            writer.writerow(venue)
    print(f"Export completed. The venues are saved in {csv_file_path}")








def export_authors_to_csv(json_file_path, fields_to_export: Optional[List[str]] = None, export_path=None):
    """
        Exports article information from a JSON file into a CSV file.

        The data to be exported is specified by the user through the `fields_to_export` parameter.
        Articles are categorized as either "core" or "periphery" based on their presence in the
        core articles set.

        Parameters:
        - json_file_path (str):
          The path to the JSON file that contains the articles data. The JSON should be structured
          such that it includes article information, including citations (both incoming and outgoing).

        - fields_to_export (Optional[List[str]]):
          A list of strings representing the fields to be included in the CSV export. If not specified,
          all available fields are exported. Available fields are:
            -   "author id"
            -   "author orcid"
            -   "author display_name"
            -   "institutions display_name"
            -   "institutions type"
            -   "countries"


          If any fields in `fields_to_export` are not valid, a `ValueError` will be raised.

        - export_path (Optional[str]):
          The path where the CSV file will be saved. If not provided, the default path `"exports"`
          will be used. If a path is provided, the CSV file will be saved in that location.

        Returns:
        - None:
          This function does not return any value. It directly exports the articles data to a CSV file.

        Example usage:
        export_articles_to_csv("path/to/articles.json", fields_to_export=["id", "orcid", "display_name"], export_path="output/authors.csv")
        """

    author_core=Counter()
    author_periphery=Counter()

    field_mapping = lambda data : {
                "author id": (data.get("author", {})  or {}).get("id",""),
                "author orcid": (data.get("author", {}) or {}).get("orcid",""),
                "author display_name": (data.get("author", {}) or {}).get("display_name",""),
                "institutions display_name": " ".join([inst["display_name"] for inst in (data.get("institutions", []) or [])]),
                "institutions type": " ".join([inst["type"] for inst in (data.get("institutions", []) or [])]),
                "countries": " ".join((data.get("countries", []) or []))
            }
    # Available fields for authors
    available_fields = list(field_mapping({}).keys())

    if fields_to_export is not None:
        invalid_fields = [field for field in fields_to_export if field not in available_fields]
        if invalid_fields:
            raise ValueError(
                f"The following fields are not available for exporting: {', '.join(invalid_fields)}. Available fields are : {', '.join(available_fields)}")
    else:
        fields_to_export = available_fields
    # Load the articles from the provided JSON file
    json_file_path = Path(json_file_path)  # Convert string to Path object
    if not json_file_path.is_file():
        raise ValueError(f"The input file does not exist: {json_file_path}")

    articles = json.load(open(json_file_path,"r"))

    # Prepare the output CSV file path
    if export_path is None:
        cache_key = Path(json_file_path).stem.replace(".json", "")
        csv_file_path = Path("exports") / f"authors_{cache_key}.csv"
    else:
        csv_file_path = Path(export_path)

    # Ensure the directory exists
    csv_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Open the CSV file for writing
    already_done_articles=set()
    authors_to_save=[]
    core_articles = set(articles.keys())
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields_to_export+["core","periphery"])
        writer.writeheader()

        already_saved_authors = set()

        # Loop over the articles
        for article_id, article_data in tqdm(articles.items(), desc="Exporting authors", unit="article"):
            if article_id not in already_done_articles:

                for author in article_data.get('authorships', []):
                    author_id = author.get("author").get("id")

                    # Skip authors that have already been saved
                    row = {field: field_mapping(author)[field] for field in fields_to_export}
                    if article_id in core_articles:
                        author_core[author_id] += 1
                    else:
                        author_periphery[author_id] += 1
                    if author_id not in already_saved_authors:
                        authors_to_save.append(row)

                        already_saved_authors.add(author_id)

            for citation_type in ['incoming_citations','outgoing_citations']:
                for cite in article_data[citation_type]:
                    article_id=cite.get("id")
                    if article_id not in already_done_articles:
                        for author in cite.get('authorships', []):
                            author_id = author.get("author").get("id")

                            # Skip authors that have already been saved
                            row = {field: field_mapping(author)[field] for field in fields_to_export}
                            if article_id in core_articles:
                                author_core[author_id]+=1
                            else:
                                author_periphery[author_id]+=1
                            if author_id not in already_saved_authors:
                                authors_to_save.append(row)
                                already_saved_authors.add(author_id)
                    already_done_articles.add(author_id)

            already_done_articles.add(article_id)
        for author in authors_to_save:
            author["core"]=author_core[author['author id']]
            author["periphery"]=author_periphery[author['author id']]
            writer.writerow(author)

    print(f"Export completed. The authors are saved in {csv_file_path}")





def export_institutions_to_csv(json_file_path, fields_to_export: Optional[List[str]] = None, export_path=None):
    """
        Exports institutions information from a JSON file into a CSV file.

        The data to be exported is specified by the user through the `fields_to_export` parameter.
        Articles are categorized as either "core" or "periphery" based on their presence in the
        core articles set.

        Parameters:
        - json_file_path (str):
          The path to the JSON file that contains the institutions' data. The JSON should be structured
          such that it institutions article information

        - fields_to_export (Optional[List[str]]):
          A list of strings representing the fields to be included in the CSV export. If not specified,
          all available fields are exported. Available fields are:
            - institutions id
            - institutions display_name
            - institutions ror
            - institutions country_code,
            - institutions type

          If any fields in `fields_to_export` are not valid, a `ValueError` will be raised.

        - export_path (Optional[str]):
          The path where the CSV file will be saved. If not provided, the default path `"exports"`
          will be used. If a path is provided, the CSV file will be saved in that location.

        Returns:
        - None:
          This function does not return any value. It directly exports the articles data to a CSV file.

        Example usage:
        export_institutions_to_csv("path/to/institutions.json", fields_to_export=["id", "orcid", "display_name"], export_path="output/authors.csv")
        """

    institution_core=Counter()
    institution_periphery=Counter()

    field_mapping = lambda data : {
                "institutions id": data.get("id",""),
                "institutions display_name": data.get("display_name",""),
                "institutions ror": data.get("ror",""),
                "institutions country_code": data.get("country_code",""),
                "institutions type": data.get("type",""),
            }
    # Available fields for authors
    available_fields = list(field_mapping({}).keys())

    if fields_to_export is not None:
        invalid_fields = [field for field in fields_to_export if field not in available_fields]
        if invalid_fields:
            raise ValueError(
                f"The following fields are not available for exporting: {', '.join(invalid_fields)}. Available fields are : {', '.join(available_fields)}")
    else:
        fields_to_export = available_fields
    # Load the articles from the provided JSON file
    json_file_path = Path(json_file_path)  # Convert string to Path object
    if not json_file_path.is_file():
        raise ValueError(f"The input file does not exist: {json_file_path}")

    articles = json.load(open(json_file_path,"r"))

    # Prepare the output CSV file path
    if export_path is None:
        cache_key = Path(json_file_path).stem.replace(".json", "")
        csv_file_path = Path("exports") / f"institutions_{cache_key}.csv"
    else:
        csv_file_path = Path(export_path)

    # Ensure the directory exists
    csv_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Open the CSV file for writing
    already_done_institutions=set()
    institutions_to_save=[]
    core_articles = set(articles.keys())
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields_to_export+["core","periphery"])
        writer.writeheader()

        already_saved_institutions = set()

        # Loop over the articles
        for article_id, article_data in tqdm(articles.items(), desc="Exporting authors", unit="article"):
            if article_id not in already_done_institutions:

                for author in article_data.get('authorships', []):
                    author_id = author.get("author").get("id")
                    institutions = author.get('institutions', [])
                    for institution in institutions:
                        institution_id = institution.get("id")

                        # Skip authors that have already been saved
                        row = {field: field_mapping(institution)[field] for field in fields_to_export}
                        if article_id in core_articles:
                            institution_core[institution_id] += 1
                        else:
                            institution_periphery[institution_id] += 1
                        if institution_id not in already_saved_institutions:
                            institutions_to_save.append(row)

                            already_saved_institutions.add(institution_id)
                        already_done_institutions.add(institution_id)

            for citation_type in ['incoming_citations','outgoing_citations']:
                for cite in article_data[citation_type]:
                    article_id=cite.get("id")
                    if article_id not in already_done_institutions:
                        for author in cite.get('authorships', []):
                            author_id = author.get("author").get("id")
                            institutions = author.get('institutions', [])
                            for institution in institutions:
                                institution_id = institution.get("id")
                                # Skip authors that have already been saved
                                row = {field: field_mapping(institution)[field] for field in fields_to_export}
                                print(row)

                                if article_id in core_articles:
                                    institution_core[institution_id]+=1
                                else:
                                    institution_periphery[institution_id]+=1
                                if institution_id not in already_saved_institutions:
                                    institutions_to_save.append(row)
                                    already_saved_institutions.add(institution_id)
                                already_done_institutions.add(institution_id)

        for institution in institutions_to_save:
            institution["core"]=institution_core[institution['institutions id']]
            institution["periphery"]=institution_periphery[institution['institutions id']]
            writer.writerow(institution)

    print(f"Export completed. The institutions are saved in {csv_file_path}")



def create_citation_graph(articles: Dict[str, Dict], export_path: Optional[str] = None,periphery=True) -> nx.DiGraph:
    """
    Creates a directed graph from the articles data where nodes represent articles and edges represent citations.

    :param articles: A dictionary of articles where the key is the article ID and the value is the article details.
    :param export_path: Optional path to export the graph in GML format. If None, the graph will not be exported.

    :return: A NetworkX directed graph representing the citation network.

    """
    allowed_keys = ["id","doi","title","display_name","publication_year","primary_topic","authorships","language","type","cited_by_count","core"]
    G = nx.DiGraph()
    core_articles=set()
    for article_id in articles.keys():
        core_articles.add(article_id)

    def save_article_fields_for_gephi(article_data):
        node_data={}
        if "core" not in article_data:
            article_data["core"]=0
        for key, value in article_data.items():
            if key in ["title","primary_topic","language","type","cited_by_count","core"]:
                if value is None:
                    node_data[key] = ""
                elif key == "primary_topic":
                    node_data["primary_topic"]=value.get("display_name","")
                    if "display_name" in value:
                        node_data["primary_subfield"]=value["subfield"].get("display_name","")
                        node_data["primary_field"]=value["field"].get("display_name","")
                        node_data["primary_domain"]=value["domain"].get("display_name","")
                else:
                    node_data[key] = value

        return node_data

    for article_id, article_data in tqdm(articles.items(), desc="Creating citation graph", unit="article"):
        #  Remove unnecessary fields to reduce the graph's size
        _article_data = {key: value for key, value in article_data.items() if key in allowed_keys}

        G.add_node(article_id, **save_article_fields_for_gephi(_article_data),info=json.dumps(_article_data))
        for citing_article in article_data.get('incoming_citations', []):
            citing_article_id = citing_article['id']
            if not periphery and citing_article_id not in core_articles:
                continue
            if not G.has_node(citing_article_id):
                _citing_article_article_data = {key: value for key, value in citing_article.items() if key in allowed_keys}
                G.add_node(citing_article_id, **save_article_fields_for_gephi(_citing_article_article_data), info=json.dumps(_citing_article_article_data))
            G.add_edge(citing_article_id, article_id)


        for cited_article in article_data.get('outgoing_citations', []):
            cited_article_id = cited_article['id']
            if not periphery and cited_article_id not in core_articles:
                continue
            if not G.has_node(cited_article_id):
                _cited_article_article_data={key: value for key, value in cited_article.items() if key in allowed_keys}
                G.add_node(cited_article_id, **save_article_fields_for_gephi(_cited_article_article_data), info=json.dumps(_cited_article_article_data))
            G.add_edge(article_id, cited_article_id)

    print(f"Performing spring layout")
    pos = nx.spring_layout(G)
    for node, (x, y) in pos.items():
        G.nodes[node]['pos_x'] = str(x)
        G.nodes[node]['pos_y'] = str(y)


    if export_path:
        nx.write_gml(G, export_path)
        print(f"Export completed. The citation graph is saved in {export_path}")

    return G


def create_coauthorship_graph(articles: Dict[str, Dict], export_path: Optional[str] = None, periphery=True) -> nx.Graph:
    """
    Creates an undirected graph representing co-authorship relationships between authors, considering both core and peripheral articles.

    :param articles: A dictionary of articles where the key is the article ID and the value is the article details.
    :param export_path: Optional path to export the graph in GML format. If None, the graph will not be exported.
    :param periphery: If True, includes both core and peripheral articles in the co-authorship graph.
    :return: A NetworkX undirected graph representing the co-authorship network.
    """
    G = nx.Graph()
    core_articles=set()
    periphery_articles=set()

    for article_id in articles.keys():
        core_articles.add(article_id)


    def save_author_fields_for_gephi(article_data):
        node_data={}
        for key, value in article_data.items():
            if key in ["display_name","countries"]:
                if value is None:
                    node_data[key] = ""

                elif key == "countries":
                    node_data["countries"]=", ".join(value)

        return node_data
    # Iterate over each article and collect authors
    for article_id, article_data in tqdm(articles.items(), desc="Creating co-authorship graph", unit="article"):
        authorships = article_data.get("authorships", [])

        # Add authors as nodes to the graph (if not already present)
        authors=[]
        for author in authorships:
            author_id=author["author"]["id"]
            authors.append(author_id)
            if not G.has_node(author_id):
                author_info = author
                G.add_node(author_id, **save_author_fields_for_gephi(author_info),info=json.dumps(author_info))

        # Add edges (co-authorship between each pair of authors)
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                author_1 = authors[i]
                author_2 = authors[j]

                # Add an edge between the two authors (only if it does not already exist)
                if not G.has_edge(author_1, author_2):
                    G.add_edge(author_1, author_2)
                # Determine whether to include the article (core or peripheral)
        if periphery:
            # Consider incoming/outgoing citations (peripheral articles) to connect authors
            for citation_type in ['incoming_citations','outgoing_citations']:
                for citing_article in article_data.get(citation_type, []):
                    citing_article_id = citing_article['id']
                    if citing_article_id in core_articles:
                        continue #already done
                    if citing_article_id in periphery_articles:
                        continue #already done

                    periphery_articles.add(citing_article_id)

                    authorships = article_data.get("authorships", [])

                    # Add authors as nodes to the graph (if not already present)
                    authors = []
                    for author in authorships:
                        author_id = author["author"]["id"]
                        authors.append(author_id)
                        if not G.has_node(author_id):
                            author_info = author
                            G.add_node(author_id, **save_author_fields_for_gephi(author_info), info=json.dumps(author_info))


                    # Add edges (co-authorship between each pair of authors)
                    for i in range(len(authors)):
                        for j in range(i + 1, len(authors)):
                            author_1 = authors[i]
                            author_2 = authors[j]

                            # Add an edge between the two authors (only if it does not already exist)
                            if not G.has_edge(author_1, author_2):
                                G.add_edge(author_1, author_2)

    print(f"Performing spring layout")
    pos = nx.spring_layout(G)
    for node, (x, y) in pos.items():
        G.nodes[node]['pos_x'] = str(x)
        G.nodes[node]['pos_y'] = str(y)

    # Export the graph in GML format, if a path is provided
    if export_path:
        nx.write_gml(G, export_path)
        print(f"Export completed. The co-authorship graph is saved in {export_path}")


    return G

