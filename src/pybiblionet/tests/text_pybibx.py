
from pybibx.base import pbx_probe

from pybiblionet.openalex.core import export_articles_to_scopus

json_file_path = "query_results/query_result_1ea020320b230de5a973a39682eaa53dce89a9bb026b441a5f825232.json"
print(json_file_path)  # the articles are saved in a json file inside the query_result folder

export_articles_to_scopus(json_file_path,
                          export_path="scopus/myscopus.csv",
                          include_periphery=True
                          )

filename = "scopus/myscopus.csv"
bibfile = pbx_probe(file_bib=filename,db="scopus",del_duplicated=True)

report= bibfile.eda_bib()