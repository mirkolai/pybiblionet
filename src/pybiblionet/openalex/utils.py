from itertools import chain
from typing import Dict, List
from datetime import datetime


def check_date_format(dt_str: str) -> bool:
    """
    Checks if a given date string is in the 'YYYY-MM-DD' format.

    This function attempts to parse the input string using the datetime module to ensure it conforms
    to the 'YYYY-MM-DD' format. If the string is correctly formatted, the function returns True.
    Otherwise, it returns False.

    :param dt_str: A string representing a date in the 'YYYY-MM-DD' format.
    :return: True if the string is in the correct 'YYYY-MM-DD' format, otherwise False.

    Example:

    >>> check_date_format('20210714')
    False
    >>> check_date_format('2021-07-14')
    True
    >>> check_date_format('20211314')
    False
    """
    try:
        date = datetime.strptime(dt_str, '%Y-%m-%d')
    except:
        return False
    return True

def abstract_inverted_index_to_abstract(abstract_inverted_index: Dict[str, List[int]]) -> str:
    """
    Generates a string of words sorted by their positional indexes from an inverted index dictionary.

    This function takes an inverted index, where keys are words and values are lists of positions
    where these words occur in the text. It then reconstructs the original text by sorting the words
    based on their positions and concatenating them into a single string.

    :param abstract_inverted_index: A dictionary where keys are words (str) and values are lists of
                                    positions (List[int]) where the words occur in the text.
    :return: A string representing the original text, reconstructed by sorting the words based on
             their positions.

    Example:

    >>> abstract_inverted_index = {
    ... "Despite":[0], "growing":[1], "interest":[2], "in":[3,57,73,110,122],
    ... "Open":[4,201], "Access":[5], "(OA)":[6], "to":[7,54,252], "scholarly":[8,105],
    ... "literature,":[9], "there":[10], "is":[11,107,116,176], "an":[12,34,85,185,199,231],
    ... "unmet":[13], "need":[14,31], "for":[15,42,174,219], "large-scale,":[16],
    ... "up-to-date,":[17], "and":[18,24,77,112,124,144,221,237,256], "reproducible":[19],
    ... "studies":[20], "assessing":[21], "the":[22,104,134,145,170,195,206,213,245],
    ... "prevalence":[23], "characteristics":[25], "of":[26,51,75,83,103,137,141,163,209],
    ... "OA.":[27,168,239], "We":[28,46,97,203,240], "address":[29], "this":[30,114,142],
    ... "using":[32,95,244], "oaDOI,":[33], "open":[35], "online":[36], "service":[37],
    ... "that":[38,89,99,113,147,155], "determines":[39], "OA":[40,56,93,108,138,159,175,210,223,254],
    ... "status":[41], "67":[43], "million":[44], "articles.":[45], "use":[47], "three":[48,58],
    ... "samples,":[49], "each":[50], "100,000":[52], "articles,":[53,152,211], "investigate":[55],
    ... "populations:":[59], "(1)":[60], "all":[61], "journal":[62,70], "articles":[63,71,79,94,164,191,224],
    ... "assigned":[64], "a":[65,250], "Crossref":[66], "DOI,":[67], "(2)":[68], "recent":[69,128],
    ... "indexed":[72], "Web":[74], "Science,":[76], "(3)":[78], "viewed":[80], "by":[81,120,235],
    ... "users":[82,91,157], "Unpaywall,":[84], "open-source":[86], "browser":[87], "extension":[88],
    ... "lets":[90], "find":[92,154], "oaDOI.":[96], "estimate":[98], "at":[100], "least":[101], "28%":[102],
    ... "literature":[106], "(19M":[109], "total)":[111], "proportion":[115], "growing,":[117],
    ... "driven":[118,233], "particularly":[119], "growth":[121], "Gold":[123], "Hybrid.":[125],
    ... "The":[126], "most":[127,171], "year":[129], "analyzed":[130], "(2015)":[131], "also":[132,204],
    ... "has":[133], "highest":[135], "percentage":[136], "(45%).":[139], "Because":[140], "growth,":[143],
    ... "fact":[146], "readers":[148], "disproportionately":[149], "access":[150], "newer":[151],
    ... "we":[153,188], "Unpaywall":[156], "encounter":[158], "quite":[160], "frequently:":[161], "47%":[162],
    ... "they":[165], "view":[166], "are":[167], "Notably,":[169], "common":[172], "mechanism":[173],
    ... "not":[177], "Gold,":[178], "Green,":[179], "or":[180], "Hybrid":[181,238], "OA,":[182], "but":[183],
    ... "rather":[184], "under-discussed":[186], "category":[187], "dub":[189], "Bronze:":[190],
    ... "made":[192], "free-to-read":[193], "on":[194], "publisher":[196], "website,":[197], "without":[198],
    ... "explicit":[200], "license.":[202], "examine":[205], "citation":[207,216], "impact":[208],
    ... "corroborating":[212], "so-called":[214], "open-access":[215], "advantage:":[217], "accounting":[218],
    ... "age":[220], "discipline,":[222], "receive":[225], "18%":[226], "more":[227], "citations":[228],
    ... "than":[229], "average,":[230], "effect":[232], "primarily":[234], "Green":[236], "encourage":[241],
    ... "further":[242], "research":[243], "free":[246], "oaDOI":[247], "service,":[248], "as":[249],
    ... "way":[251], "inform":[253], "policy":[255], "practice.":[257]
    ... }
    >>> abstract = invert_index(abstract_inverted_index)
    >>> print(abstract)
    Despite growing interest in Open Access (OA) to scholarly literature, there is an unmet need for large-scale, up-to-date, and reproducible studies assessing the prevalence and characteristics of OA. We address this using oaDOI, an open online service that determines OA status for 67 million articles. We use three samples, each of 100,000 articles, to investigate three populations: (1) all journal articles assigned a Crossref DOI, (2) recent articles indexed in Web of Science, and (3) articles viewed by users of Unpaywall, an open-source browser extension that lets users find the OA status of articles. We estimate that at least 28% of the scholarly literature (19M articles) is OA. This proportion is growing, driven particularly by growth in Gold and Hybrid OA. The most recent year analyzed (2015) also has the highest percentage (45%). Because of this growth, and the fact that readers disproportionately access newer articles, we find that 47% of the articles they view are OA. Notably, the most common mechanism for OA is not Gold, Green, or Hybrid OA, but rather an under-discussed category we dub Bronze: articles made free-to-read on the publisher website, without an explicit license. We examine the citation impact of OA articles, corroborating the so-called open-access advantage: accounting for article age and discipline, OA articles receive 18% more citations than average, an effect driven primarily by Green OA. We encourage further research using the free oaDOI service, as a way to inform policy and practice.
    """
    words = [[(key, val) for val in val_list] for key, val_list in abstract_inverted_index.items()]
    abstract = list(chain(*words))
    abstract = sorted(abstract, key = lambda x: x[1])
    abstract = [ia[0] for ia in abstract]
    abstract = ' '.join(abstract)

    return abstract




if __name__ == "__main__":
    abstract_inverted_index = {"Despite":[0],"growing":[1],"interest":[2],"in":[3,57,73,110,122],"Open":[4,201],"Access":[5],"(OA)":[6],"to":[7,54,252],"scholarly":[8,105],"literature,":[9],"there":[10],"is":[11,107,116,176],"an":[12,34,85,185,199,231],"unmet":[13],"need":[14,31],"for":[15,42,174,219],"large-scale,":[16],"up-to-date,":[17],"and":[18,24,77,112,124,144,221,237,256],"reproducible":[19],"studies":[20],"assessing":[21],"the":[22,104,134,145,170,195,206,213,245],"prevalence":[23],"characteristics":[25],"of":[26,51,75,83,103,137,141,163,209],"OA.":[27,168,239],"We":[28,46,97,203,240],"address":[29],"this":[30,114,142],"using":[32,95,244],"oaDOI,":[33],"open":[35],"online":[36],"service":[37],"that":[38,89,99,113,147,155],"determines":[39],"OA":[40,56,93,108,138,159,175,210,223,254],"status":[41],"67":[43],"million":[44],"articles.":[45],"use":[47],"three":[48,58],"samples,":[49],"each":[50],"100,000":[52],"articles,":[53,152,211],"investigate":[55],"populations:":[59],"(1)":[60],"all":[61],"journal":[62,70],"articles":[63,71,79,94,164,191,224],"assigned":[64],"a":[65,250],"Crossref":[66],"DOI,":[67],"(2)":[68],"recent":[69,128],"indexed":[72],"Web":[74],"Science,":[76],"(3)":[78],"viewed":[80],"by":[81,120,235],"users":[82,91,157],"Unpaywall,":[84],"open-source":[86],"browser":[87],"extension":[88],"lets":[90],"find":[92,154],"oaDOI.":[96],"estimate":[98],"at":[100],"least":[101],"28%":[102],"literature":[106],"(19M":[109],"total)":[111],"proportion":[115],"growing,":[117],"driven":[118,233],"particularly":[119],"growth":[121],"Gold":[123],"Hybrid.":[125],"The":[126],"most":[127,171],"year":[129],"analyzed":[130],"(2015)":[131],"also":[132,204],"has":[133],"highest":[135],"percentage":[136],"(45%).":[139],"Because":[140],"growth,":[143],"fact":[146],"readers":[148],"disproportionately":[149],"access":[150],"newer":[151],"we":[153,188],"Unpaywall":[156],"encounter":[158],"quite":[160],"frequently:":[161],"47%":[162],"they":[165],"view":[166],"are":[167],"Notably,":[169],"common":[172],"mechanism":[173],"not":[177],"Gold,":[178],"Green,":[179],"or":[180],"Hybrid":[181,238],"OA,":[182],"but":[183],"rather":[184],"under-discussed":[186],"category":[187],"dub":[189],"Bronze:":[190],"made":[192],"free-to-read":[193],"on":[194],"publisher":[196],"website,":[197],"without":[198],"explicit":[200],"license.":[202],"examine":[205],"citation":[207,216],"impact":[208],"corroborating":[212],"so-called":[214],"open-access":[215],"advantage:":[217],"accounting":[218],"age":[220],"discipline,":[222],"receive":[225],"18%":[226],"more":[227],"citations":[228],"than":[229],"average,":[230],"effect":[232],"primarily":[234],"Green":[236],"encourage":[241],"further":[242],"research":[243],"free":[246],"oaDOI":[247],"service,":[248],"as":[249],"way":[251],"inform":[253],"policy":[255],"practice.":[257]}

    abstract= abstract_inverted_index_to_abstract(abstract_inverted_index)
    print(abstract)