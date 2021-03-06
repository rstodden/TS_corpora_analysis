# A Multi-lingual and Cross-domain Analysis of Features for Text Simplification
This repository contains the implementation of the analysis and evaluation methods representend in Stodden et al. (2020). 
In this paper, we investigate the relevance of text simplification an text readability features for Czech, German, 
English, Spanish, and Italian text simplification corpora. Our multi-lingual and multi-domain corpus analysis shows 
that the relevance of different features for text simplification is different per corpora, language, and domain.

## Getting started

### Dependencies
* Python 3

### Installing
* `pip install -r requirements.txt` (or install them in a virtual enviroment during the main run)
* git clone https://github.com/rstodden/text-simplification-evaluation.git
* cd text-simplification-evaluation
* request data for Newsela, It-PaCCs, DE-Klaper and paste them in the directory "data". The other corpora will be automatically downloaded.
* Either run `source ./main.sh` to download, preprocess, align all data, and calculate the statistics
* or run `python align_sentences.py -f <filespath> -o <outputpath> -m <spacymodel>' -s <use stanza>` to align only one corpus
* or run `python do_statistics.py` to run only the statistisc


## Filestructure
* code files 
* *data/*:
    * one directory per plain corpus files
    * *data/ALL/*: tsv file per corpus with aligned data and feature values
    * *data/results/*: files containing the statistics
        * *data/results/all*: statistics per corpus (Research Question 1)
        * *data/resulst/cross-domain*: effects across domains (RQ2 - across)
        * *data/results/cross-lingual*: effects across languages (RQ3 - across)
        * *data/results/news*: stacked news results and results per each news corpus
        * *data/resulst/web*: stacked web results and results per each web corpus
        * *data/results/wiki*: stacked wiki results and results per each wiki corpus
        * *data/resulst/EN*: stacked results for all EN corpora and results per each EN corpus
        * *data/results/stacked-corpora*: results for all corpora stacked to one large corpus
        * *all_\[effect|descr\](_paired)?_results_all.csv* (RQ1)
        * *all_\[effect|descr\](_paired)?_results_\[web|wiki|news\].txt* : effects within each domain (RQ2 - within)
        * *all_effect_paired_resultsEN.txt*: effects within EN (RQ3 - within)
        * *.\*effect.\*.csv*: comma separated effect size and significance level of all features (line) per corpus (column)
        * *.\*descr.\*.csv*: comma separated count, average, and standard deviation of all features (line) per corpus (column)
        * *.\*effect.\*.txt*: LaTeX table with effect size and significance level of all features (line) per corpus (column)
        * *.\*descr.\*.txt*: LaTeX table with count, average, and standard deviation of all features (line) per corpus (column)
        * *.\*sent.\*.txt*: reported result sentences, one effect per line
        

## References
If you use this code, please cite R. Stodden, and L. Kallmeyer (2020). A multi-lingual and cross-domain analysis of features for text simplification. In Proceedings of the Workshop on Tools and Resources to Empower People with REAding DIfficulties (READI), Marseille, France. URL: https://www.aclweb.org/anthology/2020.readi-1.12.pdf
```bibtex
@inproceedings{stodden-kallmeyer-2020-multi,
    title = "A multi-lingual and cross-domain analysis of features for text simplification",
    author = "Stodden, Regina  and
      Kallmeyer, Laura",
    booktitle = "Proceedings of the 1st Workshop on Tools and Resources to Empower People with REAding DIfficulties (READI)",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://www.aclweb.org/anthology/2020.readi-1.12",
    pages = "77--84",
    abstract = "In text simplification and readability research, several features have been proposed to estimate or simplify a complex text, e.g., readability scores, sentence length, or proportion of POS tags. These features are however mainly developed for English. In this paper, we investigate their relevance for Czech, German, English, Spanish, and Italian text simplification corpora. Our multi-lingual and multi-domain corpus analysis shows that the relevance of different features for text simplification is different per corpora, language, and domain. For example, the relevance of the lexical complexity is different across all languages, the BLEU score across all domains, and 14 features within the web domain corpora. Overall, the negative statistical tests regarding the other features across and within domains and languages lead to the assumption that text simplification models may be transferable between different domains or different languages.",
    language = "English",
    ISBN = "979-10-95546-45-0",
}
```

## License
The code of the analysis is licensed under the [MIT license](license.md). If you use this implementation please cite our paper.

