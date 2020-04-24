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
* Either run `source ./main.sh` to download, preprocess and align all data,
* or run `python align_sentences.py -f <filespath> -o <outputpath> -m <spacymodel>' -s <use stanza>` to align only one corpus


## References
If you use this code, please cite R. Stodden, and L. Kallmeyer (2020, to appear). A multi-lingual and cross-domain analysis of features for text simplification. In Proceedings of the Workshop on Tools and Resources to Empower People with REAding DIfficulties (READI), Marseille, France.

## License
The code of the analysis is licensed under the [MIT license](license.md). If use this implementation please cite our paper.

## Files