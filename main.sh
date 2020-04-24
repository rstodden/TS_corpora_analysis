#! /bin/bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

sh ./download_corpora.sh

python -m spacy download en_core_web_sm
python -m spacy download it_core_news_sm
python -m spacy download de_core_news_sm
python -m spacy download es_core_news_sm

# preprocess NEWSELA
python preprocess_newsela.py -f "data/EN-Newsela_2016/newsela_article_corpus_2016-01-29/articles/" -o "data/EN-Newsela_2016/newsela_article_corpus_2016-01-29/split/" -m "en_core_web_sm"
mkdir -p "data/ES-Newsela/articles/split/"
python preprocess_newsela.py -f "data/ES-Newsela/articles/" -o "data/ES-Newsela/split/" -m "es_core_news_sm"


## todo: add an iput to decide which languages are processed. should affect alignment, preprocess, and download.
python align_sentences.py -f "data/DE-Klaper/" -o "data/ALL/" -m "de_core_news_sm"
python align_sentences.py -f "data/ES-Newsela/split/" -o "data/ALL/" -m "es_core_news_sm"
python align_sentences.py -f "data/CS-COSTRA/costra_1.0./round_2/" -o "data/ALL/" -m "cs"
python align_sentences.py -f "data/EN-TurkCorpus/truecased/" -o "data/ALL/" -m "en_core_web_sm"
python align_sentences.py -f "data/EN-Newsela_2015/newsela_data_share-20150302/" -o "data/ALL/" -m "en_core_web_sm"
python align_sentences.py -f "data/EN-Newsela_2016/newsela_article_corpus_2016-01-29/split/" -o "data/ALL/" -m "en_core_web_sm"
python align_sentences.py -f "data/IT-PaCCSS/" -o "data/ALL/" -m "it_core_news_sm"
python align_sentences.py -f "data/EN-QATS/" -o "data/ALL/" -m "en_core_web_sm"
## # python align_sentences.py -f "data/FR-CLEAR_wiki/" -o "data/ALL/" -m "fr_core_news_sm"
#
python do_statistics.py

deactivate