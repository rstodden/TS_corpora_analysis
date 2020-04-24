# download corpora
# unzip corpora
# delete zipped file

# language abbreviation following ISO 639-1 codes
mkdir -p data
mkdir -p data/CS-COSTRA
wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3123/costra_1.0.zip
unzip costra_1.0.zip -d data/CS-COSTRA
rm costra_1.0.zip


## WikiLarge/WikiSmall/ DRESS
#mkdir -p data/EN-WikiLarge
#wget https://github.com/louismartin/dress-data/raw/master/data-simplification.tar.bz2
#tar -xvjf data-simplification.tar.bz2 -C data/EN-WikiLarge
#rm data-simplification.tar.bz2

mkdir -p data/EN-TurkCorpus
svn checkout https://github.com/cocoxu/simplification/trunk/data/turkcorpus data/EN-TurkCorpus

mkdir -p data/EN-QATS
wget http://qats2016.github.io/qats2016.github.io/train.shared-task.tsv
mv train.shared-task.tsv data/EN-QATS/train.shared-task.tsv

#mkdir -p data/IT-simpitiki
#wget https://github.com/dhfbk/simpitiki/raw/master/corpus/simpitiki-v2.xml
#mv simpitiki-v2.xml data/IT-simpitiki/simpitiki-v2.xml

#mkdir -p data/SV-COCTAILL
#wget http://spraakbanken.gu.se/lb/resurser/meningsmangder/coctaill.xml.bz2
#bzip2 -dk coctaill.xml.bz2
#mv coctaill.xml data/SV-COCTAILL/coctaill.xml
#rm coctaill.xml.bz2


mkdir -p data/DE-Klaper
## insert data manually

mkdir -p data/EN-Newsela_2016
## insert complete newsela corpus
mkdir -p data/EN-Newsela_2015/
unzip data/EN-Newsela_2016/newsela_article_corpus_2016-01-29/newsela_data_share-20150302.zip -d data/EN-Newsela_2015
mkdir -p data/ES-Newsela
mkdir -p data/ES-Newsela/articles
find data/EN-Newsela_2016/newsela_article_corpus_2016-01-29/articles/ -name '*spanish.es.*.txt' -exec mv -i {} data/ES-Newsela/articles/ \;


mkdir -p data/IT-PaCCSS
## insert data manually after request
## http://www.italianlp.it/resources/paccss-it-parallel-corpus-of-complex-simple-sentences-for-italian/

