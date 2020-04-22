import os
import spacy
import sys, getopt

def main(argv):
	file_path = "massalign/EN-Newsela/newsela_article_corpus_2016-01-29/articles/"
	output_path = file_path+"split/"
	model_name = "en_core_web_sm"
	try:
		opts, args = getopt.getopt(argv,"f:m:o:",["filespath=","outputpath=", "spacymodel="])
	except getopt.GetoptError:
		print('preprocess_newsela.py -f <filespath> -o <outputpath> -m <spacymodel>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('preprocess_newsela.py -f <filespath> -o <outputpath> -m <spacymodel>')
			sys.exit()
		elif opt in ("-f", "--filespath"):
			file_path = arg
		elif opt in ("-m", "--spacymodel"):
			model_name = arg
		elif opt in ("-o", "--outputpath"):
			output_path = arg
		else:
			sys.exit(2)
	print(file_path, output_path, model_name)

	spacy_model = spacy.load(model_name, disable=["tagger", "parser", "ner", "textcat"])
	spacy_model.add_pipe(spacy_model.create_pipe('sentencizer'))
	list_files = [file for file in os.listdir(file_path) if os.path.isfile(file_path+file)]

	if not os.path.exists(output_path):
		os.mkdir(output_path)
	for i, file in enumerate(list_files):
		with open(file_path+file, "r", encoding="utf-8") as f:
			content = f.readlines()
		new_content = list()
		for paragraph in content:
			if paragraph != "\n":
				par = spacy_model(paragraph.strip())
				for sent in par.sents:
					new_content.append(str(sent).strip())
				new_content.append("")
		with open(output_path+file, "w+") as f:
			f.write("\n".join(new_content))
		if not i%100:
			print(i)


if __name__ == "__main__":
	main(sys.argv[1:])
