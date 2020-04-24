import os
import glob
# from spacy_conll import Spacy2ConllParser
import spacy
import nltk
import spacy_udpipe
import spacy
import stanza
from spacy_stanza import StanzaLanguage
import os
import pathlib
import massalign
from massalign import annotators
from massalign import models
from massalign import core
from massalign import aligners
import sys, getopt
import tseval.feature_extraction
# from tseval.feature_extraction import get_all_vectorizers, get_sentence_feature_extractors, get_sentence_pair_feature_extractors, get_average, wrap_single_sentence_vectorizer, get_wordrank_score

import warnings
warnings.filterwarnings("ignore")


def align_sentences(alignment_id, complex_id, simple_id, complex_level="0", simple_level="1", complex_text="", simple_text="", include_text=True):
	if include_text:
		return [alignment_id, complex_id, complex_level, complex_text.lstrip().rstrip(), simple_id, simple_level, simple_text.lstrip().rstrip()]
	else:
		return [alignment_id, complex_id, complex_level, simple_id, simple_level, "\n"]


def is_number(s):
	## based on Bram Vanroy, https://github.com/BramVanroy/spacy_conll
	try:
		float(s)
		return True
	except ValueError:
		return False


def get_morphology(model, tag):
	## based on Bram Vanroy, https://github.com/BramVanroy/spacy_conll
	if not model.vocab.morphology.tag_map or tag not in model.vocab.morphology.tag_map:
		return '_'
	else:
		feats = [f'{prop}={val}' for prop, val in model.vocab.morphology.tag_map[tag].items() if not is_number(prop)]
		if feats:
			return '|'.join(feats)
		else:
			return '_'


def add_conllu_per_sentence(spacy_model, text_string, identifier):
	## partially based on Bram Vanroy, https://github.com/BramVanroy/spacy_conll and Raquel G. Alhama, https://github.com/rgalhama/spaCy2CoNLLU
	text = spacy_model(text_string.lstrip().rstrip())
	conllu_text = ["# sent_id = " + identifier, "# text = " + text_string.lstrip().rstrip()]
	for token in text:
		if token.dep_.lower().strip() == 'root':
			head_idx = 0
		else:
			head_idx = token.head.i + 1 - text[0].i
		if token.text == token.text_with_ws and token.i + 1 != len(text):
			no_space = "SpaceAfter=No"
		else:
			no_space = "_"
		morph = get_morphology(spacy_model, token.tag_)

		conllu_text.append('\t'.join(
			[str(token.i+1), token.text, token.lemma_, token.pos_, token.tag_, morph, str(head_idx), token.dep_, "_", no_space]))
	conllu_text.append("")
	return conllu_text


def get_source_id_dict(path_data):
	list_articles = {}
	for article in glob.glob(path_data + "/*.txt"):
		sep_article = article.split('/')[-1][:-7].split('_')
		list_articles.setdefault('_'.join(sep_article[0:-1]), set()).add(sep_article[-1])
	# print(list_articles)
	return list_articles


def get_feature_values(spacy_model, complex_sent, simple_sent, lang="en", only_names=False):
	feature_list = list()
	all_sentence_functions = tseval.feature_extraction.get_sentence_simplification_feature_extractors()   + tseval.feature_extraction.get_sentence_feature_extractors()
	all_pair_functions = tseval.feature_extraction.get_sentence_pair_simplification_feature_extractors()   + tseval.feature_extraction.get_sentence_pair_feature_extractors()
	if only_names:
		names_list = list()
		for func in all_sentence_functions:
			# print(func.__name__)
			names_list.append(func.__name__+"_complex")
			names_list.append(func.__name__+"_simple")
		for func in all_pair_functions:
			names_list.append(func.__name__+"_paired")
		return names_list

	complex_sent, simple_sent = spacy_model(complex_sent.lstrip().rstrip()), spacy_model(simple_sent.lstrip().rstrip())
	for func in all_sentence_functions:
		feature_list.append(str(func(complex_sent, lang)))
		feature_list.append(str(func(simple_sent, lang)))
	for func in all_pair_functions:
		# print(func.__name__)
		feature_list.append(str(func(complex_sent, simple_sent, lang)))
	# print(feature_list, )
	return feature_list


def save_data(lang, corpus, aligned_sentences, conllu_text, data_path="data/ALL/", complex_level="", simple_level="", sep="\t"):
	headline = ["alignment_id", "complex_id", "complex_level", "complex_text", "simple_id", "simple_level", "simple_text"]
	headline.extend(get_feature_values("", "", "", only_names=True))
	# print(get_feature_values("", "", "", only_names=True))
	aligned_sentences = [headline] + aligned_sentences
	print(aligned_sentences[0])
	if sep == "\t":
		ending = ".tsv"
	elif sep == "," or sep == ";":
		ending = ".csv"
	else:
		ending = ".txt"
	if not os.path.exists(data_path):
		os.makedirs(data_path)
	if not os.path.exists(data_path):
		os.makedirs(data_path)
	with open(data_path + lang + '-' + corpus + complex_level + simple_level + ending, "w+") as f:
		for line in aligned_sentences:
			f.write(sep.join(line) + "\n")
	if conllu_text:
		with open(data_path + lang + '-' + corpus + complex_level + simple_level + '.conllu', "w+") as f:
			conllu_text = ["# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC"] + conllu_text
			f.write('\n'.join(conllu_text))
	return 1


def get_ids_and_conllu_text(complex_identifier, simple_identifier, list_unique_identifier, spacy_model, complex_data, simple_data, use_conllu=False):
	new_conllu, new_ids = list(), list()
	if use_conllu:
		if complex_identifier not in list_unique_identifier:
			new_conllu.extend(add_conllu_per_sentence(spacy_model, complex_data, complex_identifier))
			new_ids.append(complex_identifier)
		if simple_identifier not in list_unique_identifier:
			new_conllu.extend(add_conllu_per_sentence(spacy_model, simple_data, simple_identifier))
			new_ids.append(simple_identifier)
	return new_conllu, new_ids


def get_ids(lang, corpus, article_id, par_id_complex, par_id_simple, sent_id_complex, sent_id_simple, complex_level, simple_level):
	alignment_id = lang + "-" + corpus + "-" + article_id + "-" + par_id_complex+"_"+par_id_simple + "-" + sent_id_complex + "_" + sent_id_simple + "-" + complex_level + "_" + simple_level
	complex_identifier = lang + "-" + corpus + "-" + article_id + "-" + par_id_complex + "-" + sent_id_complex + "-" + complex_level
	simple_identifier = lang + "-" + corpus + "-" + article_id + "-" + par_id_simple + "-" + sent_id_simple + "-" + simple_level
	#source.split("_")[1] + "-"
	return alignment_id, complex_identifier, simple_identifier


def align_german_klaper(path_data, spacy_model, conllu_path, include_text=True, use_conllu=False, sep="\t", add_features=False):
	path_alignments = path_data + "alignments/"
	path_data = path_data + "tokenized/"
	lang, corpus = path_data.split('/')[1].split("-")
	lang = lang.lower()
	list_articles = get_source_id_dict(path_data)
	aligned_sentences, list_unique_identifier, conllu_text = list(), list(), list()
	complex_level, simple_level = "0", "1"
	for source in list_articles.keys():
		for i, article_id in enumerate(list_articles[source]):
			sorted_pairs = list()
			if os.path.isfile(path_alignments + source + "_" + article_id + "_AS-LS.txt"):
				try:
					with open(path_alignments + source + "_" + article_id + "_AS-LS.txt", encoding="utf-8") as f:
						alignment = f.readlines()
					with open(path_data + source + "_" + article_id + "_AS.txt", encoding="utf-8") as f:
						complex_data = f.readlines()
					with open(path_data + source + "_" + article_id + "_LS.txt", encoding="utf-8") as f:
						simple_data = f.readlines()
				except UnicodeDecodeError:
					continue
				# presort alignments so that N-1 and 1-N
				for pair in alignment:
					pair_added = False
					complex_id, simple_id = pair.strip().split('\t')

					for s, sort_pair in enumerate(sorted_pairs):
						if complex_id == sort_pair[0][0]:
							sorted_pairs[s][1].append(simple_id)
							pair_added = True
						elif simple_id == sort_pair[1][0]:
							sorted_pairs[s][0].append(complex_id)
							pair_added = True
					if not pair_added:
						sorted_pairs.append([[complex_id], [simple_id]])

				for pair in sorted_pairs:
					# if complex or simple number occur more than once , join the sentences for analysis
					complex_id, simple_id = "|".join(pair[0]), "|".join(pair[1])
					alignment_id, complex_identifier, simple_identifier = get_ids(lang, corpus, source+"_"+article_id, "x", "x", complex_id, simple_id, complex_level, simple_level)
					complex_text, simple_text = list(), list()
					for e, line in enumerate(complex_data):
						for c in pair[0]:
							if e == (int(c)-1):
								complex_text.append(complex_data[e].strip())
					for e, line in enumerate(simple_data):
						for s in pair[1]:
							if e == (int(s)-+1):
								simple_text.append(simple_data[e].strip())
					complex_text = " ".join(complex_text)
					simple_text = " ".join(simple_text)

					aligned_values = align_sentences(alignment_id, complex_identifier, simple_identifier, complex_level, simple_level, complex_text, simple_text, include_text=include_text)
					if add_features:
						aligned_values.extend(get_feature_values(spacy_model, complex_text, simple_text, lang))
					aligned_sentences.append(aligned_values)
					new_conllu, new_ids = get_ids_and_conllu_text(complex_identifier, simple_identifier,
																  list_unique_identifier, spacy_model, complex_text,
																  simple_text, use_conllu=use_conllu)
					list_unique_identifier.extend(new_ids)
					conllu_text.extend(new_conllu)

	return save_data(lang, corpus, aligned_sentences, conllu_text, conllu_path, sep=sep)


def align_turk_corpus_cased(path_data, spacy_model, output_path, sep="\t", use_conllu=False, include_text=True, add_features=False):
	lang, corpus = path_data.split('/')[1].split("-")
	lang = lang.lower()
	aligned_sentences, list_unique_identifier, conllu_text = list(), list(), list()

	for file in os.listdir(path_data):
		with open(path_data + file, "r", encoding="utf-8") as f:
			data = f.readlines()
		complex_level, simple_level = "0", "1"
		for i, line in enumerate(data):
			line_split = line.split("\t")
			for t, turker_data in enumerate(line_split[2:]):
				alignment_id, complex_identifier, simple_identifier = get_ids(lang, corpus, str(i), "x", "turker_"+str(t), str(i), str(i), complex_level, simple_level)
				aligned_values = align_sentences(alignment_id, complex_identifier, simple_identifier,
														 complex_level, simple_level, line_split[1], turker_data, include_text=include_text)
				if add_features:
					aligned_values.extend(get_feature_values(spacy_model, line_split[1], turker_data, lang))

				aligned_sentences.append(aligned_values)
				new_conllu, new_ids = get_ids_and_conllu_text(complex_identifier, simple_identifier, list_unique_identifier, spacy_model, line_split[1], turker_data, use_conllu=use_conllu)
				list_unique_identifier.extend(new_ids)
				conllu_text.extend(new_conllu)

	return save_data(lang, corpus, aligned_sentences, conllu_text, output_path, sep=sep)


def align_paccss_corpus(path_data, spacy_model, conllu_path, sep="\t", use_conllu=False, include_text=True, add_features=True):
	lang, corpus = path_data.split('/')[1].split("-")
	lang = lang.lower()
	aligned_sentences, list_unique_identifier, conllu_text = list(), list(), list()
	article_id = "x"
	complex_level, simple_level = "0", "1"
	with open(path_data + "CSAS-IT.txt", encoding="utf-8") as f:
		data = f.readlines()
	for i, line in enumerate(data[1:]):
		line_split = line.split("\t")
		alignment_id, complex_identifier, simple_identifier = get_ids(lang, corpus, article_id, "x", "x", str(i), str(i), complex_level, simple_level)
		# aligned_sentences.append(align_sentences(alignment_id, complex_identifier, simple_identifier, complex_level, simple_level, line_split[0], line_split[1],include_text=include_text))
		# conllu_text, list_unique_identifier = get_ids_and_conllu_text(complex_identifier, simple_identifier, list_unique_identifier, spacy_model, line_split[0], line_split[1], use_conllu=use_conllu)
		aligned_values = align_sentences(alignment_id, complex_identifier, simple_identifier,
										 complex_level, simple_level, line_split[0], line_split[1], include_text=include_text)
		if add_features:
			aligned_values.extend(get_feature_values(spacy_model, line_split[0], line_split[1], lang))

		aligned_sentences.append(aligned_values)
		new_conllu, new_ids = get_ids_and_conllu_text(complex_identifier, simple_identifier, list_unique_identifier,
													  spacy_model, line_split[0], line_split[1], use_conllu=use_conllu)
		list_unique_identifier.extend(new_ids)
		conllu_text.extend(new_conllu)

	save_data(lang, corpus, aligned_sentences, conllu_text, conllu_path, sep=sep)
	return 1


def align_czech_corpus(path_data, spacy_model, conllu_path, sep="\t", include_text=True, use_conllu=False, add_features=True):
	lang, corpus = path_data.split('/')[1].split("-")
	lang = lang.lower()
	aligned_sentences, list_unique_identifier, conllu_text = list(), list(), list()
	# article_id = "x"
	complex_level, simple_level = "0", "1"
	with open(path_data + "annotations.tsv", encoding="utf-8") as f:
		changed_data = f.readlines()
	with open(path_data + "source_sentences.tsv", encoding="utf-8") as f:
		source_data = f.readlines()
	dict_source_data = dict()
	for line in source_data:
		line_split = line.split(sep)
		dict_source_data[line_split[0]] = line_split[1]
	simplifications = dict()
	for line in changed_data:
		line_split = line.split(sep)
		if line_split[2] == "simple sentence":
			simplifications[line_split[0] + "_" + line_split[1]] = line_split[3]
	for key in simplifications.keys():
		id, annotator = key.split("_")
		alignment_id, complex_identifier, simple_identifier = get_ids(lang, corpus, "x", "x", annotator, id, id, complex_level,	simple_level)
		aligned_values = align_sentences(alignment_id, complex_identifier, simple_identifier,
										 complex_level, simple_level, dict_source_data[id], simplifications[key],
										 include_text=include_text)
		if add_features:
			aligned_values.extend(get_feature_values(spacy_model, dict_source_data[id], simplifications[key], lang))

		aligned_sentences.append(aligned_values)
		new_conllu, new_ids = get_ids_and_conllu_text(complex_identifier, simple_identifier, list_unique_identifier,
													  spacy_model, dict_source_data[id], simplifications[key], use_conllu=use_conllu)
		list_unique_identifier.extend(new_ids)
		conllu_text.extend(new_conllu)
	save_data(lang, corpus, aligned_sentences, conllu_text, conllu_path, sep=sep)
	return 1


def align_newsela_corpus(path_data, spacy_model, conllu_path, sep="\t", complex_level="0", simple_level="5", use_conllu=False, include_text=True, add_features=False):
	lang, corpus = path_data.split('/')[1].split("-")
	lang = lang.lower()
	aligned_sentences, list_unique_identifier, conllu_text = list(), list(), list()
	with open("stopwords.txt", "w+", encoding="utf-8") as f:
		f.writelines(spacy_model.Defaults.stop_words)

	stopword_path = "stopwords.txt"

	list_pairs_sent, list_pairs_pars, alignmentsl = list(), list(), list()
	file_list = sorted(list(set([file[:-9] for file in os.listdir(path_data)])))
	# if file.endswith("."+lang_ending+"."+complex_level+".txt") or file.endswith("."+lang_ending+"."+simple_level+".txt")])))
	n_n2n = 0
	#for complex_level in range(0, 5):
	#	for simple_level in range(1, 6):
	for complex_level, simple_level in [(0,1), (1,2), (2,3), (3, 4), (4,5)]:
			aligned_sentences, conllu_text = list(), list()
			if complex_level < simple_level:
				n_docs = 0
				n_pars = 0
				n_sents = 0
				for i, file in enumerate(file_list):
					if not os.path.isfile(path_data + file + "." + lang.lower() + "." + str(
							complex_level) + ".txt") or not os.path.isfile(
						path_data + file + "." + lang.lower() + "." + str(simple_level) + ".txt"):
						continue
					n_docs += 1
					# Train model over them:
					model = massalign.models.TFIDFModel([path_data + file + "." + lang.lower() + "." + str(complex_level) + ".txt", path_data + file + "." + lang.lower() + "." + str(simple_level) + ".txt"], stopword_path)

					# Get MASSA aligner for convenience:
					m = massalign.core.MASSAligner()

					# Get paragraph aligner:
					paragraph_aligner = massalign.aligners.VicinityDrivenParagraphAligner(similarity_model=model, acceptable_similarity=0.2)
					# Get sentence aligner:
					sentence_aligner = massalign.aligners.VicinityDrivenSentenceAligner(similarity_model=model, acceptable_similarity=0.2, similarity_slack=0.05)
					# get paragraphs per document (no sentence split
					paragraphs_file1 = m.getParagraphsFromDocument(path_data + file + "." + lang.lower() + "." + str(complex_level) + ".txt")
					paragraphs_file2 = m.getParagraphsFromDocument(path_data + file + "." + lang.lower() + "." + str(simple_level) + ".txt")

					# Align paragraphs without sentence split:
					alignments, aligned_paragraphs = m.getParagraphAlignments(paragraphs_file1, paragraphs_file2, paragraph_aligner)
					n_pars += len(aligned_paragraphs)

					# Align sentences in each pair of aligned paragraphs:
					for n, aligned_pair in enumerate(aligned_paragraphs):
						par_id_complex = "_".join([str(par_id) for par_id in alignments[n][0]])
						par_id_simple = "_".join([str(par_id) for par_id in alignments[n][1]])
						aligned_par_file1 = aligned_pair[0]
						aligned_par_file2 = aligned_pair[1]
						alignments_sentence, aligned_sents = m.getSentenceAlignments(aligned_par_file1,
																						 aligned_par_file2,
																						 sentence_aligner)
						for a, aligned_sent_pair in enumerate(aligned_sents):
							sent_id_simple, sent_id_complex, simple_sent, complex_sent = list(), list(), list(), list()
							for s_complex, sent_complex in enumerate(aligned_sent_pair[0]):
								sent_id_complex.append(str(alignments_sentence[a][0][s_complex]))
								complex_sent.append(aligned_sent_pair[0][s_complex].strip())
							for s_simple, sent_simple in enumerate(aligned_sent_pair[1]):
								sent_id_simple.append(str(alignments_sentence[a][1][s_simple]))
								simple_sent.append(aligned_sent_pair[1][s_simple].strip())
							sent_id_simple = "|".join(sent_id_simple)
							sent_id_complex = "|".join(sent_id_complex)
							complex_sent = " ".join(complex_sent)
							simple_sent = " ".join(simple_sent)
							alignment_id, complex_identifier, simple_identifier = get_ids(lang, corpus, str(i),
																						  par_id_complex, par_id_simple,
																						  sent_id_complex,
																						  sent_id_simple,
																						  str(complex_level),
																						  str(simple_level))
							aligned_values = align_sentences(alignment_id, complex_identifier, simple_identifier,
															 str(complex_level),
															 str(simple_level), complex_sent, simple_sent,
															 include_text=include_text)
							if add_features:
								aligned_values.extend(
									get_feature_values(spacy_model, complex_sent, simple_sent, lang))
							aligned_sentences.append(aligned_values)
							new_conllu, new_ids = get_ids_and_conllu_text(complex_identifier, simple_identifier,
																		  list_unique_identifier, spacy_model,
																		  complex_sent, simple_sent,
																		  use_conllu=use_conllu)
							list_unique_identifier.extend(new_ids)
							conllu_text.extend(new_conllu)
							n_sents += 1

				print(str(complex_level), str(simple_level), "num docs", n_docs, "num pars", n_pars, "num", n_sents)
				save_data(lang, corpus, aligned_sentences, conllu_text, conllu_path, str(complex_level),
						  str(simple_level))
	print("number of n2n", n_n2n)
	return 1


def align_newsela_15_corpus(path_data, spacy_model, conllu_path, sep="\t", include_text=True, use_conllu=False, add_features=False):
	lang, corpus = path_data.split('/')[1].split("-")
	lang = lang.lower()
	aligned_sentences, list_unique_identifier, conllu_text = list(), list(), list()
	with open(path_data + "newsela_articles_20150302.aligned.sents.txt", encoding="utf-8") as f:
		data = f.readlines()
	for i, line in enumerate(data):
		line_split = line.split("\t")
		article_id = line_split[0]
		complex_level = line_split[1][1]
		simple_level = line_split[2][1]
		alignment_id, complex_identifier, simple_identifier = get_ids(lang, corpus, article_id, "x", "x", str(i), str(i), complex_level, simple_level)
		aligned_values = align_sentences(alignment_id, complex_identifier, simple_identifier, complex_level, simple_level, line_split[3], line_split[4], include_text=include_text)
		if add_features:
			aligned_values.extend(
				get_feature_values(spacy_model, line_split[3], line_split[4], lang))
		aligned_sentences.append(aligned_values)
		new_conllu, new_ids = get_ids_and_conllu_text(complex_identifier, simple_identifier, list_unique_identifier,
													  spacy_model, line_split[3], line_split[4], use_conllu=use_conllu)
		list_unique_identifier.extend(new_ids)
		conllu_text.extend(new_conllu)
	save_data(lang, corpus, aligned_sentences, conllu_text, conllu_path)
	return 1


def align_qats(path_data, spacy_model, conllu_path, include_text=True, add_features=True, use_conllu=False, use_assessment=False, sep="\t"):
	lang, corpus = path_data.split('/')[1].split("-")
	lang = lang.lower()
	aligned_sentences, list_unique_identifier, conllu_text = list(), list(), list()
	complex_level, simple_level = "0", "1"

	with open(path_data + "train.shared-task.tsv", encoding="utf-8") as f:
		data = f.readlines()
	for i, line in enumerate(data[1:]):
		line_split = line.strip().split(sep)
		alignment_id, complex_identifier, simple_identifier = get_ids(lang, corpus, "x", "x", "x", str(i), str(i),
																	  complex_level, simple_level)
		aligned_values = align_sentences(alignment_id, complex_identifier, simple_identifier,
										 complex_level, simple_level, line_split[0], line_split[1],
										 include_text=include_text)
		if add_features:
			aligned_values.extend(get_feature_values(spacy_model, line_split[0], line_split[1], lang))

		aligned_sentences.append(aligned_values)
		new_conllu, new_ids = get_ids_and_conllu_text(complex_identifier, simple_identifier, list_unique_identifier,
													  spacy_model, line_split[0], line_split[1],
													  use_conllu=use_conllu)
		list_unique_identifier.extend(new_ids)
		conllu_text.extend(new_conllu)
	save_data(lang, corpus, aligned_sentences, conllu_text, conllu_path, sep=sep)
	return 1


def align_clear(path_data, spacy_model, conllu_path, include_text=True, add_features=True, sep="\t", use_conllu=False):
	lang, corpus = path_data.split('/')[1].split("-")
	lang = lang.lower()
	aligned_sentences, list_unique_identifier, conllu_text = list(), list(), list()
	complex_level, simple_level = "wiki", "viki"  # wiki=0, viki=1

	with open("stopwords.txt", "w+", encoding="utf-8") as f:
		f.writelines(spacy_model.Defaults.stop_words)
	stopword_path = "stopwords.txt"

	file_list = sorted(list(set([file[:-9] for file in os.listdir(path_data)])))
	n_n2n = 0
	for i, file in enumerate(file_list[:500]):
		if not os.path.isfile(path_data + file + "-" + str(complex_level) + ".txt") \
				or not os.path.isfile(path_data + file + "-" + str(simple_level) + ".txt"):
			continue

		# Train model over them:
		model = massalign.models.TFIDFModel([path_data + file + "-" + str(complex_level) + ".txt",
											 path_data + file + "-" + str(simple_level) + ".txt"],
											stopword_path)

		# Get MASSA aligner for convenience:
		m = massalign.core.MASSAligner()

		# Get paragraph aligner:
		paragraph_aligner = massalign.aligners.VicinityDrivenParagraphAligner(similarity_model=model,
																			  acceptable_similarity=0.2)
		# Get sentence aligner:
		sentence_aligner = massalign.aligners.VicinityDrivenSentenceAligner(similarity_model=model,
																			acceptable_similarity=0.2,
																			similarity_slack=0.05)
		# get paragraphs per document (no sentence split
		paragraphs_file1 = m.getParagraphsFromDocument(
			path_data + file + "-" + str(complex_level) + ".txt")
		paragraphs_file2 = m.getParagraphsFromDocument(
			path_data + file + "-" + str(simple_level) + ".txt")

		# Align paragraphs without sentence split:
		alignments, aligned_paragraphs = m.getParagraphAlignments(paragraphs_file1, paragraphs_file2, paragraph_aligner)

		# Align sentences in each pair of aligned paragraphs:
		for n, aligned_pair in enumerate(aligned_paragraphs):
			par_id_complex = "_".join([str(par_id) for par_id in alignments[n][0]])
			par_id_simple = "_".join([str(par_id) for par_id in alignments[n][1]])
			aligned_par_file1 = aligned_pair[0]
			aligned_par_file2 = aligned_pair[1]
			alignments_sentence, aligned_sents = m.getSentenceAlignments(aligned_par_file1,
																		 aligned_par_file2,
																		 sentence_aligner)
			for a, aligned_sent_pair in enumerate(aligned_sents):
				sent_id_simple, sent_id_complex, simple_sent, complex_sent = list(), list(), list(), list()
				for s_complex, sent_complex in enumerate(aligned_sent_pair[0]):
					sent_id_complex.append(str(alignments_sentence[a][0][s_complex]))
					complex_sent.append(aligned_sent_pair[0][s_complex].strip())
				for s_simple, sent_simple in enumerate(aligned_sent_pair[1]):
					sent_id_simple.append(str(alignments_sentence[a][1][s_simple]))
					simple_sent.append(aligned_sent_pair[1][s_simple].strip())
				sent_id_simple = "|".join(sent_id_simple)
				sent_id_complex = "|".join(sent_id_complex)
				complex_sent = " ".join(complex_sent)
				simple_sent = " ".join(simple_sent)
				alignment_id, complex_identifier, simple_identifier = get_ids(lang, corpus, str(i),
																			  par_id_complex, par_id_simple,
																			  sent_id_complex,
																			  sent_id_simple,
																			  str(complex_level),
																			  str(simple_level))
				aligned_values = align_sentences(alignment_id, complex_identifier, simple_identifier,
												 str(complex_level),
												 str(simple_level), complex_sent, simple_sent,
												 include_text=include_text)
				if add_features:
					aligned_values.extend(get_feature_values(spacy_model, complex_sent, simple_sent, lang))
				aligned_sentences.append(aligned_values)
				new_conllu, new_ids = get_ids_and_conllu_text(complex_identifier, simple_identifier,
															  list_unique_identifier, spacy_model,
															  complex_sent, simple_sent,
															  use_conllu=use_conllu)
				list_unique_identifier.extend(new_ids)
				conllu_text.extend(new_conllu)

	save_data(lang, corpus, aligned_sentences, conllu_text, conllu_path)
	return 1


def main(argv):
	path_data, conllu_path, model_name, process_stanza = "", "", "", ""
	try:
		opts, args = getopt.getopt(argv, "f:m:o:", ["filespath=", "outputpath=", "spacymodel=", "stanza="])
	except getopt.GetoptError:
		print('preprocess_newsela.py -f <filespath> -o <outputpath> -m <spacymodel>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('preprocess_newsela.py -f <filespath> -o <outputpath> -m <spacymodel> -s <stanza>')
			sys.exit()
		elif opt in ("-f", "--filespath"):
			path_data = arg
		elif opt in ("-m", "--spacymodel"):
			model_name = arg
		elif opt in ("-s", "--stanza"):
			process_stanza = arg
		elif opt in ("-o", "--outputpath"):
			conllu_path = arg
		else:
			sys.exit(2)
	print(path_data, conllu_path, model_name)
	include_text, use_conllu, add_features = True, False, True
	global SPACY_MODEL
	process_stanza = False
	if "newsela" in path_data.lower() and "2016" in path_data:
		if not process_stanza:
			SPACY_MODEL = spacy.load(model_name)
		else:
			stanza.download('en')
			snlp = stanza.Pipeline(lang="en")
			SPACY_MODEL = StanzaLanguage(snlp)
		result = align_newsela_corpus(path_data, SPACY_MODEL, conllu_path, sep="\t", include_text=include_text, add_features=add_features, use_conllu=use_conllu)
	elif "newsela" in path_data.lower() and "es" in path_data.lower():
		if not process_stanza:
			SPACY_MODEL = spacy.load(model_name)
		else:
			stanza.download('es')
			snlp = stanza.Pipeline(lang="es")
			SPACY_MODEL = StanzaLanguage(snlp)
		result = align_newsela_corpus(path_data, SPACY_MODEL, conllu_path, sep="\t", include_text=include_text, add_features=add_features, use_conllu=use_conllu)
	elif "newsela" in path_data.lower() and "2015" in path_data.lower():
		if not process_stanza:
			SPACY_MODEL = spacy.load(model_name)
		else:
			stanza.download('en')
			snlp = stanza.Pipeline(lang="en")
			SPACY_MODEL = StanzaLanguage(snlp)
		result = align_newsela_15_corpus(path_data, SPACY_MODEL, conllu_path, sep="\t", include_text=include_text, add_features=add_features, use_conllu=use_conllu)
	elif "costra" in path_data.lower():
		if not process_stanza:
			spacy_udpipe.download("cs")
			SPACY_MODEL = spacy_udpipe.load(model_name)
		else:
			stanza.download('cs')
			snlp = stanza.Pipeline(lang="cs")
			SPACY_MODEL = StanzaLanguage(snlp)
		result = align_czech_corpus(path_data, SPACY_MODEL, conllu_path, sep="\t", include_text=include_text, add_features=add_features, use_conllu=use_conllu)
	elif "paccss" in path_data.lower():
		if not process_stanza:
			SPACY_MODEL = spacy.load(model_name)
		else:
			stanza.download('it')
			snlp = stanza.Pipeline(lang="it")
			SPACY_MODEL = StanzaLanguage(snlp)
		result = align_paccss_corpus(path_data, SPACY_MODEL, conllu_path, sep="\t", include_text=include_text, add_features=add_features, use_conllu=use_conllu)
	elif "turk" in path_data.lower():
		if not process_stanza:
			SPACY_MODEL = spacy.load(model_name)
		else:
			stanza.download('en')
			snlp = stanza.Pipeline(lang="en")
			SPACY_MODEL = StanzaLanguage(snlp)
		# result = align_turk_corpus(path_data, SPACY_MODEL, "tune.8turkers.tok.", conllu_path, sep="\t")
		result = align_turk_corpus_cased(path_data, SPACY_MODEL, conllu_path, sep="\t", include_text=include_text, add_features=add_features, use_conllu=use_conllu)
	elif "klaper" in path_data.lower():
		if not process_stanza:
			SPACY_MODEL = spacy.load(model_name)
		else:
			stanza.download('de')
			snlp = stanza.Pipeline(lang="de")
			SPACY_MODEL = StanzaLanguage(snlp)
		result = align_german_klaper(path_data, SPACY_MODEL, conllu_path, include_text=include_text, add_features=add_features, sep="\t", use_conllu=use_conllu)
	elif "qats" in path_data.lower():
		if not process_stanza:
			SPACY_MODEL = spacy.load(model_name)
		else:
			stanza.download('en')
			snlp = stanza.Pipeline(lang="en")
			SPACY_MODEL = StanzaLanguage(snlp)
		result = align_qats(path_data, SPACY_MODEL, conllu_path, include_text=include_text, add_features=add_features, use_assessment=False, sep="\t", use_conllu=use_conllu)
	elif "clear" in path_data.lower():
		if not process_stanza:
			SPACY_MODEL = spacy.load(model_name)
		else:
			stanza.download('fr')
			snlp = stanza.Pipeline(lang="fr")
			SPACY_MODEL = StanzaLanguage(snlp)
		result = align_clear(path_data, SPACY_MODEL, conllu_path, include_text=include_text, add_features=add_features, sep="\t", use_conllu=use_conllu)
	else:
		result = 0
	return result


if __name__ == "__main__":
	main(sys.argv[1:])
