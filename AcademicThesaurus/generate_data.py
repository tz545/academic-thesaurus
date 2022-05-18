import re
import urllib.request
import feedparser
import tarfile
import os
import string
from spacy.lang.en.stop_words import STOP_WORDS 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import shutil


def get_latex_file(arxiv_id, temp_dir_name):
	"""Looks for source tex files on arxiv and downloads"""
	download_url = arxiv_id.replace("abs", "e-print")

	try:
		urllib.request.urlretrieve(download_url, os.path.join(temp_dir_name,"download.tar.gz"))
	except urllib.error.HTTPError:
		print("Source files not found, skipping...")
		return None

	try:
		tar = tarfile.open(os.path.join(temp_dir_name,"download.tar.gz"), "r:gz")
		tar.extractall(temp_dir_name)
		tar.close()
	except tarfile.ReadError:
		print("Invalid tarfile, skipping...")
		return None

	tex_files = [x for x in os.listdir(temp_dir_name) if ".tex" in x]

	return tex_files


def latex_by_line(latex_file):
	"""Goes through file line by line to remove equations, figures, tables, comments.
	Removes newline characters."""
	
	file_text = ""
	environments_beg = []
	beg_regex = re.compile(r"\\begin\{\w+\*?\}")
	end_regex = re.compile(r"\\end\{\w+\*?\}")
	comments = re.compile(r"[^\\]%.+")
	record = False
	abstract = False
	doc = False

	with open(latex_file) as fp:

		try:
			for i, line in enumerate(fp):

				if len(line.lstrip()) > 0:
					if line.lstrip()[0] == '%':
						continue

				if r"\newcommand{" in line and r"\begin" in line:
					return None

				if r"\newcommand{" in line and r"\end" in line:
					return None

				if r"\begin" in line:

					line = line.replace(' ', '')

					matches = re.findall(beg_regex, line)
					for m in matches:
						env_keyword = m[7:-1]
						environments_beg.append(env_keyword)

						if env_keyword == 'abstract':
							record = True
							abstract = True

						if env_keyword == 'document':
							doc = True

					continue

				if r"\end" in line and doc:

					line = line.replace(' ', '')

					matches = re.findall(end_regex, line)
					for m in matches:
						env_keyword = m[5:-1]

						if env_keyword == 'document':
							break

						else:
							try:
								environments_beg.remove(env_keyword)
							except ValueError:
								return None

						if env_keyword == 'abstract':
							abstract = False

					continue

				## if no abstract, text begins at first section
				if r"\section" in line:
					record = True
					continue

				if record:
					if abstract:

						## check no comments in line
						line = re.sub(r"[^\\]%.+", '', line)
						file_text += line.strip() + ' '

					else:
						
						if environments_beg == ["document"]:

							line = re.sub(r"[^\\]%.+", '', line)
							file_text += line.strip() + ' '

		except UnicodeDecodeError:
			return None

	return file_text


def latex_by_pattern(file_text):
	"""Removes inline math, citations, references etc. 
	Tries to perserve text from section headings and italicized/bolded text"""

	## remove math
	file_text = re.sub(r"\$\$[^\$]+\$\$", '', file_text)
	file_text = re.sub(r"\$[^\$]+\$", '', file_text)

	## extract words from \section, \textit \emph etc (currently also includes \bibstyle)
	bracket_text1 = re.compile(r"{[a-z\sA-Z\-]+}")
	command_bracket = re.finditer(r"\\[\w\s*]+{[a-z\sA-Z\-]+}", file_text)
	for i, comm in enumerate(command_bracket):	
		words = bracket_text1.search(comm.group())
		if words != None:
			file_text = file_text.replace(comm.group(), words.group(0)[1:-1])

	## extract words from {\sl }, {\it } etc
	## this still keeps the trailing '}', but this can be removed later with all punctuation
	bracket_text2 = re.compile(r"{\s*\\[a-zA-Z]+")
	bracket_command = re.finditer(r"{\s*\\[a-z\sA-Z\-]+}", file_text)
	## comm will contain many repeats. Can generate list->set instead to only execute replace
	## once for each unique comm. 
	for i, comm in enumerate(bracket_command):
		command = bracket_text2.search(comm.group())
		if command != None:
			file_text = file_text.replace(command.group(0), '')


	## remove remaining ~\*{*}, \*{*} and ~\*, \* commands
	remove1 = re.finditer(r"~?\\[\w\s*]+{[^}]+}", file_text)
	remove2 = re.finditer(r"~?\\[\w*]+", file_text)

	for i, comm in enumerate(remove1):
		file_text = file_text.replace(comm.group(), '')
	for i, comm in enumerate(remove2):
		file_text = file_text.replace(comm.group(), '')

	return file_text


def text_preprocessing(file_text, lemmatize=False):
	"""Implements standard text preprocessing steps on extracted LaTex text
	Lemmatization is optional as it is slow and only minimally successful"""

	## convert to lower case
	file_text = file_text.lower()

	## remove numbers
	file_text = re.sub(r'\d+', '', file_text)

	## separate hyphenated words
	file_text = file_text.replace('-', ' ')

	## remove remaining punctuation
	file_text = file_text.translate(str.maketrans("","", string.punctuation))

	## filter stop words and non-english words (e.g. author names)
	STOP_WORDS.update(['et', 'al', 'introduction', 'methods', 'discussion', 'conclusions', 'conclusion', 'figure', 'equation', 'ref', 'equ', 'eg'])
	words = set(nltk.corpus.words.words())

	tokens = word_tokenize(file_text)

	if lemmatize:
		lemmatizer = WordNetLemmatizer() 
		filtered_text = [lemmatizer.lemmatize(i) for i in tokens if not i in STOP_WORDS and i in words]
	
	else:
		filtered_text = [i for i in tokens if not i in STOP_WORDS and i in words]

	return(filtered_text)


def id_to_text(arxiv_id, save=False):
	"""Pipeline function for getting cleaned text from arxiv id"""

	temp_dir_name = "raw_download"
	if not os.path.exists(temp_dir_name):
		os.mkdir(temp_dir_name)

	file_seach = get_latex_file(arxiv_id, temp_dir_name)

	combined_text = ''

	if file_seach != None:

		for file in file_seach:
			raw_text = latex_by_line(os.path.join(temp_dir_name, file))
			if raw_text != None:
				combined_text += latex_by_pattern(raw_text)

	else:
		shutil.rmtree(temp_dir_name)
		return 1

	shutil.rmtree(temp_dir_name)

	if combined_text == '':

		return 2

	else:

		preprocessed_text = text_preprocessing(combined_text)

		if save:
			save_name = re.sub(r'http.+\/', '', arxiv_id) + ".txt"
			curr_path = os.path.realpath('.')
			save_path = curr_path.replace("AcademicThesaurus/AcademicThesaurus", "AcademicThesaurus/data")
			with open(os.path.join(save_path, save_name), "w") as outfile:
				outfile.write(" ".join(preprocessed_text))

		return preprocessed_text


def arxiv_query(keywords, no_results):

	base_url = 'http://export.arxiv.org/api/query?'

	search_query = 'abs:' + keywords
	arxiv_ids = []

	counter = 0

	## query times out if too many results requested at once
	while counter < no_results:

		start = counter
		max_results = min(200, no_results-counter)

		query = 'search_query=%s&start=%i&max_results=%i' % (search_query,
													  start,
													  max_results)
		response = urllib.request.urlopen(base_url+query).read()
		feed = feedparser.parse(response)
		
		if len(feed.entries) == 0:
			break

		for entry in feed.entries:
			arxiv_ids.append(entry.get('id'))

		counter += len(feed.entries)

	return arxiv_ids


if __name__ == "__main__":
	print(id_to_text("https://arxiv.org/abs/astro-ph/0608371v1", True))