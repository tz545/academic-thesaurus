import re
import urllib.request
import tarfile
import os


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
	environments_end = []
	beg_regex = re.compile(r"\\begin\{\w+\}")
	end_regex = re.compile(r"\\end\{\w+\}")
	comments = re.compile(r"[^\\]%.+")

	with open(latex_file) as fp:

		for i, line in enumerate(fp):

			if len(line.lstrip()) > 0:
				if line.lstrip()[0] == '%':
					continue

			if r"\begin{" in line:

				env = beg_regex.search(line)
				environments_beg.append(env.group(0)[7:-1])
				continue

			if r"\end{" in line:

				env = end_regex.search(line)
				environments_end.append(env.group(0)[5:-1])
				continue

			if 'document' in environments_beg and 'abstract' in environments_beg:
				if 'abstract' in environments_end:

					## check if we are in another environment
					## skip figures, tables, equations
					if len(environments_beg) - len(environments_end) == 1:
						file_text += line.strip() + ' '

				else:
					## check no comments in line
					line = re.sub(r"[^\\]%.+", '', line)

					file_text += line.strip() + ' '

	return file_text


def latex_by_pattern(file_text):
	"""Removes inline math, citations, references etc. 
	Tries to perserve text from section headings and italicized/bolded text"""

	## remove math
	file_text = re.sub(r"\$[^\$]+\$", '', file_text)

	## extract words from \section, \textit \emph etc (currently also includes \bibstyle)
	bracket_text1 = re.compile(r"{[a-z\sA-Z\-]+}")
	command_bracket = re.finditer(r"\\[\w\s*]+{[a-z\sA-Z\-]+}", file_text)
	for i, comm in enumerate(command_bracket):	
		words = bracket_text1.search(comm.group())
		file_text = file_text.replace(comm.group(), words.group(0)[1:-1])

	## extract words from {\sl }, {\it } etc
	## this still keeps the trailing '}', but this can be removed later with all punctuation
	bracket_text2 = re.compile(r"{\s*\\[a-zA-Z]+")
	bracket_command = re.finditer(r"{\s*\\[a-z\sA-Z\-]+}", file_text)
	## comm will contain many repeats. Can generate list->set instead to only execute replace
	## once for each unique comm. 
	for i, comm in enumerate(bracket_command):
		command = bracket_text2.search(comm.group())
		file_text = file_text.replace(command.group(0), '')


	## remove remaining ~\*{*}, \*{*} and ~\*, \* commands
	remove1 = re.finditer(r"~?\\[\w\s*]+{[^}]+}", file_text)
	remove2 = re.finditer(r"~?\\[\w\s*]+", file_text)

	for i, comm in enumerate(remove1):
		file_text = file_text.replace(comm.group(), '')
	for i, comm in enumerate(remove2):
		file_text = file_text.replace(comm.group(), '')

	return file_text


def id_to_text(arxiv_id):
	"""Pipeline function for getting cleaned text from arxiv id"""

	temp_dir_name = "raw_download"
	if not os.path.exists(temp_dir_name):
		os.mkdir(temp_dir_name)

	file_seach = get_latex_file(arxiv_id, temp_dir_name)

	if file_seach == None:
		return None
	else:
		combined_text = ''
		for file in file_seach:
			raw_text = latex_by_line(os.path.join(temp_dir_name, file))
			combined_text += latex_by_pattern(raw_text)

	temp_files = os.listdir(temp_dir_name)
	for f in temp_files:
		os.remove(os.path.join(temp_dir_name, f))

	return combined_text


if __name__ == "__main__":
	print(id_to_text("https://arxiv.org/abs/astro-ph/0608371v1"))