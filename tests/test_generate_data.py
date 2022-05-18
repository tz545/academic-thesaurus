import pytest
import os
from AcademicThesaurus.generate_data import get_latex_file, latex_by_line, latex_by_pattern, text_preprocessing, id_to_text


def test_get_latex_file_url_error(tmpdir):
	assert get_latex_file("https://arxiv.org/abs/does-not-exist/0608371v1", tmpdir) == None


@pytest.fixture
def rootdir():
    return os.path.dirname(os.path.abspath(__file__))

@pytest.mark.parametrize("sample_tex_files", [
	"test_with_abstract.tex",
	"test_without_abstract.tex"])
def test_latex_by_line_parses_file(sample_tex_files, rootdir):
	file = os.path.join(rootdir, sample_tex_files)
	assert latex_by_line(file).strip() == "extract this text and this text and finally this."


@pytest.mark.parametrize("remove_patterns", [
	r"$\beta$", 
	r"\cite{zilman2009}", 
	r"\ref{equ:fokker-planck}",
	r"~\ref{equ22}",
	r"\maketitle",
	r"$ a $"])
def test_latex_by_pattern_removes(remove_patterns):
	assert latex_by_pattern(remove_patterns) == ""


@pytest.mark.parametrize("extract_patterns", [
	r"\textbf{keep-this text}",
	r"\emph{ keep-this text }", 
	"keep-this text", 
	r"\tiny keep-this text"])
def test_latex_by_pattern_extracts(extract_patterns):
	assert latex_by_pattern(extract_patterns).strip() == "keep-this text"

def test_latex_by_pattern_extracts_between_brackets():
	assert latex_by_pattern(r"{\sl  keep-this text}").strip() == "keep-this text}"


@pytest.mark.parametrize("word_samples", [
	"eLeCtron20 52 gas et al", 
	" $electron-gas?"])
def test_text_preprocessing(word_samples):
	assert text_preprocessing(word_samples) == ["electron", "gas"]


def test_id_to_text_success_return():
	assert type(id_to_text("https://arxiv.org/abs/astro-ph/0608371v1")[0]) == str

def test_id_to_text_failed_return():
	assert id_to_text("https://arxiv.org/abs/does-not-exist") == 1