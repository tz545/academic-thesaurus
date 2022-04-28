import pytest
from AcademicThesaurus.generate_data import get_latex_file, latex_by_line, latex_by_pattern, id_to_text

@pytest.mark.parametrize("remove_patterns", [
	r"$\beta$", 
	r"\cite{zilman2009}", 
	r"\ref{equ:fokker-planck}",
	r"~\ref{equ22}"])
def test_latex_by_pattern_removes(remove_patterns):
	assert latex_by_pattern(remove_patterns) == ""


@pytest.mark.parametrize("extract_patterns", [
	r"\textbf{keep-this text}",
	r"\emph{ keep-this text }", 
	r"{\sl  keep-this text}"])
def text_latex_by_pattern_extracts(extract_patterns):
	assert latex_by_pattern(extract_patterns).strip() == "keep-this text"


def test_id_to_text_returns_string():
	assert type(id_to_text("https://arxiv.org/abs/astro-ph/0608371v1")) == str