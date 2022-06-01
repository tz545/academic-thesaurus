import numpy as np
from AcademicThesaurus.word2vec_cbow import Vocabulary
from AcademicThesaurus.generate_related_words import get_top_similar


def test_cosine_similarity_calculation():
	voc = Vocabulary("test")
	voc.add_wordlist(["one", "two", "three", "four", "five", "six", "seven"])
	embeddings = np.array([[1, 0, 0, 0],
							[2.9, 0.1, 0, 0], 
							[0, 0, 3, 5], 
							[5.8, 0.2, 0, 0], 
							[0, 0, 2, 1], 
							[0, 1, 6, 9], 
							[0, 1, 1, 1]])
	sim_dict = get_top_similar(embeddings, voc, "one", 2)
	assert sim_dict["two"] >= 0.99940
	assert sim_dict["two"] <= 0.99941
	assert sim_dict["four"] >= 0.99940
	assert sim_dict["four"] <= 0.99941
	assert len(sim_dict) == 2