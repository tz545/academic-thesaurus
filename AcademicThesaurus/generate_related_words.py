import pickle
import os
import torch
import numpy as np 
from word2vec_cbow import Vocabulary, CBOW_Model

## with code for computing cosine similarity taken from https://github.com/OlgaChernytska/word2vec-pytorch

def get_top_similar(embeddings_norm, vocab, word, topN):
	word_id = vocab.word2index[word]
	if word_id == 0:
		print("Out of vocabulary word")
		return

	word_vec = embeddings_norm[word_id]
	word_vec = np.reshape(word_vec, (len(word_vec), 1))
	dists = np.matmul(embeddings_norm, word_vec).flatten()
	topN_ids = np.argsort(-dists)[1 : topN + 1]

	topN_dict = {}
	for sim_word_id in topN_ids:
		sim_word = vocab.index2word[sim_word_id]
		topN_dict[sim_word] = dists[sim_word_id]
	return topN_dict


def academic_thesaurus(keywords, word, cuda=False, topN=10):

	voc_path = os.path.realpath('.').replace("AcademicThesaurus/AcademicThesaurus", "AcademicThesaurus/data_processed")

	with open(os.path.join(voc_path, keywords.replace(' ', '_') + "_vocab.pkl"), 'rb') as inp:
		voc = pickle.load(inp)

	model_path = os.path.realpath('.').replace("AcademicThesaurus/AcademicThesaurus", "AcademicThesaurus/models")

	model = CBOW_Model(voc.num_words, 600)
	if cuda:
		model.load_state_dict(torch.load(os.path.join(model_path, keywords.replace(' ', '_') + '.pt')))
	else:
		model.load_state_dict(torch.load(os.path.join(model_path, keywords.replace(' ', '_') + '.pt'), map_location=torch.device('cpu')))

	# embedding from first model layer
	embeddings = list(model.parameters())[0]
	embeddings = embeddings.cpu().detach().numpy()

	# normalization
	norms = (embeddings ** 2).sum(axis=1) ** (1 / 2)
	norms = np.reshape(norms, (len(norms), 1))
	embeddings_norm = embeddings / norms

	return get_top_similar(embeddings_norm, voc, word, topN)


if __name__ == '__main__':
	print(academic_thesaurus("ecology", "biology"))