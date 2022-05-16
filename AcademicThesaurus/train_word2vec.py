import pickle
import os
from generate_data import arxiv_query, id_to_text
from word2vec_cbow import Vocabulary, CBOWDataset, CBOW_Model
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn


def download_data(keywords, no_results):

	keywords = "%22" + keywords.replace(' ', '+') + "%22"

	arxiv_ids = arxiv_query(keywords, no_results)
	for paper_id in arxiv_ids:
		id_to_text(paper_id, save=True)


def build_dataset(keywords, window_size, save=False):

	datapath = os.path.realpath('.').replace("AcademicThesaurus/AcademicThesaurus", "AcademicThesaurus/data")

	voc = Vocabulary(keywords)
	for textfile in os.listdir(datapath):
		with open(os.path.join(datapath, textfile)) as f:
			data = f.readline()
			data = data.split(" ")
			voc.add_wordlist(data)

	dataset = CBOWDataset(datapath, voc, window_size)

	if save:

		savepath = os.path.realpath('.').replace("AcademicThesaurus/AcademicThesaurus", "AcademicThesaurus/data_processed")

		with open(os.path.join(savepath, keywords.replace(' ', '_') + "_vocab.pkl"), 'wb') as outp:
			pickle.dump(voc, outp, pickle.HIGHEST_PROTOCOL)

		with open(os.path.join(savepath, keywords.replace(' ', '_') + "_ds.pkl"), 'wb') as outp:
			pickle.dump(dataset, outp, pickle.HIGHEST_PROTOCOL)

	return voc, dataset


def train_model(keywords, dataset, vocab, batch_size, epochs, device):

	no_trainng_samples = int(0.8*len(dataset.samples))
	no_val_samples = len(dataset.samples) - no_trainng_samples
	
	trainset, valset = random_split(dataset, [no_trainng_samples, no_val_samples])

	train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)
	
	model = CBOW_Model(vocab.num_words, 600)
	model.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	train_loss = []
	val_loss = []

	for e in range(epochs):

		total_train_loss = 0
		total_val_loss = 0

		for i, batch in enumerate(train_loader):

			optimizer.zero_grad()

			output = model(batch[0].to(device))
			loss = criterion(output, batch[1].to(device))

			loss.backward()
			optimizer.step()

			total_train_loss += loss.item()/len(batch[1])

			train_loss.append(loss.item()/len(batch[1]))

		print("Total training loss: ", total_train_loss/(i+1))

		with torch.no_grad(): 

			for i, batch in enumerate(val_loader):

				output = model(batch[0].to(device))
				loss = criterion(output, batch[1].to(device))

				total_val_loss += loss.item()/len(batch[1])

				val_loss.append(loss.item()/len(batch[1]))

			print("Total validation loss: ", total_val_loss/(i+1))

	torch.save(model.state_dict(), os.path.join(os.path.realpath('.').replace("AcademicThesaurus/AcademicThesaurus", "AcademicThesaurus/models"), keywords.replace(' ', '_') + ".pt"))

	return train_loss, val_loss


if __name__ == '__main__':
	# download_data("ecology", 200)
	build_dataset("ecology", 2, save=True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	savepath = os.path.realpath('.').replace("AcademicThesaurus/AcademicThesaurus", "AcademicThesaurus/data_processed")

	with open(os.path.join(savepath, "ecology_vocab.pkl"), 'rb') as inp:
		voc = pickle.load(inp)

	with open(os.path.join(savepath, "ecology_ds.pkl"), 'rb') as inp:
		dataset = pickle.load(inp)
	train_loss, val_loss = train_model("ecology", dataset, voc, batch_size=10, epochs=1, device=device)