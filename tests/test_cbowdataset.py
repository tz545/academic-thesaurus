import pytest
import os
import torch
from AcademicThesaurus.word2vec_cbow import CBOWDataset, Vocabulary
from torch.utils.data import DataLoader

@pytest.fixture
def rootdir():
    return os.path.dirname(os.path.abspath(__file__))

def test_dataset_initializes(rootdir):
	voc = Vocabulary("test")
	voc.add_wordlist(["one", "two", "three", "four", "five", "six", "seven"])
	testset = CBOWDataset(os.path.join(rootdir, "test_dataset"), voc, 2)

	assert testset.__len__() == 3
	assert testset.__getitem__(1)[1] == 3
	assert torch.equal(testset.__getitem__(1)[0], torch.tensor([1, 2, 4, 5], dtype=torch.long))
	assert testset.samples[0][1] == 2
	assert testset.samples[1][1] == 3
	assert testset.samples[2][1] == 4
	assert torch.equal(testset.samples[0][0], torch.tensor([0, 1, 3, 4], dtype=torch.long))
	assert torch.equal(testset.samples[1][0], torch.tensor([1, 2, 4, 5], dtype=torch.long))
	assert torch.equal(testset.samples[2][0], torch.tensor([2, 3, 5, 6], dtype=torch.long))

def test_dataset_dataloader(rootdir):
	voc = Vocabulary("test")
	voc.add_wordlist(["one", "two", "three", "four", "five", "six", "seven"])
	testset = CBOWDataset(os.path.join(rootdir, "test_dataset"), voc, 2)
	
	minibatches0 = [[0, 1, 3, 4], [1, 2, 4, 5], [2, 3, 5, 6]]
	minibatches1 = [2, 3, 4]
	dataloader = DataLoader(testset, batch_size=1, shuffle=False)
	for i, batch in enumerate(dataloader):
		cbow = batch[0][0]
		word = batch[1]
		
		assert cbow.tolist() == minibatches0[i]
		assert word.item() == minibatches1[i]
