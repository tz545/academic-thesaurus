import torch
import torch.nn as nn
import pytest
import os
from AcademicThesaurus.word2vec_cbow import CBOW_Model, CBOWDataset, Vocabulary
from torch.utils.data import DataLoader


def test_model_output_dimensions():
	model = CBOW_Model(10, 600)
	test_input1 = torch.tensor([[1, 4, 9, 6]], dtype=torch.int64)
	test_input2 = torch.tensor([[1, 4, 9, 6],[1, 4, 9, 6]], dtype=torch.int64)
	out1 = model(test_input1)
	out2 = model(test_input2)
	assert out1.shape == torch.Size([1, 10])
	assert out2.shape == torch.Size([2, 10])

@pytest.fixture
def rootdir():
    return os.path.dirname(os.path.abspath(__file__))

def test_training_decreases_loss(rootdir):
	voc = Vocabulary("test")
	voc.add_wordlist(["one", "two", "three", "four", "five", "six", "seven"])
	testset = CBOWDataset(os.path.join(rootdir, "test_dataset"), voc, 2)
	model = CBOW_Model(voc.num_words, 600)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	loader = DataLoader(testset, batch_size=3, shuffle=False)
	optimizer.zero_grad()
	batch = next(iter(loader))
	output = model(batch[0])
	loss1 = criterion(output, batch[1])
	loss1.backward()
	optimizer.step()
	optimizer.zero_grad()
	output = model(batch[0])
	loss2 = criterion(output, batch[1])
	assert loss2.item() < loss1.item()






	
