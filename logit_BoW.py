import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

torch.manual_seed(1)

data = [
	("me gusta comer en la cafeteria".split(), "SPANISH"),
	("Give it to me".split(), "ENGLISH"),
	("No creo que sea una buena idea".split(), "SPANISH"),
	("No it is not a good idea to get lost at sea".split(), "ENGLISH")
]

# print(data)
test_data = [
	("Yo creo que si".split(), "SPANISH"),
	("it is lost on me".split(), "ENGLISH")
] 

word_to_ix = {}

for sent, _ in data + test_data:
	for word in sent:
		if word not in word_to_ix:
			word_to_ix[word] = len(word_to_ix)

print(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2

class BoWClassifier(nn.Module):
	def __init__(self, num_labels, vocab_size):
		super(BoWClassifier, self).__init__()

		# Parameters of the model - linear layer (vocab x 2)
		self.linear = nn.Linear(vocab_size, num_labels)

	def forward(self, bow_vec):
		return(F.log_softmax(self.linear(bow_vec), dim = 1))


def bow_vectorize(sentence, word_to_ix):
	vec = torch.zeros(len(word_to_ix))
	for word in sentence:
		vec[word_to_ix[word]] += 1
	return vec.view(1, -1)

# print(bow_vectorize("Yo creo que si".split(), word_to_ix))

def make_target(label, label_to_ix):
	return(torch.LongTensor([label_to_ix[label]]))

model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

for param in model.parameters():
	print(param)

# sample = data[0]
# bow_vector = bow_vectorize(sample[0], word_to_ix)
# probs = model(autograd.Variable(bow_vector))
# print(probs)

label_to_ix = { "SPANISH": 0, "ENGLISH": 1 }


## Before training

print("\n #### BEFORE TRAINING #### \n")
for instance, label in test_data:
	bow_vec = autograd.Variable(bow_vectorize(instance, word_to_ix))
	log_probs = model(bow_vec)
	print(log_probs)

print("\n For creo \n")
print(next(model.parameters())[:,word_to_ix["creo"]])

# print(make_target("SPANISH", label_to_ix))

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)

print("\n #### TRAINING #### \n")
for epoch in range(100):
	for instance, label in data:
		## Clear the model's gradients before each instance.
		model.zero_grad()
		bow_vec = autograd.Variable(bow_vectorize(instance, word_to_ix))
		target = autograd.Variable(make_target(label, label_to_ix))

		log_probs = model(bow_vec)

		loss = loss_function(log_probs, target)
		loss.backward()
		optimizer.step()

print("\n #### TRAINING COMPLETE #### \n")

for instance, label in test_data:
	bow_vec = autograd.Variable(bow_vectorize(instance, word_to_ix))
	log_probs = model(bow_vec)
	print(log_probs)

print("\n For creo \n")
print(next(model.parameters())[:,word_to_ix["creo"]])