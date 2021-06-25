import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from get_data import get_mnist
from network import Network

EPOCHS=3
BATCH_SIZE=32
LEARNING_RATE=0.002

train_loader, test_loader = get_mnist(BATCH_SIZE)

TRAINING_SIZE = BATCH_SIZE * len(train_loader)
TESTING_SIZE = BATCH_SIZE * len(test_loader)

network = Network()
optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)

training_accuracies = []
testing_accuracies = []


if not os.path.exists("./artifacts/"):
    os.mkdir("./artifacts")
else:
    for file in os.listdir("./artifacts/"):
        os.remove(f"./artifacts/{file}")

f = open("./artifacts/log.txt", "w")

def get_num_correct(preds, labels) -> int:
    return preds.argmax(dim=1).eq(labels).sum().item()


def train() -> None:
    network.train()
    correct_in_episode = 0
    episode_loss = 0

    for batch in train_loader:
        images, labels = batch

        predictions = network(images.squeeze(1).reshape(-1, 28 * 28))
        loss = F.cross_entropy(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_loss += loss.item()
        correct_in_episode += get_num_correct(predictions, labels)        

    training_accuracies.append(correct_in_episode * 100 / TRAINING_SIZE)

    log = f"Epoch: {epoch + 1} accuracy: {correct_in_episode * 100 / TRAINING_SIZE:.2f} loss: {episode_loss}"
    f.write(log)
    print(log, end="\t")


def test() -> None:
    network.eval()
    episode_loss = 0
    correct_in_episode = 0

    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch

            predictions = network(images.squeeze(1).reshape(-1, 28 * 28))
            loss = F.cross_entropy(predictions, labels)

            episode_loss = loss.item()
            correct_in_episode += get_num_correct(predictions, labels)

    testing_accuracies.append(correct_in_episode * 100 / TESTING_SIZE)

    log = f'Validation: Accuracy: {correct_in_episode * 100 / TESTING_SIZE:.2f}'
    f.write(log)
    print(log)


for epoch in range(EPOCHS):
    train()
    test()


plt.plot(list(range(1, len(training_accuracies)+1)), training_accuracies, color='blue')
plt.plot(list(range(1, len(testing_accuracies)+1)), testing_accuracies, color='red')

plt.legend(['Train Accuracy', 'Test Accuracy'], loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.savefig("./artifacts/plot.png")
