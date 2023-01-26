##Block 1: Doing some Common imports and see if cuda and processing on gpu instead of cpu is available
import torch
import torch.nn as nn
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(DEVICE)




from torchtext.models import RobertaClassificationHead, XLMR_LARGE_ENCODER


text_transform = XLMR_LARGE_ENCODER.transform()

import torch
xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large')
xlmr.to(DEVICE)
a = xlmr.encode('Hello World!')
print(a)
tensor([0, 35378, 6661,38,2])
##Block 4:
from torchtext.datasets import SST2
from torch.utils.data import DataLoader
batch_size = 64

train_datapipe = SST2(split='train')
dev_datapipe = SST2(split='dev')
print(train_datapipe.map)
# Transform the raw dataset using non-batched API (i.e apply transformation line by line)
train_datapipe = train_datapipe.map(lambda x: (text_transform(x[0]), x[1]))
train_datapipe = train_datapipe.batch(batch_size)
train_datapipe = train_datapipe.rows2columnar(["token_ids", "target"])
train_dataloader = DataLoader(train_datapipe, batch_size=None)

dev_datapipe = dev_datapipe.map(lambda x: (text_transform(x[0]), x[1]))
dev_datapipe = dev_datapipe.batch(batch_size)
dev_datapipe = dev_datapipe.rows2columnar(["token_ids", "target"])
dev_dataloader = DataLoader(dev_datapipe, batch_size=None)

print("hello")

#Block 6
num_classes = 2
input_dim = 1024

classifier_head = RobertaClassificationHead(num_classes=num_classes, input_dim=input_dim)
model = XLMR_LARGE_ENCODER.get_model(head=classifier_head)
model.to(DEVICE)




#Block 7
import torchtext.functional as F
from torch.optim import AdamW

learning_rate = 1e-5
optim = AdamW(model.parameters(), lr=learning_rate)
criteria = nn.CrossEntropyLoss()


def train_step(input, target):
    output = model(input)
    loss = criteria(output, target)
    optim.zero_grad()
    loss.backward()
    optim.step()


def eval_step(input, target):
    output = model(input)
    loss = criteria(output, target).item()
    return float(loss), (output.argmax(1) == target).type(torch.float).sum().item()


def evaluate():
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    counter = 0
    with torch.no_grad():
        for batch in dev_dataloader:
            input = F.to_tensor(batch['token_ids'], padding_value=padding_idx).to(DEVICE)
            target = torch.tensor(batch['target']).to(DEVICE)
            loss, predictions = eval_step(input, target)
            total_loss += loss
            correct_predictions += predictions
            total_predictions += len(target)
            counter += 1

    return total_loss / counter, correct_predictions / total_predictions






##Block 8
num_epochs = 10

for e in range(num_epochs):
    for batch in train_dataloader:
        input = F.to_tensor(batch['token_ids'], padding_value=padding_idx).to(DEVICE)
        target = torch.tensor(batch['target']).to(DEVICE)
        train_step(input, target)

    loss, accuracy = evaluate()
    print("Epoch = [{}], loss = [{}], accuracy = [{}]".format(e, loss, accuracy))

print("End")

