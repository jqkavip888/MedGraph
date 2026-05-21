import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from importlib import import_module

PAD, CLS = '[PAD]', '[CLS]'


# =====================
# Dataset
# =====================
class MyDataset(Dataset):
    def __init__(self, path, tokenizer, pad_size):
        self.data = []
        self.tokenizer = tokenizer
        self.pad_size = pad_size

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                content, label = line.split('\t')
                self.data.append((content, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        content, label = self.data[idx]

        token = self.tokenizer.tokenize(content)
        token = [CLS] + token

        seq_len = len(token)

        if seq_len < self.pad_size:
            mask = [1] * seq_len + [0] * (self.pad_size - seq_len)
            token += [PAD] * (self.pad_size - seq_len)
        else:
            mask = [1] * self.pad_size
            token = token[:self.pad_size]
            seq_len = self.pad_size

        input_ids = self.tokenizer.convert_tokens_to_ids(token)

        return (
            torch.LongTensor(input_ids),
            torch.LongTensor(mask),
            torch.LongTensor([label])
        )


# =====================
# Train Function
# =====================
def train(config, model, train_iter, dev_iter):
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0

    flag = False

    for epoch in range(config.num_epochs):
        print(f"\nEpoch [{epoch+1}/{config.num_epochs}]")

        for i, (input_ids, mask, labels) in enumerate(train_iter):
            input_ids = input_ids.to(config.device)
            mask = mask.to(config.device)
            labels = labels.squeeze().to(config.device)

            outputs = model(input_ids, mask, None)

            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if total_batch % 50 == 0:
                true = labels.data.cpu()
                pred = torch.argmax(outputs, dim=1).cpu()

                acc = (pred == true).sum().item() / len(true)

                print(f"step: {total_batch}, loss: {loss.item():.4f}, acc: {acc:.4f}")

            total_batch += 1

        dev_loss = evaluate(config, model, dev_iter, loss_fn)

        print(f"\nDev Loss: {dev_loss:.4f}")

        if dev_loss < dev_best_loss:
            dev_best_loss = dev_loss
            torch.save(model.state_dict(), config.save_path)
            print("✔ Saved best model")
            last_improve = total_batch
        else:
            if total_batch - last_improve > config.require_improvement:
                print("⚠ Early stop triggered")
                flag = True
                break

        if flag:
            break


# =====================
# Eval Function
# =====================
def evaluate(config, model, data_iter, loss_fn):
    model.eval()

    total_loss = 0
    total_acc = 0
    n = 0

    with torch.no_grad():
        for input_ids, mask, labels in data_iter:
            input_ids = input_ids.to(config.device)
            mask = mask.to(config.device)
            labels = labels.squeeze().to(config.device)

            outputs = model(input_ids, mask, None)

            loss = loss_fn(outputs, labels)

            total_loss += loss.item()

            pred = torch.argmax(outputs, dim=1)
            total_acc += (pred == labels).sum().item()

            n += len(labels)

    return total_loss / len(data_iter)


# =====================
# Main
# =====================
if __name__ == "__main__":

    x = import_module("models.bert")
    config = x.Config("red_spider")
    model = x.Model(config).to(config.device)

    print("Loading dataset...")

    train_dataset = MyDataset(config.train_path, config.tokenizer, config.pad_size)
    dev_dataset = MyDataset(config.dev_path, config.tokenizer, config.pad_size)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size)

    print("Start training...")

    train(config, model, train_loader, dev_loader)