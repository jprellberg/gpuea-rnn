import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.optim import RMSprop
from tqdm import tqdm

from dataset import get_sequential_mnist
from utils import unique_string, set_seeds, loop_iter, count_parameters


class RnnModel1(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNNCell(1, 128)
        self.fc = nn.Linear(128, 10)

        nn.init.xavier_normal_(self.rnn.weight_ih)
        nn.init.xavier_normal_(self.rnn.weight_hh)
        nn.init.zeros_(self.rnn.bias_ih)
        nn.init.zeros_(self.rnn.bias_hh)
        nn.init.xavier_normal_(self.fc.weight.data)
        nn.init.zeros_(self.fc.bias.data)

    def forward(self, input, hidden=None):
        for i in range(input.size(1)):
            hidden = self.rnn(input[:, i], hidden)
        out = self.fc(hidden)
        return out


class RnnModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn1 = nn.RNNCell(1, 50)
        self.rnn2 = nn.RNNCell(50, 50)
        self.rnn3 = nn.RNNCell(50, 50)
        self.fc = nn.Linear(50, 10)

        for rnn in [self.rnn1, self.rnn2, self.rnn3]:
            nn.init.xavier_normal_(rnn.weight_ih)
            nn.init.xavier_normal_(rnn.weight_hh)
            nn.init.zeros_(rnn.bias_ih)
            nn.init.zeros_(rnn.bias_hh)

        nn.init.xavier_normal_(self.fc.weight.data)
        nn.init.zeros_(self.fc.bias.data)

    def forward(self, input, h1=None, h2=None, h3=None):
        for i in range(input.size(1)):
            h1 = self.rnn1(input[:, i], h1)
            h2 = self.rnn2(h1, h2)
            h3 = self.rnn3(h2, h3)
        out = self.fc(h3)
        return out


def evaluate(model, testloader, device):
    model.eval()

    all_labels = []
    all_preds = []
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
            all_labels.append(y.cpu())
            all_preds.append(pred.argmax(dim=1).cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()

    acc = accuracy_score(all_labels, all_preds)
    return acc


def main(args):
    trainloader, testloader = get_sequential_mnist(args.dataroot, args.batch_size)
    model = RnnModel1().to(args.device)
    opt = RMSprop(model.parameters(), lr=args.lr)

    print("Model parameters:", count_parameters(model))
    for i, (x, y) in tqdm(zip(range(args.iterations), loop_iter(trainloader)), total=args.iterations):
        if i % args.test_interval == 0:
            test_acc = evaluate(model, testloader, args.device)
            print(f"\niter={i:5d} test_acc={test_acc:.4f}")

        model.train()

        x, y = x.to(args.device), y.to(args.device)
        pred = model(x)
        loss = F.cross_entropy(pred, y)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()


def parse_args():
    # Adam: ~55% test accuracy, RMSprop: ~60% test accuracy with RnnModel1 (single layer)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--iterations', type=int, default=int(1e4))
    parser.add_argument('--test-interval', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--clip', type=float, default=1)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', default='results')
    args = parser.parse_args()
    args.out = os.path.join(args.out, unique_string())
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    set_seeds(args.seed)
    main(args)
