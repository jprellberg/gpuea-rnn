import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from dataset import get_sequential_mnist
from popmodules import PopModule, PopRNNCell, PopLinear, pop_init_, add_popdim, pop_batch_merge, pop_batch_split
from utils import unique_string, set_seeds, loop_iter


class RnnModel1(PopModule):
    def __init__(self, popsize):
        super().__init__()
        self.rnn = PopRNNCell(popsize, 1, 128)
        self.fc = PopLinear(popsize, 128, 10)

        pop_init_(self.rnn.ih.weight, nn.init.xavier_normal_)
        pop_init_(self.rnn.hh.weight, nn.init.xavier_normal_)
        pop_init_(self.rnn.ih.bias, nn.init.zeros_)
        pop_init_(self.rnn.hh.bias, nn.init.zeros_)
        pop_init_(self.fc.weight, nn.init.xavier_normal_)
        pop_init_(self.fc.bias, nn.init.zeros_)

    def forward(self, input, hidden=None):
        for i in range(input.size(2)):
            hidden = self.rnn(input[:, :, i], hidden)
        out = self.fc(hidden)
        return out


def evaluate(model, testloader, index, device):
    model.eval()
    model.set_forward_index(index)

    all_labels = []
    all_preds = []
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        x = x.unsqueeze(0)
        pred = model(x)
        pred = pred.squeeze(0)
        all_labels.append(y.cpu())
        all_preds.append(pred.argmax(dim=1).cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()

    acc = accuracy_score(all_labels, all_preds)
    return acc


def main(args):
    trainloader, testloader = get_sequential_mnist(args.dataroot, args.batch_size)
    model = RnnModel1(args.popsize).to(args.device)
    ranks = torch.zeros(1, dtype=torch.int64, device=args.device)

    for i, (x, y) in tqdm(zip(range(args.iterations), loop_iter(trainloader)), total=args.iterations):
        if i % args.test_interval == 0:
            test_acc = evaluate(model, testloader, ranks[0], args.device)
            print(f"\niter={i:5d} test_acc={test_acc:.4f}")

        model.train()
        model.set_forward_index(None)

        x, y = x.to(args.device), y.to(args.device)
        x = add_popdim(x, args.popsize)
        y = add_popdim(y, args.popsize)

        pred = model(x)
        assert pred.size() == (args.popsize, args.batch_size, 10)

        pred = pop_batch_merge(pred)
        assert pred.size() == (args.popsize * args.batch_size, 10)

        y = pop_batch_merge(y.contiguous())
        assert y.size() == (args.popsize * args.batch_size,)

        loss = F.cross_entropy(pred, y, reduction='none')
        assert loss.size() == (args.popsize * args.batch_size,)

        loss = pop_batch_split(loss, args.popsize)
        assert loss.size() == (args.popsize, args.batch_size)

        loss = loss.mean(1)
        assert loss.size() == (args.popsize,)

        ranks = torch.argsort(loss)
        reproduce_(model, ranks, args)


def reproduce_(popmodel, ranks, args):
    cE = int(round(args.popsize * args.pe))
    cC = int(round(args.popsize * args.pc))
    cM = int(round(args.popsize * args.pm))
    assert cE + cC + cM == args.popsize

    eligible = int(round(len(ranks) * args.rho))
    idx_trunc = torch.multinomial(torch.ones(eligible, device=args.device), num_samples=2 * cC + cM, replacement=True)

    idx_cross1 = idx_trunc[0*cC:1*cC]
    idx_cross2 = idx_trunc[1*cC:2*cC]
    idx_mut = idx_trunc[2*cC:]
    idx_elite = ranks[:cE]

    def crossover_uniform(tensor):
        c1 = tensor[idx_cross1]
        c2 = tensor[idx_cross2]
        return torch.where(torch.empty_like(c1, dtype=torch.uint8).random_(0, 2), c1, c2)

    def reproduce_tensor(tensor):
        e = tensor[idx_elite]
        c = crossover_uniform(tensor)
        m = tensor[idx_mut]
        m = m + args.sigma * torch.randn_like(m)

        result = torch.cat((e, c, m))
        assert tensor.shape == result.shape
        return result

    def reproduce_module_(module):
        if isinstance(module, PopLinear):
            module.weight.data = reproduce_tensor(module.weight)
            module.bias.data = reproduce_tensor(module.bias)

    popmodel.apply(reproduce_module_)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--iterations', type=int, default=int(1e4))
    parser.add_argument('--test-interval', type=int, default=100)
    parser.add_argument('--popsize', type=int, default=100)
    parser.add_argument('--sigma', type=float, default=1e-4)
    parser.add_argument('--pe', type=float, default=0.05)
    parser.add_argument('--pc', type=float, default=0.50)
    parser.add_argument('--pm', type=float, default=0.45)
    parser.add_argument('--rho', type=float, default=0.5)
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

    with torch.no_grad():
        main(args)
