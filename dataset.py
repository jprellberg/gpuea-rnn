from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


def get_sequential_mnist(dataroot, batch_size):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1, 1))
    ])

    trainset = MNIST(dataroot, train=True, transform=tf)
    testset = MNIST(dataroot, train=False, transform=tf)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
    testloader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=1)

    return trainloader, testloader
