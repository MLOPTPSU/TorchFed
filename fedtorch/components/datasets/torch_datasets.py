# -*- coding: utf-8 -*-
import os

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from fedtorch.components.dataset_builder import DATASET

@DATASET.register_module()
def cifar10(root, split, transform, target_transform, download, num_classes=10,  dimension=[3,32,32]):
    return _get_cifar('cifar10', root, split, transform, target_transform, download)

@DATASET.register_module()
def cifar100(root, split, transform, target_transform, download, num_classes=100,  dimension=[3,32,32]):
    return _get_cifar('cifar100', root, split, transform, target_transform, download)

@DATASET.register_module()
def mnist(root, split, transform, target_transform, download, num_classes=10, dimension=[28,28]):
    return _get_mnist(root, split, transform, target_transform, download)



@DATASET.register_module()
def fashion_mnist(root, split, download, num_classes=10, dimension=[28,28]):
    return _get_fashion_mnist(root, split, download)

@DATASET.register_module()
def stl10(root, split, transform, target_transform, download, num_classes=10, dimension=[3,96,96]):
    return _get_stl10(root, split, transform, target_transform, download)



def _get_cifar(name, root, split, transform, target_transform, download):
    is_train = (split == 'train')

    # decide normalize parameter.
    if name == 'cifar10':
        dataset_loader = datasets.CIFAR10
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    elif name == 'cifar100':
        dataset_loader = datasets.CIFAR100
        normalize = transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    # decide data type.
    if is_train:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((32, 32), 4),
            transforms.ToTensor(),
            normalize])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    return dataset_loader(root=root,
                          train=is_train,
                          transform=transform,
                          target_transform=target_transform,
                          download=download)


def _get_mnist(root, split, transform, target_transform, download):
    is_train = (split == 'train')

    if is_train:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    return datasets.MNIST(root=root,
                          train=is_train,
                          transform=transform,
                          target_transform=target_transform,
                          download=download)


def _get_fashion_mnist(root, split, download):
    is_train = (split == 'train')
    return  datasets.FashionMNIST(
                    root = root,
                    train = is_train,
                    download = download,
                    transform = transforms.Compose([
                        transforms.ToTensor()                                 
                    ])
                )


def _get_stl10(root, split, transform, target_transform, download):
    return datasets.STL10(root=root,
                          split=split,
                          transform=transform,
                          target_transform=target_transform,
                          download=download)




# def _get_synthetic(args, root, split,):
#     reg = 'least_square' in args.arch
#     dim=60
#     return Synthetic(root, split=split, client_id=rank,
#                     num_tasks=n_nodes, alpha=synthetic_alpha, 
#                     beta=synthetic_beta, regression=reg,num_dim=dim)



# def get_dataset(
#         args, name, datasets_path, split='train', transform=None,
#         target_transform=None, download=True):
#     root = os.path.join(datasets_path, name)
#     download = True if args.graph.rank == 0 else False # Only the server downloads the dataset to avoid filesystem error

#     if name == 'cifar10' or name == 'cifar100':
#         return _get_cifar(
#             name, root, split, transform, target_transform, download)
#     elif name == 'mnist':
#         return _get_mnist(root, split, transform, target_transform, download)
#     elif 'emnist' in name:
#         if name =='emnist':
#             only_digits = True
#         elif name == 'emnist_full':
#             only_digits = False
#         else:
#             raise ValueError("The dataset %s does not exist!" % name)
#         return _get_emnist(root, split,args.graph.rank, download, args.fed_personal, only_digits)
#     elif name == 'fashion_mnist':
#         return _get_fashion_mnist(root, split, transform, target_transform, download)
#     elif name == 'stl10':
#         return _get_stl10(root, split, transform, target_transform, download)
#     elif name == 'epsilon':
#         return _get_epsilon_or_rcv1_or_MSD(args, root, name, split)
#     elif name == 'rcv1':
#         return _get_epsilon_or_rcv1_or_MSD(args, root, name, split)
#     elif name == 'higgs':
#         return _get_epsilon_or_rcv1_or_MSD(args, root, name, split)
#     elif name == 'MSD':
#         return _get_epsilon_or_rcv1_or_MSD(args, root, name, split)
#     elif name == 'synthetic':
#         return _get_synthetic(args, root, name, split)
#     elif name == 'adult':
#         return _get_adult(root, split)
#     elif name == 'shakespeare':
#         return _get_shakespeare(root, split,args.graph.rank, download, 
#                 args.fed_personal, args.batch_size, args.rnn_seq_len)
#     else:
#         raise NotImplementedError
