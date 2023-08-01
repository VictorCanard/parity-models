import os
from random import shuffle
import torch

import torch.utils.data as data
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder

from util.util import construct, try_cuda

from data.wikitext import get_wikitext_data




VOCAB_SIZE = 50304
SEQ_LEN = 512
BATCH_SIZE = 50


class CodeDataset(Dataset):
    """
    PyTorch dataset that groups samples from an underlying dataset together so
    that they are ready for encoding.
    """

    def __init__(self, name, base_model, num_classes, base_dataset,
                 ec_k, code_dataset=None):
        """
        Parameters
        ----------
        name: str
            One of {"train", "val", "test"}
        base_model: ``torch.nn.Module``
            Base model on which inference is being performed and over which a
            code imparts resilience.
        num_classes: int
            The number of classes in the underlying dataset.
        base_dataset: ``torchvision.datasets.Dataset``
            A dataset from the datasets provided by torchvision.
        ec_k: int
            Number of samples from ``base_dataset`` that will be encoded
            together.
        code_dataset: ``torchvision.dataset.Dataset``
            Dataset containing a set of transforms to apply to samples prior to
            encoding. These transforms may differ from those in
            `base_transform` as one may wish to include transformations such as
            random cropping and rotating of images so as to reduce overfiting.
            Such transformations would not be included in `base_dataset` as
            they could lead to noisy labels being generated.
        put_gpu: bool
            Whether to put data and labels on GPU. This is untenable for large
            datasets.
        """
        self.name = name
        self.base_model = base_model
        self.ec_k = ec_k
        self.dataset = base_dataset

        # Since we are not directly calling this DataLoader when we perform
        # iterations when training a code, it is OK not to shuffle the
        # underlying dataset.
        #dataloader = data.DataLoader(self.dataset, batch_size=50,
        #                             shuffle=False)

        # in_size = self.dataset[0][0].view(-1).size(0)
        self.num_channels = 1 #self.dataset[0][0].size(0)
        # if self.num_channels > 1:
        #     assert self.num_channels == 3, "Only currently support 3 channels for multi-channel input"

        # Preprate data, outputs from base model, and the true labels for
        # samples. We will populate these tensors so that we can later access
        # them without pulling PIL images from the underlying dataset.
        # self.data = torch.zeros(len(self.dataset), in_size)
        # self.outputs = torch.zeros(len(self.dataset), num_classes)
        # self.true_labels = torch.zeros(len(self.dataset))

        #acc_list = []
        # cur = 0
        # for inputs, targets in dataloader:
        #     inputs = try_cuda(inputs.squeeze(1).view(inputs.size(0), -1))
        #
        #     out = self.base_model.forward(inputs)
        #     acc_list.append(torch.equal(out['logits'].argmax(-1), targets).float().mean())  # changed == to torch.equal

        # last = cur + inputs.size(0)
        # self.data[cur:last, :] = inputs.data
        # self.outputs[cur:last, :] = out.data
        # self.true_labels[cur:last] = targets
        # cur = last

        # base_model_accuracy = torch.stack(acc_list).mean().item()

        # We don't print the accuracy for the validation dataset because we
        # only split the training set into a training and validation set
        # after getting all inference results from the training dataset.
        # Printing accuracy for the validation dataset can lead to confusion.
        # if name != "val":
        #     print("Base model", name, "accuracy is", str(base_model_accuracy * len(self.dataset)),
        #           "/", str(len(self.dataset)), "=", base_model_accuracy)

        # self.true_labels = self.true_labels.long()
        # if put_gpu:
        # Move data, outputs, and true labels to GPU for fast access.
        # self.data = try_cuda(self.data)
        # self.outputs = try_cuda(self.outputs)
        # self.true_labels = try_cuda(self.true_labels)

        # If extra transformations are passed, create a new dataset containing
        # these so that a caller can pull new, transformed samples with calls
        # to `__getitem__`.
        if code_dataset is not None:
            self.dataset = code_dataset
            self.extra_transforms = True
        else:
            self.extra_transforms = False

    def __getitem__(self, idx):
        # If there are extra transformations to perform, we pull directly from
        # the underlying dataset rather than from the cached `data` tensor
        # because we'd like a new sample, and extra transformations often
        # contain some random components.
        #
        # Note, however, that even though we are pulling a "new" sample from
        # the underlying dataset, we will still use the same output for the
        # sample as we calculated when we initially performed inference to get
        # the `outputs` tensor during `__init__`. This avoids having to perform
        # inference over the base model in-line with `__getitem__` calls.

        #print("went here at " + str(idx))
        # if self.extra_transforms:
        #     data, _ = self.dataset[idx]
        #     data = data.view(-1)
        # else:
        #     data = self.dataset.data[idx]

        data, target = self.dataset[idx] #Calls getitem from WikiDataset
        out = self.base_model(data)
        return torch.squeeze(data), torch.squeeze(out["logits"]), torch.squeeze(target)
        #return self.dataset[idx]

    def __len__(self):
        # Number of samples in an epoch is equal to the number of `ec_k`-sized
        # groups are contained in our dataset.
        return (len(self.dataset) // self.ec_k) * self.ec_k

    def encoder_in_dim(self):
        """
        Returns size of each input that will be given to the encoder.
        """
        return self.dataset[0].size()

    def decoder_in_dim(self):
        """
        Returns dimensionality of input that will be given to the decoder.
        """
        return VOCAB_SIZE


class DownloadCodeDataset(CodeDataset):
    """
    Wrapper class around CodeDataset for handling datasets made available
    through torchvision.
    """

    def __init__(self, name, base_model, num_classes, base_dataset,
                 base_dataset_dir, ec_k, base_transform=None,
                 code_transform=None):
        """
        Parameters (that are different from CodeDataset)
        ------------------------------------------------
        base_dataset_dir: str
            Location where ``base_dataset`` has been or will be saved. This
            avoids re-downloading the dataset.
        base_transform: ``torchvision.transforms.Transform``
            Set of transforms to apply to samples when generating base model
            outputs that will (potentially) be used as labels.
        """
        if base_transform is None:
            base_transform = transforms.ToTensor()

        # Draw from the torchvisions "train" datasets for training and
        # validation datasets
        is_train = (name != "test")

        # Create the datasets from the underlying `base_model_dataset`.
        # When generating outputs from running samples through the base model,
        # we do apply `base_transform`.
        full_base_dataset = base_dataset(root=base_dataset_dir, train=is_train,
                                         download=True, transform=base_transform)

        if code_transform is not None:
            full_code_dataset = base_dataset(root=base_dataset_dir, train=is_train,
                                             download=True, transform=code_transform)
        else:
            full_code_dataset = None

        super().__init__(name=name,
                         base_model=base_model,
                         base_dataset=full_base_dataset,
                         ec_k=ec_k,
                         num_classes=num_classes,
                         code_dataset=full_code_dataset)


class FolderCodeDataset(CodeDataset):
    """
    Wrapper class around CodeDataset for handling datasets downloaded on our
    own.
    """

    def __init__(self, name, base_model, num_classes, base_dataset_dir,
                 ec_k, base_transform=None, code_transform=None):
        """
        Parameters (that are different from CodeDataset)
        ------------------------------------------------
        base_dataset_dir: str
            Location where ``base_dataset`` has been or will be saved. This
            avoids re-downloading the dataset.
        base_transform: ``torchvision.transforms.Transform``
            Set of transforms to apply to samples when generating base model
            outputs that will (potentially) be used as labels.
        """
        if base_transform is None:
            base_transform = transforms.ToTensor()

        full_base_dataset = datasets.ImageFolder(
            root=base_dataset_dir, transform=base_transform)

        if code_transform is not None:
            full_code_dataset = datasets.ImageFolder(
                root=base_dataset_dir, transform=code_transform)
        else:
            full_code_dataset = None

        super().__init__(name=name,
                         base_model=base_model,
                         base_dataset=full_base_dataset,
                         ec_k=ec_k,
                         num_classes=num_classes,
                         code_dataset=full_code_dataset,
                         put_gpu=False)


class MNISTCodeDataset(DownloadCodeDataset):
    def __init__(self, name, base_model, ec_k, encoder_transforms):
        base_dataset = datasets.MNIST
        base_dataset_dir = "data/mnist"
        code_transform = transforms.Compose([
            *encoder_transforms,
            transforms.ToTensor()
        ])

        super().__init__(name=name,
                         base_model=base_model,
                         base_dataset=base_dataset,
                         base_dataset_dir=base_dataset_dir,
                         ec_k=ec_k, num_classes=10,
                         code_transform=code_transform)


class FashionMNISTCodeDataset(DownloadCodeDataset):
    def __init__(self, name, base_model, ec_k, encoder_transforms):
        base_dataset = datasets.FashionMNIST
        base_dataset_dir = "data/fashion-mnist"
        code_transform = transforms.Compose([
            *encoder_transforms,
            transforms.ToTensor()
        ])

        super().__init__(name=name,
                         base_model=base_model,
                         base_dataset=base_dataset,
                         base_dataset_dir=base_dataset_dir,
                         ec_k=ec_k, num_classes=10,
                         code_transform=code_transform)


class CIFARCodeDataset(DownloadCodeDataset):
    def __init__(self, name, base_model, ec_k, num_classes, encoder_transforms):
        assert num_classes == 10 or num_classes == 100

        if num_classes == 10:
            base_dataset = datasets.CIFAR10
            base_dataset_dir = "data/cifar10"
        else:
            base_dataset = datasets.CIFAR100
            base_dataset_dir = "data/cifar100"

        # For `base_transform`, we only apply normalization.
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010))])

        # We add extra transformations for CIFAR-10 as is done in:
        #   https://github.com/kuangliu/pytorch-cifar
        if name == "train":
            code_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                *encoder_transforms,
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010))])
        else:
            code_transform = transforms.Compose([
                *encoder_transforms,
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010))])

        super().__init__(name=name,
                         base_model=base_model,
                         base_dataset=base_dataset,
                         base_dataset_dir=base_dataset_dir,
                         ec_k=ec_k, num_classes=num_classes,
                         base_transform=base_transform,
                         code_transform=code_transform)


class CIFAR10CodeDataset(CIFARCodeDataset):
    def __init__(self, name, base_model, ec_k, encoder_transforms):
        super().__init__(name=name, base_model=base_model,
                         ec_k=ec_k, num_classes=10,
                         encoder_transforms=encoder_transforms)


class CIFAR100CodeDataset(CIFARCodeDataset):
    def __init__(self, name, base_model, ec_k, encoder_transforms):
        super().__init__(name=name, base_model=base_model,
                         ec_k=ec_k, num_classes=100,
                         encoder_transforms=encoder_transforms)


class CatDogCodeDataset(FolderCodeDataset):
    def __init__(self, name, base_model, ec_k, encoder_transforms):
        dataset_dir = "data/cat_v_dog/{}"
        base_dataset_dir = dataset_dir.format(name)
        num_classes = 2

        # See: https://github.com/pytorch/examples/blob/master/imagenet/main.py
        base_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

        if name == "train":
            code_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                *encoder_transforms,
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            ])
        else:
            code_transform = transforms.Compose([
                *encoder_transforms,
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            ])

        super().__init__(name=name, base_model=base_model, ec_k=ec_k,
                         base_dataset_dir=base_dataset_dir,
                         num_classes=num_classes,
                         base_transform=base_transform,
                         code_transform=code_transform)


data_dir = "data/datasets/wikitext/{}"
data_dictionary = get_wikitext_data()



class WikiText(Dataset):
    def __init__(self, name):
        # self, name, base_model, num_classes, base_dataset, ec_k,

        assert name in {'train', 'test', 'val'}

        self.seq_length = SEQ_LEN  # rename to seq len
        self.data = data_dictionary[name]

    def __getitem__(self, i):
        print("got item at " + str(i))
        x = torch.stack([torch.from_numpy((self.data[i:i + self.seq_length]).astype(np.int64))])
        y = torch.stack([torch.from_numpy((self.data[i + 1:i + 1 + self.seq_length]).astype(np.int64))])
        if torch.cuda.is_available():
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x = x.pin_memory().to('cuda:0', non_blocking=True)
            y = y.pin_memory().to('cuda:0', non_blocking=True)
        return x, y

    def __len__(self):
        #return 8
        return len(self.data) - self.seq_length


class WikiTextDataset(CodeDataset):
    def __init__(self, name, base_model, ec_k, encoder_transforms):  # Todo important to adapt for encoder_transforms?

        super().__init__(name=name, base_model=base_model, ec_k=ec_k, base_dataset=WikiText(name=name),
                         num_classes=VOCAB_SIZE)


def get_dataloaders(dataset_path, base_model, ec_k, batch_size,
                    encoder_transforms):
    """
    Generates training, validation, and test datasets.

    Parameters
    ----------
    dataset_path: str
        Classpath of underlying dataset to use.
    base_model: ``torch.nn.Module``
        Base model on which inference is being performed and over which a
        code imparts resilience.
    ec_k: int
        Number of samples from ``base_dataset`` that will be encoded
        together.
    batch_size: int
        Number of samples (group of `ec_k` inputs) to be run in a single
        minibatch.
    encoder_transforms: list
        List of transforms to be applied on inputs before being converted
        to a tensor.

    Returns
    -------
    {train, val, test}_dataloader: ``torch.utils.data.DataLoader``
        Dataloaders to be used for training, validation, and testing.
    """
    train_dataset = construct(dataset_path,
                              {"name": "train",
                               "base_model": base_model,
                               "ec_k": ec_k,
                               "encoder_transforms": encoder_transforms})

    val_dataset = construct(dataset_path,
                            {"name": "val",
                             "base_model": base_model,
                             "ec_k": ec_k,
                             "encoder_transforms": encoder_transforms})

    test_dataset = construct(dataset_path,
                             {"name": "test",
                              "base_model": base_model,
                              "ec_k": ec_k,
                              "encoder_transforms": encoder_transforms})

    # Each sample for the encoder/decoder consists of `ec_k` images from
    # the underlying dataset. Thus, the batch size for drawing samples from
    # the underlying dataset is `batch_size * ec_k`
    batch_size_for_loading = ec_k * batch_size  # todo bs here should be 50 and not 64
    if "CatDog" in dataset_path["class"] or "GCommand" in dataset_path["class"]:
        num_workers = 4
        pin_mem = torch.cuda.is_available()
        train_loader = data.DataLoader(train_dataset,
                                       batch_size=batch_size_for_loading,
                                       shuffle=True, num_workers=num_workers,
                                       pin_memory=pin_mem)
        val_loader = data.DataLoader(val_dataset,
                                     batch_size=batch_size_for_loading,
                                     shuffle=True, num_workers=num_workers,
                                     pin_memory=pin_mem)
        test_loader = data.DataLoader(test_dataset,
                                      batch_size=batch_size_for_loading,
                                      shuffle=True, num_workers=num_workers,
                                      pin_memory=pin_mem)

    else:
        #total_train = len(train_dataset)
        #indices = list(range(total_train))
        # shuffle(indices)

        print("finished shuffling indices")
        #num_val = 5000
        #remainder = num_val % ec_k
        # Make sure that the training and validation sets have a multiple of
        # ec_k.
        #if remainder != 0:
        #    num_val += (ec_k - remainder)

        #train_indices = indices[num_val:]  # todo what is this for? how can i adpat it?
        #val_indices = indices[:num_val]
        #train_sampler = data.sampler.SequentialSampler(train_indices)
        #val_sampler = data.sampler.SequentialSampler(val_indices)

        train_loader = data.DataLoader(train_dataset,
                                       shuffle=True,
                                       batch_size=batch_size_for_loading)  # set drop last to false?

        val_loader = data.DataLoader(val_dataset,
                                     shuffle=True,
                                     batch_size=batch_size_for_loading)


        test_loader = data.DataLoader(test_dataset,
                                      batch_size=batch_size_for_loading, shuffle=False)
    return train_loader, val_loader, test_loader
