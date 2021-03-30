"""
Multi-MNIST dataset
Mostly copy-and-paste from https://github.com/shaohua0116/MultiDigitMNIST
"""
import torch

import os
import json
from multiprocessing import Pool
import numpy as np
from imageio import imwrite, imread
from PIL import Image

from torchvision import datasets

datasets.MNIST.resources = [
    ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
]


def download_mnist(data_dir):
    datasets.MNIST(train=True, transform=None, download=True, root=data_dir)


def check_mnist_dir(data_dir):
    downloaded = np.all([os.path.isfile(os.path.join(data_dir, 'MNIST/raw', key)) for key in
                         ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 't10k-images-idx3-ubyte',
                          't10k-labels-idx1-ubyte']])
    if not downloaded:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        download_mnist(data_dir)
    else:
        print('MNIST was found')


def extract_mnist(data_dir):
    print("extract MNIST to " + data_dir)
    num_mnist_train = 60000
    num_mnist_test = 10000

    fd = open(os.path.join(data_dir, 'MNIST/raw/train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_image = loaded[16:].reshape((num_mnist_train, 28, 28, 1))

    fd = open(os.path.join(data_dir, 'MNIST/raw/train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_label = np.asarray(loaded[8:].reshape(num_mnist_train))

    fd = open(os.path.join(data_dir, 'MNIST/raw/t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_image = loaded[16:].reshape((num_mnist_test, 28, 28, 1))

    fd = open(os.path.join(data_dir, 'MNIST/raw/t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_label = np.asarray(loaded[8:].reshape(num_mnist_test))

    return train_image, train_label, test_image, test_label


def sample_coordinate(high, size):
    if high > 0:
        return np.random.randint(high, size=size)
    else:
        return np.zeros(size).astype(np.int)


def load_image(arg):
    path, label, coord = arg
    img = imread(path)
    return img, label, coord


class MultiMNIST(torch.utils.data.Dataset):
    def __init__(self, train, transform, mnist_path, root, do_generate=True,
                 num_digit=2, train_val_ratio=(6, 1), image_size=(64, 64), num_image_per_class=700, random_seed=123):
        self.train = train
        self.transform = transform
        self.mnist_path = mnist_path
        self.root = root
        self.multimnist_path = os.path.join(root, "MultiMNIST")
        if do_generate and not os.path.exists(self.multimnist_path):
            self.generate(num_digit=num_digit,
                          train_val_ratio=train_val_ratio,
                          image_size=image_size,
                          num_image_per_class=num_image_per_class,
                          random_seed=random_seed)
        self.data, self.targets, self.coords = self.extract()

    def extract(self):
        digits_path = os.path.join(self.multimnist_path, "train" if self.train else "val")
        cache_path = os.path.join(digits_path, f"cached.pth")
        json_path = os.path.join(self.multimnist_path, "train.json" if self.train else "val.json")
        with open(json_path, 'r') as f:
            metadata = json.load(f)

        paths = list()
        labels = list()
        coords = list()
        for k, v in metadata.items():
            image_name = k
            image_path = os.path.join(digits_path, image_name)
            coord = v['coord']
            label = v['label']
            paths.append(image_path)
            coords.append(coord)
            labels.append(label)

        if os.path.exists(cache_path):
            print(f'Found MultiMNIST at {cache_path}')
            return torch.load(cache_path)

        with Pool(32) as p:
            images, targets, coords = tuple(zip(*p.map(load_image, zip(paths, labels, coords))))

        def flatten(l):
            return [item for sublist in l for item in sublist]

        data = np.stack(images), np.stack(targets), np.stack(coords)  # [N, H, W], [N, 2]
        torch.save(data, cache_path)
        print(f'Saved MultiMNIST at {cache_path}')
        return data

    def generate(self, num_digit, train_val_ratio, image_size, num_image_per_class, random_seed):
        # check if mnist is downloaded. if not, download it
        print("check MNIST directory " + self.mnist_path)
        check_mnist_dir(self.mnist_path)

        # extract mnist images and labels
        train_image, train_label, test_image, test_label = extract_mnist(self.mnist_path)
        h, w = train_image.shape[1:3]  # [N, H, W, 1]

        num_original_class = len(np.unique(train_label))
        num_class = len(np.unique(train_label)) ** num_digit
        classes = list(np.array(range(num_class)))

        # split: train, val
        images = [train_image, test_image]
        labels = [train_label, test_label]
        nums_image_per_class = [
            int(float(ratio) / np.sum(train_val_ratio) * num_image_per_class)
            for ratio in train_val_ratio]
        # label index
        train_idx = [list(np.where(train_label == c)[0]) for c in range(num_original_class)]
        test_idx = [list(np.where(test_label == c)[0]) for c in range(num_original_class)]
        indexes = [train_idx, test_idx]

        # generate images for every class
        assert image_size[1] // num_digit >= w
        np.random.seed(random_seed)

        if not os.path.exists(self.multimnist_path):
            os.makedirs(self.multimnist_path)

        count = 1
        for i, split_name in enumerate(['train', 'val']):
            path = os.path.join(self.multimnist_path, split_name)
            print('Generate images for {} at {}'.format(split_name, path))
            os.makedirs(path, exist_ok=True)

            image = images[i]
            label = labels[i]
            num_image = nums_image_per_class[i]
            idx = indexes[i]
            coords = {}

            for j, current_class in enumerate(classes):
                class_str = str(current_class)
                class_str = '0' * (num_digit - len(class_str)) + class_str
                class_path = os.path.join(path, class_str)

                print('{} (progress: {}/{})'.format(class_path, count, len(classes)))

                if not os.path.exists(class_path):
                    os.makedirs(class_path)

                for k in range(num_image):
                    # sample images
                    digits = [int(class_str[l]) for l in range(num_digit)]
                    imgs = [np.squeeze(image[np.random.choice(idx[d])]) for d in digits]
                    background = np.zeros((image_size)).astype(np.uint8)  # [H, W]

                    # sample coordinates
                    ys = sample_coordinate(image_size[0] - h, num_digit)
                    xs = sample_coordinate(image_size[1] // num_digit - w, size=num_digit)
                    xs = [l * image_size[1] // num_digit + xs[l] for l in range(num_digit)]

                    # xyxy format coordinates
                    coord = np.stack((xs, ys)).transpose().reshape(-1).tolist()

                    # combine images
                    for i in range(num_digit):
                        background[ys[i]:ys[i] + h, xs[i]:xs[i] + w] = imgs[i]

                    # write the image
                    key = os.path.join(class_str, '{}_{}.png'.format(k, class_str))
                    coords[key] = {'coord': coord, 'label': [int(x) for x in class_str]}
                    image_path = os.path.join(class_path, '{}_{}.png'.format(k, class_str))
                    imwrite(image_path, background)

                count += 1
            with open(os.path.join(self.multimnist_path, f'{split_name}.json'), 'w') as f:
                json.dump(coords, f)

        return images, labels, indexes

    def __getitem__(self, index):
        img, target, coord = self.data[index], self.targets[index], self.coords[index]
        img = Image.fromarray(img, mode='L')
        if self.transform is not None:
            img = self.transform(img)
        return img, target, coord

    def __len__(self):
        return len(self.data)
