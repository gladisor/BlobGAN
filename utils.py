import torch
from torchvision import transforms
import numpy as np
from Blob_Maker import mk_blob
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Pool
import os
import time
from PIL import Image, ImageDraw
import random

def generate_centers(num_centers: int, grid_size: float) -> np.array:
    """
    Generates evenly spaced centers on a variable size grid with a
    minimum distance of 1.0 between each point.
    """
    valid = False
    while not valid:
        centers = np.random.uniform(-grid_size, grid_size, (num_centers, 2))
        valid = True
        for i in range(num_centers):
        	for j in range(i, num_centers): # O((n-1) + (n-2) + ... + 1) Runtime complexity
        		if i != j:
        			x1, y1 = centers[i]
        			x2, y2 = centers[j]
        			d = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        			if d <= 1.0:
        				valid = False
    return centers

def create_blob_image(centers: np.array, grid_size: float, path: str):
    """
    Creates and saves an image with randomly generated blobs at each coordinate location
    for the center specified in the centers parameter.
    """
    IMAGE_SIZE = 128
    bounds = grid_size + 1

    with Image.new(mode='1', size=(IMAGE_SIZE, IMAGE_SIZE)) as im:

        draw = ImageDraw.Draw(im)
        for i in range(centers.shape[0]):
            ## Generating blob coordinates
            x, y = mk_blob(dim='2d', value=5, scale=2.5, size=2e2, sigma=3.0, plot=False)
            ## Normalizing between 0 and 1 then shifting it to the corresponding center
            x = (x - x.min()) / (x.max() - x.min()) + centers[i, 0]
            y = (y - y.min()) / (y.max() - y.min()) + centers[i, 1]
            ## Scaling coordinates between 0 and 1 again then scaling by image size
            x = ((x + bounds) / (bounds * 2)) * IMAGE_SIZE
            y = ((y + bounds) / (bounds * 2)) * IMAGE_SIZE
            ## Plotting
            xy = list(zip(x, y))
            draw.polygon(xy, fill=128)
        ## Saving
        im.save(path)

def build_dataset(images_per_class: int, blob_numbers: list, train: bool = True):
    """
    Creates the dataset of evently balanced classes. Each class is a number of blobs.
    Parallelizes the image generation across all cpu workers. If train = True, builds
    dataset under train folder. Otherwise builds it under test folder.
    """
    ## Make data folder if not exist
    path = Path().absolute()
    data_path = path / 'data'
    ## Specify train or test
    if train:
        data_path = data_path / 'train'
    else:
        data_path = data_path / 'test'

    os.makedirs(data_path, exist_ok=True)

    num_workers = mp.cpu_count() - 1
    grid_size = 2.5 ## Grid size in which to place blobs

    ## For each number of blobs
    for num_centers in blob_numbers:
        start = time.time()

        ## Create directory
        blob_path = data_path / str(num_centers)
        os.makedirs(blob_path, exist_ok=True)

        ## Create list of arguments for image generation
        num_centers = np.full(shape=images_per_class, fill_value=num_centers, dtype=np.int)
        grid_size = np.full(shape=images_per_class, fill_value=grid_size, dtype=np.float)
        file_names = [blob_path / (str(i) + '.png') for i in range(images_per_class)]

        ## Paralell pool of workers
        with Pool(num_workers) as p:
            ## Generate centers for blobs
            centers = p.starmap(generate_centers, zip(num_centers, grid_size))
            ## Create images with specified number of blobs
            p.starmap(create_blob_image, zip(centers, grid_size, file_names))

        ## Report time for generating
        elapse = time.time() - start
        print(f'{images_per_class} images of {num_centers[0]} blobs took {elapse} seconds.')

class BlobData(torch.utils.data.Dataset):
    """
    Custom dataset for blob images.
    """
    def __init__(self, path):
        super().__init__()
        x = []
        y = []

        ## Grab folders containing images of each number of blobs
        classes = os.listdir(path)
        ## Creating dict to convert number of blobs to class index
        self.class_dict = {int(k): v for v, k in enumerate(classes)}
        self.n_class = len(self.class_dict)

        for class_name in classes:
            ## Grab image paths
            image_paths = os.listdir(path / class_name)
            ## Image paths
            x += map(lambda x: path / class_name / x, image_paths)
            ## Label
            label = self.class_dict[int(class_name)]
            y += [label] * len(image_paths)

        ## Shuffling image path and label together
        xy = list(zip(x, y))
        random.shuffle(xy)
        self.x, self.y = zip(*xy)

        ## Creating transform for image
        self.transform = transforms.ToTensor()

    def __getitem__(self, idx):
        im = self.transform(Image.open(self.x[idx]))
        return im, torch.tensor(self.y[idx])

    def __len__(self):
        return len(self.x)

if __name__ == '__main__':
    blob_numbers = list(range(5, 10))

    ## Training dataset
    build_dataset(
        images_per_class=10000,
        blob_numbers=blob_numbers,
        train=True)

    ## Testing dataset
    build_dataset(
        images_per_class=1000,
        blob_numbers=blob_numbers,
        train=False)
