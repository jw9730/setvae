# SetVAE (PyTorch)
This repository is an official implementation of the paper:  
[**SetVAE: Learning Hierarchical Composition for Generative Modeling of Set-Structured Data**](https://arxiv.org/abs/2103.15619)  
[Jinwoo Kim*](https://bit.ly/jinwoo-kim), 
[Jaehoon Yoo*](https://github.com/Ugness), 
[Juho Lee](https://juho-lee.github.io/), 
[Seunghoon Hong](https://maga33.github.io/) (* equal contribution)  
CVPR 2021

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/setvae-learning-hierarchical-composition-for/point-cloud-generation-on-shapenet-car)](https://paperswithcode.com/sota/point-cloud-generation-on-shapenet-car?p=setvae-learning-hierarchical-composition-for)

<p align="center">
  <img src="docs/github_key.png"/>
</p>

## Installation
You can either use the Docker image ```wogns98/setvae```, or build your own Docker image using the provided Dockerfile.
```bash
docker pull wogns98/setvae:latest
```

If you don't want to use Docker, please follow bellow steps; but we highly recommend you to use Docker.
```bash
sudo apt-get update
sudo apt-get install python3.6
git clone https://github.com/jw9730/setvae.git setvae
cd setvae
pip install -r requirements.txt
bash install.sh
```

## Datasets
For MNIST and Set-MultiMNIST, the datasets will be downloaded from Torchvision server and processed automatically.

For ShapeNet, we use the processed version provided by authors of [PointFlow](https://github.com/stevenygd/PointFlow). 
Please download the dataset from this [link](https://drive.google.com/drive/folders/1G0rf-6HSHoTll6aH7voh-dXj6hCRhSAQ?usp=sharing). 
After downloading, you should update the training, evaluation, and visualization scripts accordingly as in below example:
```bash
shapenet_data_dir=".../ShapeNet/ShapeNetCore.v2.PC15k"
deepspeed train.py --shapenet_data_dir ${shapenet_data_dir} [other args]
```

## Training
You can train a SetVAE with the same setup as in our paper using the scripts in the ```scipts/``` folder.
```bash
# Train SetVAE models from scratch
bash scripts/mnist.sh
bash scripts/multimnist.sh
bash scripts/shapenet_aiplane.sh
bash scripts/shapenet_car.sh
bash scripts/shapenet_chair.sh
```

## Pre-Trained Models
To reproduce the results in our paper, download ```checkpoints.zip``` from this [link](https://drive.google.com/drive/folders/1uO_Pi96U6IUqnmxjU1gGvYrIjw8cEcTl?usp=sharing).  
Then, unzip ```checkpoints``` and place it in the project root, so that each checkpoint is located at ```checkpoints/gen/[DATASET]/camera-ready/checkpoint-[EPOCH].pt```.

Note: Although we fixed the random seed for evaluation, the results can differ across different CUDA devices and versions. 
For reproducibility, we provide the CUDA device specification and driver versions we used to produce the numbers of our paper. 
We also provide the exact samples as ```shapenet15k-samples.zip``` in this [link](https://drive.google.com/drive/folders/1uO_Pi96U6IUqnmxjU1gGvYrIjw8cEcTl?usp=sharing).

Dataset | CUDA device | CUDA driver
---|---|---|
ShapeNet-Airplane | Titan Xp with driver version 450.102.04 | 10.1.243
ShapeNet-Car | Titan Xp with driver version 450.102.04 | 10.1.243
ShapeNet-Chair | RTX 2080ti with vriver version 455.23.04 | 10.1.243

## Evaluation
After downloading the pre-trained models, you can run the scripts ```scripts/[DATASET]_test.sh``` to evaluate the ShapeNet models.
```bash
# Load a ShapeNet model checkpoint, generate point clouds, and run evaluation.
bash scripts/shapenet_airplane_test.sh
bash scripts/shapenet_chair_test.sh
bash scripts/shapenet_car_test.sh
```

## Visualization
After downloading the pre-trained models, you can run the scripts ```scrpts/[DATASET]_viz.sh``` to prepare the data for visualization.
```bash
# Load a trained model checkpoint, generate and save all the data needed for visualization under each checkpoint directory.
bash scripts/mnist_viz.sh
bash scripts/multimnist_viz.sh
bash scripts/shapenet_airplane_viz.sh
bash scripts/shapenet_char_viz.sh
bash scripts/shapenet_car_viz.sh
```

After that, see the Jupyter notebooks in ```figures/``` folder and follow their instructions to visualize the saved data.
- ```open3d_vis_samples.ipynb```: Visualize ShapeNet samples.
- ```open3d_vis_attn_[CATEGORY].ipynb```: Visualize attention on ShapeNet samples.
- ```MNIST_viz_samples.ipynb```: Visualize Set-MNIST/MultiMNIST samples.
- ```MNIST_viz_attn.ipynb```: Visualize attention on Set-MNIST/MultiMNIST samples.
- ```cardinality_generalization.ipynb```: Visualize cardinalty disentanglement and generalization.

Note: You need to install [open3d](http://www.open3d.org/docs/release/getting_started.html) for visualization of ShapeNet point clouds.

## Citation
If you find our work useful, please consider citing it:
```
@InProceedings{Kim_2021_CVPR,
    author    = {Kim, Jinwoo and Yoo, Jaehoon and Lee, Juho and Hong, Seunghoon},
    title     = {SetVAE: Learning Hierarchical Composition for Generative Modeling of Set-Structured Data},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {15059-15068}
}
```
