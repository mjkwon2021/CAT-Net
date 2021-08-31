# CAT-Net
This is the official repository for Compression Artifact Tracing Network (CAT-Net). Given a possibly manipulated image, this network outputs a probability map of each pixel being manipulated.

Keywords: CAT-Net, Image forensics, Multimedia forensics, Image manipulation detection, Image manipulation localization, Image processing

* v1 (WACV2021) [[link to the paper]](https://openaccess.thecvf.com/content/WACV2021/html/Kwon_CAT-Net_Compression_Artifact_Tracing_Network_for_Detection_and_Localization_of_WACV_2021_paper.html)


Myung-Joon Kwon, In-Jae Yu, Seung-Hun Nam, and Heung-Kyu Lee, “CAT-Net: Compression Artifact Tracing Network for Detection and Localization of Image Splicing”, Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2021, pp. 375–384

* v2 (arXiv, under review) [[link to the paper]](https://arxiv.org/abs/2108.12947)


Myung-Joon Kwon, Seung-Hun Nam, In-Jae Yu, Heung-Kyu Lee, and Changick Kim, “Learning JPEG Compression Artifacts for Image Manipulation Detection and Localization,” arXiv:2108.12947 [cs, eess], Aug. 2021

## Setup
##### 1. Clone this repo.

##### 2. Download weights from [link](https://drive.google.com/drive/folders/1hBEfnFtGG6q_srBHVEmbF3fTq0IhP8jq?usp=sharing).
````
CAT-Net
├── pretrained_models  (pretrained weights for each stream)
│   ├── DCT_djpeg.pth.tar
│   └── hrnetv2_w48_imagenet_pretrained.pth
├── output  (trained weights for CAT-Net)
│   └── splicing_dataset
│       ├── CAT_DCT_only
│       │   └── DCT_only_v2.pth.tar
│       └── CAT_full
│           └── CAT_full_v1.pth.tar
│           └── CAT_full_v2.pth.tar
````
If you are trying to test the network, you only need CAT_full_v1.pth.tar or CAT_full_v2.pth.tar.

v1 indicates the WACV model while v2 indicates the journal model (on progress). If you are planning to train from scratch, you can skip downloading.

##### 3. Setup environment.
````
conda create -n cat python=3.6
conda activate cat
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
pip install -r requirements.txt
````
You need to manually install **JpegIO** from [here](https://github.com/dwgoon/jpegio). PIP installing is not supported.
After cloning JpegIO repo, go to JpegIO directory and run:
````
python setup.py install
````

##### 4. Modify configuration files.
Set paths properly in 'project_config.py'.

Set settings properly in 'experiments/CAT_full.yaml'. If you are using single GPU, set GPU=(0,) not (0).


## Inference
Put input images in 'input' directory. Use English file names.

At the root of this repo, run:
````
python tools/infer.py
````
The predictions are saved in 'output_pred' directory as heatmaps.

## Train
##### 1. Prepare datasets.
Obtain datasets you want to use for training.

Set training and validation set configuration in Splicing/data/data_core.py.

CAT-Net only allows JPEG images for training. 
So non-JPEG images in each dataset must be JPEG compressed (with Q100 and no chroma subsampling) before you start training.
You may run each dataset file (EX: Splicing/data/dataset_IMD2020.py), for automatic compression.

If you wish to add additional datasets, you should create dataset class files similar to the existing ones.

##### 2. Train.
At the root of this repo, run:
````
python tools/train.py
````
Training starts from pretrained weight if you place them properly.

## Licence
This code is built on top of [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation). You need to follow their licence.

For CAT-Net, you may freely use it for research purpose.

## Citation
* v1 (WACV2021)
````
@inproceedings{kwon2021cat,
  title={CAT-Net: Compression Artifact Tracing Network for Detection and Localization of Image Splicing},
  author={Kwon, Myung-Joon and Yu, In-Jae and Nam, Seung-Hun and Lee, Heung-Kyu},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={375--384},
  year={2021}
}
````
* v2 (arXiv, under review)
````
@article{
  title = {Learning {JPEG} Compression Artifacts for Image Manipulation Detection and Localization},
  url = {http://arxiv.org/abs/2108.12947},
  journaltitle = {{arXiv}:2108.12947 [cs, eess]},
  author = {Kwon, Myung-Joon and Nam, Seung-Hun and Yu, In-Jae and Lee, Heung-Kyu and Kim, Changick},
  date = {2021-08-29},
  eprinttype = {arxiv},
  eprint = {2108.12947},
}
````
