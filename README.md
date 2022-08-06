# CAT-Net
This is the official repository for Compression Artifact Tracing Network (CAT-Net). Given a possibly manipulated image, this network outputs a probability map of each pixel being manipulated.
This repo provides <B>codes, pretrained/trained weights, and our five custom datasets</B>. For more details, see the papers below. 
The IJCV paper is an extension of the WACV paper and it covers almost all contents provided by the WACV paper.

* CAT-Net v1: WACV 2021 [[link to the paper]](https://openaccess.thecvf.com/content/WACV2021/html/Kwon_CAT-Net_Compression_Artifact_Tracing_Network_for_Detection_and_Localization_of_WACV_2021_paper.html)

Myung-Joon Kwon, In-Jae Yu, Seung-Hun Nam, and Heung-Kyu Lee, “CAT-Net: Compression Artifact Tracing Network for Detection and Localization of Image Splicing”, Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2021, pp. 375–384

* CAT-Net v2: International Journal of Computer Vision (IJCV), 2022 [[link to the paper]](https://link.springer.com/article/10.1007/s11263-022-01617-5) [[arXiv]](https://arxiv.org/abs/2108.12947)

Myung-Joon Kwon, Seung-Hun Nam, In-Jae Yu, Heung-Kyu Lee, and Changick Kim, “Learning JPEG Compression Artifacts for Image Manipulation Detection and Localization”, International Journal of Computer Vision, 2022, vol. 130, no. 8, pp. 1875–1895, Aug. 2022.




## Setup
##### 1. Clone this repo.

##### 2. Download the weights from: [[Google Drive Link]](https://drive.google.com/drive/folders/1hBEfnFtGG6q_srBHVEmbF3fTq0IhP8jq?usp=sharing) or [[Baiduyun Link]](https://pan.baidu.com/s/1hecZC0IZXdgh5WRbRoAytQ) (extract code: ycft).
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

v1 indicates the WACV model while v2 indicates the journal model. Both models have same architecture but the trained weights are different. v1 targets only splicing but v2 also targets copy-move forgery. If you are planning to train from scratch, you can skip downloading.

##### 3. Setup environment.
````
conda create -n cat python=3.6
conda activate cat
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
pip install -r requirements.txt
````

##### 4. Modify configuration files.
Set paths properly in 'project_config.py'.

Set settings properly in 'experiments/CAT_full.yaml'. If you are using single GPU, set GPU=(0,) not (0).


## Inference
Put input images in 'input' directory. Use English file names.

Choose between full CAT-Net and the DCT stream by commenting/uncommenting lines 65-66 and 75-76 in `tools/infer.py`. Also, choose between v1 and v2 in the lines 65-66 by modifying the strings.

At the root of this repo, run:
````
python tools/infer.py
````
The predictions are saved in 'output_pred' directory as heatmaps.

## Train
##### 1. Prepare datasets.
Obtain datasets you want to use for training.

You can download tampCOCO datasets on [[Baiduyun Link]](https://pan.baidu.com/s/1n9nN6cB0FGxsl6VH53CRwQ?pwd=ycft).

Note that tampCOCO consists of four datasets: cm_COCO, sp_COCO, bcm_COCO (=CM RAISE), bcmc_COCO (=CM-JPEG RAISE).

Also note that compRAISE is an alias of JPEG RAISE in the journal paper.

You are allowed to use the datasets for research purpose only.

[6 Aug 2022 update] Now the link changed from Google Drive to Baiduyun. 
compRAISE can be easily created by just JPEG compressing RAISE. 

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
Training starts from the pretrained weight if you place it properly.

## Licence
This code is built on top of [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation). You need to follow their licence.

For CAT-Net, you may freely use it for research purpose.

Commercial usage is strictly prohibited.



## Citation
If you use some resources provided by this repo, please cite these papers.
* CAT-Net v1 (WACV2021)
````
@inproceedings{kwon2021cat,
  title={CAT-Net: Compression Artifact Tracing Network for Detection and Localization of Image Splicing},
  author={Kwon, Myung-Joon and Yu, In-Jae and Nam, Seung-Hun and Lee, Heung-Kyu},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={375--384},
  year={2021}
}
````
* CAT-Net v2 (IJCV)
````
@article{kwon2022learning,
  title={Learning JPEG Compression Artifacts for Image Manipulation Detection and Localization},
  author={Kwon, Myung-Joon and Nam, Seung-Hun and Yu, In-Jae and Lee, Heung-Kyu and Kim, Changick},
  journal={International Journal of Computer Vision},
  volume = {130},
  number = {8},
  pages={1875--1895},
  month = aug,
  year={2022},
  publisher={Springer},
  doi = {10.1007/s11263-022-01617-5}
}
````


##### Keywords
CAT-Net, Image forensics, Multimedia forensics, Image manipulation detection, Image manipulation localization, Image processing
