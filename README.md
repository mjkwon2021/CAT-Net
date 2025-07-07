# 😺 CAT-Net

Welcome to the official repository for **Compression Artifact Tracing Network (CAT-Net)**. CAT-Net specializes in detecting and localizing manipulated regions in images by analyzing compression artifacts. This repository provides **code, pretrained/trained weights, and five custom datasets** for image forensics research.

CAT-Net has two versions:
- **CAT-Net v1**: Targets only splicing forgery (WACV 2021).
- **CAT-Net v2**: Extends to general forgery types with mathematical analysis (IJCV 2022).

For more details, refer to the papers below.

---

## 📄 Papers

### CAT-Net v1: WACV 2021
- **Title**: CAT-Net: Compression Artifact Tracing Network for Detection and Localization of Image Splicing  
- **Authors**: Myung-Joon Kwon, In-Jae Yu, Seung-Hun Nam, and Heung-Kyu Lee  
- **Publication**: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2021, pp. 375–384  
- **Links**: [WACV Paper](https://openaccess.thecvf.com/content/WACV2021/html/Kwon_CAT-Net_Compression_Artifact_Tracing_Network_for_Detection_and_Localization_of_WACV_2021_paper.html)

### CAT-Net v2: IJCV 2022
- **Title**: Learning JPEG Compression Artifacts for Image Manipulation Detection and Localization  
- **Authors**: Myung-Joon Kwon, Seung-Hun Nam, In-Jae Yu, Heung-Kyu Lee, and Changick Kim  
- **Publication**: International Journal of Computer Vision, vol. 130, no. 8, pp. 1875–1895, Aug. 2022  
- **Links**: [IJCV Paper](https://link.springer.com/article/10.1007/s11263-022-01617-5), [arXiv](https://arxiv.org/abs/2108.12947)

---

## 🎨 Example Input / Output

<div style="display: flex; justify-content: space-between; gap: 5px;">
  <figure style="text-align: center; width: 400px;">
    <img src="https://github.com/mjkwon2021/CAT-Net/blob/main/github_images/example_input.jpg" width="400px">
  </figure>

  <figure style="text-align: center; width: 400px;">
    <img src="https://github.com/mjkwon2021/CAT-Net/blob/main/github_images/example_output_pred.png" width="400px">
  </figure>
</div>

---

## ⚙️ Setup

### 1. Clone this repository
```bash
   git clone https://github.com/mjkwon2021/CAT-Net.git
   cd CAT-Net
```

### 2. Download weights
Pretrained and trained weights can be downloaded from:
- [Google Drive](https://drive.google.com/drive/folders/1hBEfnFtGG6q_srBHVEmbF3fTq0IhP8jq?usp=sharing)
- [Baiduyun Link](https://pan.baidu.com/s/1hecZC0IZXdgh5WRbRoAytQ) (Extract code: `ycft`)

Place the weights as follows:
```
CAT-Net
├── pretrained_models  (pretrained weights for each stream)
│   ├── DCT_djpeg.pth.tar
│   └── hrnetv2_w48_imagenet_pretrained.pth
├── output  (trained weights for CAT-Net)
│   └── splicing_dataset
│       ├── CAT_DCT_only
│       │   └── DCT_only_v2.pth.tar
│       └── CAT_full
│           ├── CAT_full_v1.pth.tar
│           └── CAT_full_v2.pth.tar
```
- **CAT_full_v1**: WACV model (splicing forgery only).
- **CAT_full_v2**: IJCV model (splicing + copy-move forgery).

### 3. Setup environment
```bash
conda create -n cat python=3.6
conda activate cat
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
pip install -r requirements.txt
```

### 4. Modify configuration files
- Set paths in `project_config.py`.
- Update GPU settings in `experiments/CAT_full.yaml` (e.g., `GPU=(0,)` for single GPU).

---

## 🚀 Inference

### Steps
1. **Prepare Input Images**: Place images in the `input` directory. Use English filenames.
2. **Select Model and Stream**: Modify `tools/infer.py`:
   - Comment/uncomment lines 65-66 and 75-76 to select full CAT-Net or DCT stream.
   - Update lines 65-66 to select v1 or v2 weights.
3. **Run Inference**: At the root of this repository, run:
```bash
python tools/infer.py
```
4. **View Results**: Predictions are saved in the `output_pred` directory as heatmaps.

---

## 🏗️ Training

### 1. Download tampCOCO / compRAISE datasets
- **tampCOCO**: [Kaggle Link](https://www.kaggle.com/datasets/qsii24/tampcoco) or [Baiduyun Link](https://pan.baidu.com/s/1n9nN6cB0FGxsl6VH53CRwQ?pwd=ycft) (Extract code: `ycft`)
  - Contains: `cm_COCO`, `sp_COCO`, `bcm_COCO (=CM RAISE)`, `bcmc_COCO (=CM-JPEG RAISE)`.
  - Follows MS COCO licensing terms.
- **compRAISE**: [Kaggle Link](https://www.kaggle.com/datasets/qsii24/compraise)
  - Also referred to as `JPEG RAISE` in the IJCV paper.
  - Follows RAISE licensing terms.

**Note**: Use datasets for research purposes only.

#### 📦️ Other Training Dataset

Our training settings have become the standard setup for image manipulation (=forgery) detection and localization, which has been referred to as CAT-Net settings or the CAT protocol. CAT-Net settings use five datasets for training: FantasticReality, CASIA v2, IMD2020, TampCOCO, and CompRAISE. Also, the setting uses balanced sampling: see the [[__getitem__ function]](https://github.com/mjkwon2021/CAT-Net/blob/e330b5238e8f6a85133819616b24475807e97784/Splicing/data/data_core.py#L97).

So we provide links to other training datasets.

- **FantasticReality**: The official link is broken. Use: [[Link to Download]](https://github.com/mjkwon2021/CAT-Net/issues/51#issuecomment-2537517937)

- **CASIA**: The official link is broken. Use: [[Link to Download]](https://drive.google.com/drive/folders/13jyChWqg_aKMAxqj-0T2SwSxRrUP7V_X?usp=sharing)
 
- **IMD2020**: Official link: [[Link to Download]](https://staff.utia.cas.cz/novozada/db/)

---

### 2. Prepare datasets
- Obtain the required datasets.
- Configure training/validation paths in `Splicing/data/data_core.py`.
- JPEG-compress non-JPEG images before training. Run dataset-specific scripts (e.g., `Splicing/data/dataset_IMD2020.py`) for automatic compression.
- To add custom datasets, create dataset class files similar to the existing ones.

### 3. Start training
Run the following command at the root of the repository:
```bash
python tools/train.py
```
Training starts from pretrained weights if they are placed properly.

---

## 📚 Citation

If you use CAT-Net or its resources, please cite the following papers:

### CAT-Net v1 (WACV 2021)
```bibtex
@inproceedings{kwon2021cat,
  title={CAT-Net: Compression Artifact Tracing Network for Detection and Localization of Image Splicing},
  author={Kwon, Myung-Joon and Yu, In-Jae and Nam, Seung-Hun and Lee, Heung-Kyu},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={375--384},
  year={2021}
}
```

### CAT-Net v2 (IJCV 2022)
```bibtex
@article{kwon2022learning,
  title={Learning JPEG Compression Artifacts for Image Manipulation Detection and Localization},
  author={Kwon, Myung-Joon and Nam, Seung-Hun and Yu, In-Jae and Lee, Heung-Kyu and Kim, Changick},
  journal={International Journal of Computer Vision},
  volume={130},
  number={8},
  pages={1875--1895},
  month={aug},
  year={2022},
  publisher={Springer},
  doi={10.1007/s11263-022-01617-5}
}
```

---

## 🔑 Keywords
CAT-Net, Image Forensics, Multimedia Forensics, Image Manipulation Detection, Image Manipulation Localization, Image Processing

---
## 💎 Check Out SAFIRE!
I have published a new image forgery localization paper, SAFIRE: Segment Any Forged Image Region (AAAI 2025). SAFIRE can perform multi-source partitioning in addition to traditional binary prediction. Check it out on GitHub: [[SAFIRE GitHub Link]](https://github.com/mjkwon2021/SAFIRE)

