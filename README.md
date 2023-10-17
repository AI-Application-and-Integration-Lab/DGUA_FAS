# DGUA-FAS

The implementation of [Domain-Generalized Face Anti-Spoofing with Unknown Attacks](https://ieeexplore.ieee.org/abstract/document/10223078)

The architecture of the proposed DGUA-FAS method:

<div align=center>
<img src="https://github.com/AI-Application-and-Integration-Lab/DGUA_FAS/blob/master/architecture/fig.png" width="800" height="296" />
</div>

## Congifuration Environment

- python 3.9.12
- torch 1.10.0
- torchvision 0.11.1
- cuda 12.0

## Requirements

### Install MobileViT and modify base_cls.py to [our version](https://drive.google.com/file/d/1shq23SpC4X2OoYFELFjHMWpYyolmMEpj/view?usp=share_link)

```python
git clone https://github.com/apple/ml-cvnets
cd ml-cvnets
git checkout 84d992f413e52c0468f86d23196efd9dad885e6f

# replace ./cvnets/models/classification/base_cls.py to our version
pip install -r requirements.txt
pip install --editable .
pip install pandas
pip install tensorboard
cd ..
```

### Our data Pre-processing is like SSDG, so please ref their dataset setting.

```python
# After setting up the dataset path, run below codes.
cd ./data_label
python generate_label.py
```

## Training

```python
cd ./experiment/m/
python train.py
```

The file `config.py` contains all the hype-parameters used during training.

## Testing

Change the name of testing dataset in config.py and
run like this:

```python
python dg_test.py
```

We also provide our pretrained model [[Google drive]](https://drive.google.com/drive/folders/1D8WZjO62Kv4uzzNouzJWs2BrBayZq_0l?usp=sharing)

## Acknowledgment

This work can not be finished well without the following reference, many thanks for the author's contribution:

[SSDG](https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection), [ml-cvnets](https://github.com/apple/ml-cvnets), [DiVT](https://openaccess.thecvf.com/content/WACV2023/html/Liao_Domain_Invariant_Vision_Transformer_Learning_for_Face_Anti-Spoofing_WACV_2023_paper.html)

## Citation

Please cite our works if the code is helpful to your research.

```
@INPROCEEDINGS{10223078,
  author={Hong, Zong-Wei and Lin, Yu-Chen and Liu, Hsuan-Tung and Yeh, Yi-Ren and Chen, Chu-Song},
  booktitle={2023 IEEE International Conference on Image Processing (ICIP)},
  title={Domain-Generalized Face Anti-Spoofing with Unknown Attacks},
  year={2023},
  volume={},
  number={},
  pages={820-824},
  doi={10.1109/ICIP49359.2023.10223078}}
```
