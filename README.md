# S2ANet

A reimplementation of the S2ANet algorithm for Oriented Object Detection.

**Note:** Support **DDP training** and **Auto Mixed Precision** in Pytorch, so the training is faster!

## 1. Environment dependency  

Our environment:  Ubuntu18.04 + pytorch1.7 + cuda10.2  
We don't try other environment, but we  recommend pytorch>=1.6 for DDP training.

## 2. Installation

### (1)Clone the S2ANet repository

```bash
git clone https://github.com/chongkuiqi/S2ANet.git   
cd S2ANet  
```  

### (2)Install DOTA_devkit  

```bash
sudo apt-get install swig  
cd DOTA_devkit/polyiou  
swig -c++ -python csrc/polyiou.i  
python setup.py build_ext --inplace  
```

### (3)Compilate the C++ Cuda Library

```bash
cd S2ANet
python setup.py build_ext --inplace
```

## 3. Prepare datasets  

We take the DOTA dataset for example.

### （1）Dataset folder structure

We should download the DOTA dataset, then change the folder structure like so:  

```
your_dir
├── DOTA
│   ├── train
│   │   ├── images
│   │   ├── labelTxt
│   ├── val
│   │   ├── images
│   │   ├── labelTxt
```

### (2) Split images and and convert annotations format  

The original images in DOTA have so large size, so we need to split them into chip images, like this:  
**note:** the DOTA path and save path in `1_prepare_dota1_ms.py` should be changed.  

```bash
cd S2ANet/DOTA_devkit/
python 1_prepare_dota1_ms.py
```

Then convert the DOTA annotations to yolo format.  
**note:** the DOTA_split path in `2_convert_dota_to_yolo.py` should be changed.

```bash
python 2_convert_dota_to_yolo.py
```

Besides, `val_split.txt` is needed for evulate the model, this file records image names without extension.
**note:** the path in `3_create_txts.py` should be changed.

```bash
python 3_create_txt.py
```

### (3) Config dota.yaml  

The [dota.yaml](data/dota.yaml) is also needed.
**note:** the path in `dota.yaml` should be changed.

Finally, the folder structure will be like this:

```
your_dir
├── DOTA
│   ├── train
│   │   ├── images
│   │   ├── labelTxt
│   ├── val
│   │   ├── images
│   │   ├── labelTxt
├── DOTA_split
│   ├── train
│   │   ├── empty_images
│   │   ├── empty_labels
│   │   ├── images
│   │   ├── labels
│   │   ├── labelTxt
│   ├── val
│   │   ├── images
│   │   ├── labels
│   │   ├── labelTxt
│   │   ├── val_split.txt
```

## 4. Train S2ANet model  

### (1) Single-GPU

```bash
cd S2ANet
python train.py
```

### (2) Mutil-GPU

**Note:** We support pytorch Multi-GPU DistributedDataParallel Mode !  

```bash
python -m torch.distributed.launch --nproc_per_node 2 train.py --device 0,1
```

## 5. Results and trained weights on DOTA dataset  

**Note:** We only use the DOTA train set, and record the mAP50 on DOTA val set.  

| Model               | Backbone |      train    |      mAP50     | Download |
| ------------------- | :------: | :-----------: | :------------: | :-----:  |
| S2ANet (paper)      | R-50-FPN | train+val set | 74.04(test set) | ------   |
| S2ANet (paper)      | R-50-FPN | train set     | 70.2(val set)  | ------   |
| S2ANet (this impl.) | R-50-FPN | train set     | 70.2(val test) | [model](https://drive.google.com/file/d/1Vb50k5zp_WyC-u5lwtN11xzgwOwhQLS_/view?usp=sharing) |

## 6.Refenerce

- (1) [Offical_S2ANet](https://github.com/csuhan/s2anet.git)
- (2) [YOLOv5](https://github.com/ultralytics/yolov5.git)
