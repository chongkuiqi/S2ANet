# S2ANet

A reimplementation of the S2ANet algorithm for Oriented Object Detection

## 1. Environment dependency  

Our environment:  Ubuntu18.04 + pytorch1.7 + cuda10.2  
We don't try other environment, but we  recommend pytorch>=1.6 .

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
│   │   ├── images
│   │   ├── labelTxt
│   ├── val
│   │   ├── images
│   │   ├── labelTxt

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

## Results and Pretained weights on DOTA dataset  

**Note:**: We only use the DOTA train set, and record the mAP50 on DOTA val set.  

| Model               | Backbone |      train    |      mAP50     | Download |
| ------------------- | :------: | :-----------: | :------------: | :-----:  |
| S2ANet (paper)      | R-50-FPN | train+val set | 74.04(test set) | ------   |
| S2ANet (paper)      | R-50-FPN | train set     | 70.2(val set)  | ------   |
| S2ANet (this impl.) | R-50-FPN | train set     | 70.2(val test) | [model](https://drive.google.com/file/d/1Vb50k5zp_WyC-u5lwtN11xzgwOwhQLS_/view?usp=sharing) |
