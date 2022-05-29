# S2ANet
A reimplementation of the S2ANet algorithm for Oriented Object Detection

## 1. 环境依赖  
建议环境 pytorch1.7 + cuda10.2

## 2. 安装流程  
```
git clone https://github.com/chongkuiqi/S2ANet.git   
cd S2ANet  
```  

### (1)安装DOTA_devkit  
```
    sudo apt-get install swig  
    cd DOTA_devkit/polyiou  
    swig -c++ -python csrc/polyiou.i  
    python setup.py build_ext --inplace  
```

### (2)编译所需的C++库  
    python setup.py build_ext --inplace


## 3. 制作数据集  
    以DOTA数据集为例，包括数据的路径，数据切分，以及数据集的配置文件dota.yaml  
### （1）数据集的路径  
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
```  
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