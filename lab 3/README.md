# 当代人工智能实验3——图像分类及经典CNN实现
温兆和 10205501432

## 实验环境
出于实际需要，本次实验是在autodl上租来的云实例上进行的。

**云实例的具体配置情况：**
- 镜像：
	- PyTorch 2.0.0
	- Python 3.8(ubuntu 20.04)
	- Cuda 11.8
- GPU：RTX 4090(24GB) * 1
- CPU：12 vCPU Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz
- 内存：90GB
- 硬盘：
	- 系统盘：30 GB
	- 数据盘:
		- 免费:50GB
		- 付费:0GB
- 附加磁盘：无
- 端口映射：无
- 网络：同一地区实例共享带宽

**需要安装的工具包有：**
- `scikit_learn`
- ` torch`
- `torchvision`

如果需要安装这些包，可以在项目路径下执行`pip install -r requirements.txt`命令。

## 代码执行方法
在项目路径下执行`python main.py --lr <学习率> --batch_size <batch的大小> --epoch <训练周期数> --model <模型名称> --optimizer <优化器名称> --lr_decay_gamma <学习率衰减速度>`

其中， `--model`只能从`LeNet`、`AlexNet`、`ResNet`、`VGG16`和`GoogleNet`中选择，`--optimizer`只能从`SGD`和`Adam`中选择。

## 项目文件结构
```assemble
.
│   2023实验三要求.pptx
│   main.py
│   PlotDrawing.ipynb
│   README.md
│   requirements.txt
│
├───.idea
│   │   .gitignore
│   │   lab 3.iml
│   │   misc.xml
│   │   modules.xml
│   │   workspace.xml
│   │
│   └───inspectionProfiles
│           profiles_settings.xml
│           Project_Default.xml
│
├───.ipynb_checkpoints
│       Untitled-checkpoint.ipynb
│
├───10205501432_温兆和_当代人工智能实验报告3
│       10205501432_温兆和_当代人工智能实验报告3.pdf
│       AlexResult.png
│       GoogleResult.png
│       LeResult.png
│       main.tex
│       ResResult.png
│       TestRes.png
│       ValAcc.png
│
├───models
│   │   AlexNet.py
│   │   GoogleNet.py
│   │   LeNet.py
│   │   ResNet.py
│   │   VGG16.py
│   │
│   └───__pycache__
│           AlexNet.cpython-310.pyc
│           LeNet.cpython-310.pyc
│
└───train_and_test
    │   acc_rate.py
    │   train.py
    │
    └───__pycache__
            acc_rate.cpython-310.pyc
            test.cpython-310.pyc
            train.cpython-310.pyc
```