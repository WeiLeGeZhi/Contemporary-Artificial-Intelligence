# 当代人工智能实验4——文本摘要
温兆和 10205501432

## 实验环境
出于实际需要，本次实验是在autodl上租来的云实例上进行的。

**云实例的具体配置情况：**
- 镜像：
	- PyTorch 2.0.0
	- Python 3.8(ubuntu 20.04)
	- Cuda 11.8
- GPU：RTX 4090(24GB) * 1
- CPU：22 vCPU AMD EPYC 7T83 64-Core Processor
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
- `numpy`
- `torch`
- `pandas`
- `rouge_score`
- `tensorflow`
- `tensorflow_intel`
- `matplotlib`

如果需要安装这些包，可以在项目路径下执行`pip install -r requirements.txt`命令。

## 代码执行方法
在项目路径下执行`python main.py --lr <学习率> --batch_size <batch的大小> --epoch <训练周期数> --model <模型名称> --optimizer <优化器名称>`

其中， `--model`只能从`RNN`、`LSTM`和`GRU`中选择，`--optimizer`只能从`SGD`和`Adam`中选择。

## 项目文件结构
```assemble
.
│   2023实验四要求.pptx
│   main.py
│   README.md
│   requirements.txt
│   test.csv
│   train.csv
│
├───.idea
│   │   .gitignore
│   │   lab 4.iml
│   │   misc.xml
│   │   modules.xml
│   │   workspace.xml
│   │
│   └───inspectionProfiles
│           profiles_settings.xml
│           Project_Default.xml
│
├───10205501432_温兆和_当代人工智能实验报告4
│       10205501432_温兆和_当代人工智能实验报告4.pdf
│       GRUResult.png
│       LSTMResult.png
│       main.tex
│       RNNResult.png
│
├───models
│       GRU.py
│       LSTM.py
│       RNN.py
│
└───result
        GRUResult.png
        LSTMResult.png
        RNNResult.png
```