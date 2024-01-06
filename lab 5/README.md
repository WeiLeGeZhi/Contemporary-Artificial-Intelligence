# 当代人工智能实验5——多模态情感分析
温兆和 10205501432

## 实验环境
由于构建的模型并不复杂，本次实验是在本地进行的。需要安装Python 3.10。

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