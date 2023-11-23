# 当代人工智能实验2——A*算法
温兆和 10205501432

## 实验环境
PyCharm 2022.1
Python 3.10

**需要安装的工具包有：**
- `queue`
- `sys`

如果有哪个工具包（假设叫`X`）需要安装，就同时按`Win`+`R`，选择`cmd`，在 Window shell 里面执
行`!pip install X`即可,或者执行`pip install -r requirements.txt`来安装这些包。

## 代码执行方法
该项目中包含`problem_1.py`和`problem_2.py`两个Python文件，分别对应实验手册中的问题一和问题二。如果要运行`problem_1.py`，就在命令行中输入`python problem_1.py {输入内容}`再按回车键即可，比如
```shell
python problem_1.py 150732684
```
如果需要运行`problem_2.py`，就先输入`python problem_2.py`，按一下回车键，在下方黏贴输入内容，再按一下回车键即可，如
```shell
python problem_2.py
5 7 3
1 2 1
1 3 4
2 4 3
3 4 2
3 5 1
4 5 2
5 1 5
```

## 项目文件结构
```shell
.
│   problem_1.py
│   problem_2.py
│   README.md
│   requirements.txt
│   实验二----Astar算法.pdf
│   实验二要求.pptx
│
├───.idea
│   │   .gitignore
│   │   lab 2.iml
│   │   misc.xml
│   │   modules.xml
│   │   workspace.xml
│   │
│   └───inspectionProfiles
│           profiles_settings.xml
│
└───10205501432_温兆和_当代人工智能实验报告2
        10205501432_温兆和_当代人工智能实验报告2.pdf
        image1.png
        image2.png
        image3.png
        image4.png
        image5.png
        image6.png
        image7.png
        image8.png
        main.tex
```
