# 深度学习作业一——FashionMnist分类识别



## 任务

完成FashionMnist数据集分类识别，准确率(Accuary)达90%以上



## 工作思路

### 1.数据集分析

FashionMnist数据集包括 `60000个训练样本` 和 `10000个测试样本` ，每个样本大小为 `28*28的灰度图`。

**（*这个我就不上传GitHub了，若运行失败则在代码处将download改为True即可）**

包括 `10个类别` ：T-shirt（T恤）、Trouser（牛仔裤）、Pullover（套衫）、Dress（裙子）、Coat（外套）、Sandal（凉鞋）、Shirt（衬衫）、Sneaker（运动鞋）、Bag（包）、Ankle Boot（短靴）。

-   训练数据图片 `train-images-idx3-ubyte`
-   训练数据标签 `train-labels-idx1-ubyte`
-   测试数据图片 `t10k-images-idx3-ubyte`
-   测试数据标签 `t10k-labels-idx1-ubyte`

### 2.数据处理

对数据进行标准化操作，使用 `ComputeMeanAndStd` 计算得到均值和方差。

### 3.构建模型

-   LeNetv5

-   SimpleNet：

    ​		在LeNetv5上进行修改

    -   将conv层的channels降低
    -   同时舍弃了maxpooling
    -   全连接神经元增加

-   CNNmodel：

    ​		在LeNetv5上进行修改

    -   将conv层的channels降低
    -   添加padding
    -   使用Xavier对conv层参数进行初始化

-   ResNet18

    ​		使用Fine-Tuning操作，只对分类层进行修改，输出为10个类别。

    -   对输入数据，直接将Gray图像复制成三通道作为输入
    -   使用 `lr_scheduler `调整学习率：`step_size=7` 、`gamma=0.1`



-   优化器optimizer

    ​	采用SGD，引入牛顿动量Nesterov法：`lr=0.015`、`momentum=0.9`



### 4.实验

#### Requirements

-   python>=3.8
-   torch=1.10
-   torchvision=0.11.2
-   tqdm



#### 过程

-   Workone.ipynb

    -   加载数据（load datasets）：torchvision.datasets.FashionMNIST模块完成
    -   数据处理：构建 `ComputeMeanAndStd` 函数计算均值方差，使用torchvision.transforms.Normalize完成标准化
    -   构建模型：LeNetv5、SimpleNet、CNNmodel、ResNet18
    -   训练：损失函数 `CrossEntropyLoss` 和优化器 `SGD`
    -   测试
    -   保存模型：在 `model` 文件夹下
    -   加载模型

-   Workone_resnet：

    ​		同上，但是使用字典一并加载数据的，训练和验证写在了一个函数里面。

-   model：保存 `model.state_dict`

#### 结果

​	在`5个epoch`下各个模型的表现：

-   LeNetv5：90.82%

-   SimpleNet：88.91%
-   CNNmodel：90.56%
-   ResNet18：91.44%



### 代码

GitHub：[GuoPingPan/DeepLearning-Fashion (github.com)](https://github.com/GuoPingPan/DeepLearning-Fashion)

