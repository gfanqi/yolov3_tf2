# HUMAN_BEHAVIOR_YOLOV3_TF2

#### 介绍
基于SSD300的口罩检测模型。

#### 安装

```
$pip install -r requirements.txt

```

#### 使用

```
下载voc数据集
更改 ./core/config 中的配置信息
使用 ./data_process/get_data_from_intenet.py 下载图片，并自行标注
使用 ./data_process/parse_VOC 解析数据，可以使用第一个函数来解析自己的数据，或用后面几个数据获得从voc数据集中得到的任务动作数据。
使用show_image 展示数据，并验证生成的数据文件有没有问题，有问题手动调一下。
使用train.py 来训练数据
```

#### Reference:

####  https://github.com/YunYang1994/TensorFlow2.0-Examples

#### ,
效果：

#### ![Snipaste_2020-03-10_17-12-04](.\imgs\Snipaste_2020-03-10_17-12-04.jpg)

 ![Snipaste_2020-03-10_17-12-11](.\imgs\Snipaste_2020-03-10_17-12-11.jpg)





![Snipaste_2020-03-10_17-12-23](D:\projects\TensorFlow2.0-Examples\ObjectDetection_TF2\HumanBehaviorDetecor\imgs\Snipaste_2020-03-10_17-12-23.jpg) 