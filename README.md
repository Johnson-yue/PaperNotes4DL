# PaperNotes4DL
 This is markdown file and follow Deeplearning papers

# Attention

# Classification

# Object Detection
  
  This folder contains:
  ## 1. SSD
    Main work:
* One Stage的检测算法。在不同尺度上的feature map 生成不同面积和长宽比的multi-boxes。比如在conv4_3((_38*38_) 上生成4个boxes：长宽比为 {2，1/2}，面积比为{1， sqrt(skxsk-1)}
* 每个 anchor 输出的维度数是 __num_classes + 4__ , 4 means (centreX,centreY, width, height), 因此feature map上 每个cell有 _k_ 个 anchor 输出就是 k*(num_classes + 4) .
* 总的loss等于 分类的softmax loss + 回归的 smooth L1 loss
* 生成的anchor的面积由 S _k_ 决定 公式如下：
    ![sk.img](data_images/sk.png)

    在论文中 Smin=0.2， Smax=0.9 ，PriorBox 层中每一层都由一个smin和smax需要设置，根据理解，这是anchor的最小尺寸，即SSD可以检测出最小物体的尺寸为0.2*300（image_width）=60 pixels
* SSD 在训练中保持正负样本比为1：3 。其中正样本是anchor与GT的IOU大于阈值的，负样本是IOU小于阈值的。
* 在SSD中训练的类别数等于 要识别的类别数+1 ，比如voc是20，则num_classes=20+1=21. 第0类是背景类
    ![Lconf_loss](data_images/lconf.png)

* 数据增强部分，除了使用传统的crop，旋转，还是resize，将原图放大或者缩小，其他的地方补0.
* 使用了孔洞卷积(atrous)  通过修改conv layer中的dilate参数实现。扩大的kernel size=dilate*(k_size-1) + 1 使用孔洞卷积 或者stride=2 来代替pool进行downsample。

 ## 2. MobileNetv1
 * 提出了一种适用于移动端的网路结构，separable convolutions. 它是由3x3的 separable *convolution 后面接一个1x1的pointwise卷积构成。
 * separable convolution的概念就是一个Filter只处理一个channel。与正常的卷积一个Filter要处理所有的channel相比，计算量大致变为原来的1/(K_size * K_size) 。如果k_size=3，则计算量约为原来的1/8~1/9
 * separable conv与standard conv的计算量比较：

    ![Mobilev1_calc](data_images/mobileV1_calc.png)

* 在训练的过程中，使用了《Knowledge Distilling》 来提高mobileV1的精度。
* 下采用都使用的stride=2 而不是pooling layer。  
* standard conv与mobileV1的结构比较：

    ![archit](data_images/mobileV1_archit.png) 
*  separable conv 需要采用较小的weight decay （L2 norm）甚至不需要regularization ，它的时间消耗97%来源于1x1 的pointwise。
* 在人脸的网络了， 测试了FaceNet.

## 3. MobileNetv2
* 主要的两个工作：1）反残差结构，2）线性的bottleNeck，最后一个1x1的conv不加relu
* input （N，H，W，C）首先通过一个1x1的conv 来增加channel（N,H,W,t*C），然后再经过 3x3 的separable conv ，最后再通过1x1的conv 减少channel（N，H，W，C）。
* 主要网络架构：

    ![archit](data_images/mobileV2_archit.png)

* 网络中的激活函数都使用的relu6
* SSD Lite 是将SSD中所有的预测层都改成mobilev1的版本
* 通过扩大输入尺寸来提高模型的性能，MobileNetV2（1.4） 模型的性能最高其输入的分辨率为224的1.4倍。

- [*] MobileNet + ResNeXt

## 4. R-FCN
* 提供Roi pooling层是有悖自然规律的，这个paper中使用Roi-wise subnetwork来代替Roi-pool

* rfcn 使用position-sensitive 层来代替faster中的 ROI pooling。 在卷积层后面输出的是k^2（C+1） 个channel的 feature map， k是roi的spatial grid， C是分类的类别数。
* 他实验中使用的backbone是ResNet-101
* 训练中使用OHEM来训练。
* R-FCN的输入尺寸 被resize了，最短边resize到600， 每个GPU 一张图，选择128个ROI来反向传播。
* 使用了 _atrous conv_ （使用孔洞卷积）提升了2.6个mapl
* 在R-FCN中,ResNet-101（basebone）的conv5层的stride=2 改为stride=1，来增大feature map的尺度。 conv5层中所有的卷积层都改成空洞卷积（atrous）,修改前后的对比结果：
![rfcn_atrous](data_images/rfcn_atrous.png)
* R-FCN 也是把ROI区域分成kxk个区域，但是每一块对应一个channel。实现表明，k越大准确率越高。实验中k=7的时候准确率最高，但是channel的数量也会很大。
![rfcn_k](data_images/rfcn_k.png)
* 训练的时候输入尺寸从{400，500，600，700，800}随机采样，但是测试的时候固定为600
* 训练COCO的时候使用80k的train，40k的val和20k的test。前90k iter的学习率为1e-3，后30k iter 的学习率为1e-4.
![rfcn](data_images/rfcn.png)

# Face 

# GANs

# Image Retrieval

# Network Architecture

# NLP

# OCR

# Recommend

# Segmentation

# Tracker