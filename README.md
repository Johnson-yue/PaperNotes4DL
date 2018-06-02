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
 * 


# Face 

# GANs

# Image Retrieval

# Network Architecture

# NLP

# OCR

# Recommend

# Segmentation

# Tracker