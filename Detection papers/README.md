# Performance

## 1.SSD mAP
| architecture |train data| VOC07 | VOC12 | COCO with Iou=0.5(only coco train ) | 
| ---- | ---- | ---- | ----- | -------|
| Faster | VOC07_train | 69.9| - |- |
|        | VOC07+12   | 73.2| 70.4|-|
|        | 07+12+coco| 78.8| 75.9 | 45.3|
| Yolov1 | 07+12     | - | 57.9|
| SSD300 | VOC07_train | 68.0| - |-|
|        | VOC07+12   | 74.3| 72.4 |-|
|        | 07+12+coco| __79.6__| 77.5|41.2|
| SSD512 | VOC07_train | 71.6|  -|-|
|        | VOC07+12   | 76.8| 74.9|-|
|        | 07+12+coco| __81.6__|__80.0__|__46.5__|


![ssd_result](../data_images/ssd_result.png)
* Todo List
- [ ] SSD+channel pruning
- [ ] SSD+FPN
- [ ] SSD+Focal loss

## 2. MobileNet_V1
| architecture |ImageNet Accuracy| Million Mult-Adds | Million Parameters  |
| ---- | ---- | ---- | ----- |
| MobileNet_V1 | 70.6% | 569  | 4.2 |

* Todo List
- [ ] Mobilev1 + ResNet
- [ ] Mobilev1 + ResNeXt 

## 3. MobileNet_V2
* Imagenet Accuracy

![imagenet](../data_images/mobileV2_imagenet.png)

* COCO MAP

![imagenet](../data_images/mobileNetv2_coco.png)

* Todo List

## 4. R-FCN
| architecture |train data| VOC07 | VOC12 | COCO with Iou=0.5(val/test ) |  inference time (s)|
| ----         | ----     | ----  | ----- | -------       |   -  |
| RFCN         |   k=3    | 75.5  |   -   |   -           | 0.17 |
| (with atrous)|   k=7    | 76.6  |   -   |   -           | 0.17 |
| (with OHEM)  |  300 RoI | 79.5  |   -   |     -         | 0.17 |
| (multi_sc train)|07+12  | 80.5  |   -   |     _         | 0.17 |
|(multi-sc train)|07+12+coco|__83.6__ |   -   |     -         | 0.17 |
|Faster RCNN+++|  coco trainval | -  | -  |  test-55.7    | 3.36 |
| R-FCN multi-sc | coco train | - | -     |  val-49.1     | 0.17 |
| R-FCN multi-sc train,test| coco trainval |-|-| test-__53.2__| 1.00 |

* Todo List