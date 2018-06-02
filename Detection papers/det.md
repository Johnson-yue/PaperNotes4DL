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