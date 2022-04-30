# PoseEstimationDataMark
一款人体姿态标注工具

- dataMark.py 人体姿态估计标注工具，会生成基于coco的标注格式
- HRNetMark.py 在HRNet的网络中替换掉demo文件，生成一个HRNet预估计的csv文件，放入数据集与图片同目录下即可，运行方式同demo.py，加入参数 (--inputFile 图片集目录)
