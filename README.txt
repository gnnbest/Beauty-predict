颜值预测，效果还可以，但是必须要keras版本2.1.0, tensorflow版本1.2

# 预测颜值

# 运行：python beauty_predict.py

算法详解：

step1: 利用dlib检测人脸框

step2: 利用颜值预测模型（Resnet50）预测颜值得分(1分到5分之间)

step3: 把1-5分值映射到自己想要的得分区间（这里是40 - 100分）


error: 这里代码有点问题，有的图像运行时会报错
