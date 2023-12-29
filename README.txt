这是一个根据ECAPA-TDNN制作的声纹识别模型
介绍一下各版块
configs/ 配置文件
data_utils/ 包含对音频处理的软件包
dataset/ 训练的数据集
models/ 存放模型的软件包
sound_database/ 声纹识别库
test/ 测试模型正确率的测试集
train_model/ 存放保存的模型结果以及评估图像
utils/ 工具代码软件包

predict.py 预测模型
recognition.py 识别声音的入口
setup_data.py 启动数据集，训练模型开始前的准备工作
test_model_acc.py 测试模型的总体准确度
train.py 训练模型的入口
trainer.py 训练模型的主体

3个bat程序为启动项，在启动前应打开bat程序修改pytorch的启动环境
# 剩下的文件太大，推送不上去
