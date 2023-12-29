chcp 65001
@echo off

echo 正在启动训练程序...
echo 正在准备数据集...

D:\Anaconda\envs\pytorch\python.exe setup_data.py

echo 数据集准备完成...
echo 开始训练...

D:\Anaconda\envs\pytorch\python.exe train.py

pause
# 剩下的文件太大，推送不上去