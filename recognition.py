import argparse
import functools
import os

import predict
from utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/configuration.yml',   '配置文件')
add_arg('use_gpu',          bool,   True,                       '是否使用GPU预测')
add_arg('audio_db_path',    str,    'sound_database/',                '音频库的路径')
add_arg('sound_index_path',    str,    'sound_database/index.bin',                '音频库的路径')
add_arg('cos_threshold',        float,  0.4,                        '利用余弦相似度判断是否为同一个人的阈值')
add_arg('euc_threshold',        float,  10.5,                        '利用欧氏距离判断是否为同一个人的阈值')
add_arg('model_path',       str,    'train_model/EcapaTdnn_MelSpectrogram/best_model/', '使用的预测模型')
args = parser.parse_args()
print_arguments(args=args)



predictor = predict.SoundPredict(configs=args.configs,
                                 cos_threshold=args.cos_threshold,
                                 euc_threshold=args.euc_threshold,
                                 audio_db_path=args.audio_db_path,
                                 sound_index_path=args.sound_index_path,
                                 model_path=args.model_path,
                                 use_gpu=args.use_gpu)
while True:
    path = input("输入音频的绝对路径，输入quit为退出:")
    if path == "quit":
        print("已退出")
        break
    if not os.path.exists(path):
        print("路径不存在，请重新输入")
        continue
    choice = input("选择预测方法: 1.余弦相似度(不推荐), 2.欧氏距离(推荐):")
    try:
        choice = int(choice)
        if choice > 2:
            raise ValueError
    except ValueError:
        print("输入有误,请检查输入是否合法")
        continue

    name = predictor.recognition(path, choice)
    if name is not None:
        print(f"预测说话人:{name}")
    else:
        print("未识别到说话人")

