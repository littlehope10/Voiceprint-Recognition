import argparse
import functools

import predict
from utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/configuration.yml',   '配置文件')
add_arg('use_gpu',          bool,   True,                       '是否使用GPU预测')
add_arg('audio_db_path',    str,    'sound_database/',                '音频库的路径')
add_arg('sound_index_path',    str,    'sound_database/index.bin',                '音频库的路径')
add_arg('threshold',        float,  0.6,                        '判断是否为同一个人的阈值')
add_arg('model_path',       str,    'train_model/EcapaTdnn_MelSpectrogram/best_model/', '使用的预测模型')
args = parser.parse_args()
print_arguments(args=args)


# while True:
#     path = input("输入音频的绝对路径，输入quit为退出")
predictor = predict.SoundPredict(configs=args.configs,
                                 threshold=args.threshold,
                                 audio_db_path=args.audio_db_path,
                                 sound_index_path=args.sound_index_path,
                                 model_path=args.model_path,
                                 use_gpu=args.use_gpu)
name, acc = predictor.recognition("C:\\Users\\86183\\Desktop\\zhvoice\\zhmagicdata\\5_541\\5_541_20170607160727.mp3",
                      threshold=args.threshold)
print(f"预测说话人:{name},准确率:{acc}")

