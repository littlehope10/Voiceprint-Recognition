import argparse
import functools
import os

import predict
from utils.utils import add_arguments, print_arguments

# 剩下的文件太大，推送不上去

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/configuration.yml',   '配置文件')
add_arg('use_gpu',          bool,   True,                       '是否使用GPU预测')
add_arg('audio_db_path',    str,    'sound_database/',                '音频库的路径')
add_arg('cos_threshold',    float,  0.25,                        '利用余弦相似度判断是否为同一个人的阈值')
add_arg('euc_threshold',    float,  8.5,                        '利用欧氏距离判断是否为同一个人的阈值')
add_arg('model_path',       str,    'train_model/EcapaTdnn_MelSpectrogram/best_model/', '使用的预测模型')
args = parser.parse_args()
print_arguments(args=args)



predictor = predict.SoundPredict(configs=args.configs,
                                 cos_threshold=args.cos_threshold,
                                 euc_threshold=args.euc_threshold,
                                 audio_db_path=args.audio_db_path,
                                 model_path=args.model_path,
                                 use_gpu=args.use_gpu)

test = "test"
name_in_db = os.listdir("sound_database")
persons = os.listdir(test)
dict = {}
cos_acc = []
euc_acc = []
for person in persons:
    sound_path = []
    dict[person] = sound_path
    person_path = os.path.join(test, person).replace('\\', '/')
    paths = os.listdir(person_path)
    for sound in paths:
        path = os.path.join(person_path, sound).replace('\\', '/')
        dict[person].append(path)

for person in persons:
    p_cos_acc = 0
    list = dict[person]
    for path in list:
        name = predictor.recognition(path, 1)
        if name is None and person not in name_in_db:
            p_cos_acc += 1
        else:
            if name == person:
                p_cos_acc += 1

    cos_acc.append(p_cos_acc / len(dict[person]))

for i in range(len(persons)):
    print(f"测试人: {persons[i]}, cos准确率: {cos_acc[i]: .3f}")
print(f"cos总准确率: {sum(cos_acc) / len(cos_acc)}")

