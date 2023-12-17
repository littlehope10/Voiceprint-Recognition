import argparse
import functools
import distutils.util
from utils.logger import setup_logger
from trainer import SoundTrainer
from utils.utils import print_arguments, add_arguments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('configs',          str,    'configs/configuration.yml',      '配置文件')
    add_arg("local_rank",       int,    0,                             '多卡训练需要的参数')
    add_arg("use_gpu",          bool,   True,                          '是否使用GPU训练')
    #add_arg('augment_conf_path',str,    'configs/augmentation.json',   '数据增强的配置文件，为json格式')
    add_arg('save_model_path',  str,    'models/',                  '模型保存的路径')
    add_arg('resume_model',     str,    None,                       '恢复训练，当为None则不使用预训练模型')
    add_arg('pretrained_model', str,    None,                       '预训练模型的路径，当为None则不使用预训练模型')
    args = parser.parse_args()

    soundtrainer = SoundTrainer(configs=args.configs, use_gpu=args.use_gpu)
    soundtrainer.train()
