import torch
import yaml
import os
import pickle
import numpy as np

from tqdm import tqdm
from utils.utils import print_arguments
from utils.utils import dict_to_object
from data_utils.featurizer import AudioFeaturizer
from data_utils.audio import AudioSegment
from models.ecapa_tdnn import EcapaTdnn
from models.res2net import Res2Net
from models.resnet_se import ResNetSE
from models.tdnn import TDNN
from models.fc import SpeakerIdetification
#from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.functional import cosine_similarity, softmax

class SoundPredict:
    def __init__(self, configs, threshold, audio_db_path, sound_index_path, model_path, use_gpu):
        self.name_dict = {}
        self.names = []
        self.feature = None
        # 已经位于index.bin中的音频路径
        self.have_loaded_sound_path = []
        # 音频库路径
        self.audio_db_path = audio_db_path
        # index.bin索引路径
        self.sound_index_path = sound_index_path

        self.cdd_num = 5

        # 加载gpu
        if use_gpu:
            assert torch.cuda.is_available(), f"GPU不可用"
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # 加载config配置
        if isinstance(configs, str):
            with open(configs, 'r', encoding='utf-8') as f:
                config = yaml.load(f.read(), Loader=yaml.FullLoader)
            print_arguments(configs=config)

        self.config = dict_to_object(config)

        # 获取所有的音频数据集
        self.all_sound_path = []

        # 加载特征器
        self.featurizer = AudioFeaturizer(feature_conf=self.config.feature_conf, **self.config.preprocess_conf)
        self.featurizer.to(self.device)

        for name in os.listdir(self.audio_db_path):
            self.names.append(name)
            name_path = os.path.join(self.audio_db_path, name).replace('\\', '/')
            for sound in os.listdir(name_path):
                sound_path = os.path.join(name_path, sound).replace('\\', '/')
                self.all_sound_path.append(sound_path)

        self.__create_index()

        # 获取模型
        if self.config.use_model == 'EcapaTdnn' or self.config.use_model == 'ecapa_tdnn':
            backbone = EcapaTdnn(input_size=self.featurizer.feature_dim, **self.config.model_conf)
        elif self.config.use_model == 'Res2Net':
            backbone = Res2Net(input_size=self.featurizer.feature_dim, **self.config.model_conf)
        elif self.config.use_model == 'ResNetSE':
            backbone = ResNetSE(input_size=self.featurizer.feature_dim, **self.config.model_conf)
        elif self.config.use_model == 'TDNN':
            backbone = TDNN(input_size=self.featurizer.feature_dim, **self.config.model_conf)
        else:
            raise Exception(f'{self.config.use_model} 模型不存在！')
        model = SpeakerIdetification(backbone=backbone, num_class=self.config.dataset_conf.num_speakers)
        model.to(self.device)

        # 加载模型
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, 'model.pt')
        assert os.path.exists(model_path), f"{model_path} 模型不存在！"
        if use_gpu:
            model_state_dict = torch.load(model_path)
        else:
            model_state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(model_state_dict)
        print(f"成功加载模型参数：{model_path}")
        model.eval()
        self.predictor = model

        self.__load_sound()


    def __create_index(self):
        assert len(self.names) > 0, "音频库无数据"
        i = 0
        for name in self.names:
            self.name_dict[i] = name
            i = i + 1


    def __load_sound(self):
        """
        加载声音库中的声音到模型中，形成一个输出结果为length的模型
        Returns: None

        """
        sounds = []
        for sound_path in tqdm(self.all_sound_path):
            sound_segment = AudioSegment.from_file(sound_path)
            assert sound_segment.duration > self.config.dataset_conf.min_length, f"{sound_path}的音频过短"

            if sound_segment.duration > self.config.dataset_conf.max_length:
                sound_segment.crop(self.config.dataset_conf.max_length)

            if sound_segment.sample_rate != self.config.dataset_conf.sample_rate:
                sound_segment.resample(self.config.dataset_conf.sample_rate)

            if self.config.dataset_conf.normalization:
                sound_segment.normalize(self.config.dataset_conf.target_dB)

            sound = sound_segment.samples
            sounds.append(sound)

            if len(sounds) == self.config.dataset_conf.num_speakers:
                if self.feature is None:
                    self.feature = self.__predict_batch(sounds)
                else:
                    new_sounds = self.__predict_batch(sounds)
                    self.feature = np.vstack((self.feature, new_sounds))
                sounds = []

        if len(sounds) != 0:
            if self.feature is None:
                self.feature = self.__predict_batch(sounds)
            else:
                new_sounds = self.__predict_batch(sounds)
                self.feature = np.vstack((self.feature, new_sounds))


    def __predict_one(self, pred_sound):
        """
        只预测一个数据
        Args:
            pred_sound: 要预测的数据

        Returns:
            tensor: (1, length)

        """
        sound_segment = AudioSegment.from_file(pred_sound)
        assert sound_segment.duration > self.config.dataset_conf.min_length, f"{pred_sound}的音频过短"

        if sound_segment.duration > self.config.dataset_conf.max_length:
            sound_segment.crop(self.config.dataset_conf.max_length)

        if sound_segment.sample_rate != self.config.dataset_conf.sample_rate:
            sound_segment.resample(self.config.dataset_conf.sample_rate)

        if self.config.dataset_conf.normalization:
            sound_segment.normalize(self.config.dataset_conf.target_dB)


        sound = sound_segment.samples
        input = np.zeros((1, sound.shape[0]), dtype='float32')
        input[0, :sound.shape[0]] = sound[:]
        sound = torch.tensor(data=input, dtype=torch.float32, device=self.device)
        label = torch.tensor([1], dtype=torch.float32,device=self.device)

        feature = self.featurizer(sound, label)
        feature = self.predictor(feature[0])
        return feature

    def __predict_batch(self, new_sounds):

        """
        预测一组数据
        Args:
            new_sounds: list: tensor(48000,)

        Returns:
            feature: tuple(2)

        """

        # 找出音频长度最长的
        batch = sorted(new_sounds, key=lambda a: a.shape[0], reverse=True)
        max_audio_length = batch[0].shape[0]
        batch_size = len(batch)
        # 以最大的长度创建0张量
        inputs = np.zeros((batch_size, max_audio_length), dtype='float32')
        lens = []
        for x in range(batch_size):
            tensor = new_sounds[x]
            seq_length = tensor.shape[0]
            # 将数据插入都0张量中，实现了padding
            inputs[x, :seq_length] = tensor[:]
            lens.append(seq_length / max_audio_length)
        sounds = torch.tensor(inputs, dtype=torch.float32,device=self.device)
        lens = torch.tensor(lens, dtype=torch.float32,device=self.device)
        audio_feature, _ = self.featurizer(sounds, lens)
        # 执行预测
        features = self.predictor(audio_feature).data.cpu().numpy()
        return features

    def __sk_retrieval(self, np_feature):
        labels = []
        for feature in self.feature:
            similarity = cosine_similarity(np_feature.reshape(-1, 1), feature.reshape(-1, 1))
            abs_similarity = np.abs(similarity)
            # 获取候选索引
            if len(abs_similarity) < self.cdd_num:
                candidate_idx = np.argpartition(abs_similarity, -len(abs_similarity))[-len(abs_similarity):]
            else:
                candidate_idx = np.argpartition(abs_similarity, -self.cdd_num)[-self.cdd_num:]
            # 过滤低于阈值的索引
            remove_idx = np.where(abs_similarity[candidate_idx] < self.threshold)
            candidate_idx = np.delete(candidate_idx, remove_idx)
            # 获取标签最多的值
            candidate_label_list = list(np.array(self.names)[candidate_idx])
            if len(candidate_label_list) == 0:
                max_label = None
            else:
                max_label = max(candidate_label_list, key=candidate_label_list.count)
            labels.append(max_label)
        return labels

    def __pt_retrieval(self, np_feature):
        np_feature = np_feature[0].data.cpu()
        feature = torch.tensor(self.feature[1])
        similarity = cosine_similarity(np_feature, feature, dim=0)
        index = np.argmax(np_feature)
        acc = np_feature[0][index]
        return self.name_dict[index], acc



    def recognition(self, audio_data, threshold=None, sample_rate=16000):
        """声纹识别
        :param audio_data: 需要识别的数据，支持文件路径，文件对象，字节，numpy。如果是字节的话，必须是完整的字节文件
        :param threshold: 判断的阈值，如果为None则用创建对象时使用的阈值
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 识别的用户名称，如果为None，即没有识别到用户
        """
        if threshold:
            self.threshold = threshold
        feature = self.__predict_one(audio_data)
        name, acc = self.__pt_retrieval(np_feature=feature)
        return name, acc

