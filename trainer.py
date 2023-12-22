import time
import torch
import yaml
import numpy as np
import os
import platform
import shutil

from tqdm import tqdm
from models.ecapa_tdnn import EcapaTdnn
from models.fc import SpeakerIdetification
from models.loss import AAMLoss, ARMLoss, CELoss, AMLoss
from models.resnet_se import ResNetSE
from models.tdnn import TDNN
from utils.utils import print_arguments, add_arguments, dict_to_object
from data_utils.reader import AudioReader
from torch.utils.data import DataLoader
from visualdl import LogWriter
from utils.logger import setup_logger
from data_utils.featurizer import AudioFeaturizer
from models.res2net import Res2Net
from datetime import timedelta
from models.metrics import TprAtFpr


logger = setup_logger(__name__)


"""     
    设置dataloader的返回组
    并且将一组音频组合为batch*48000的tensor
    将label转化为batch长度的int类型
    延长音频并进行预处理
"""

def collate_fn(batch):
    # 找出音频长度最长的
    new_batch = sorted(batch, key=lambda audio: audio[0].samples.shape, reverse=True)
    max_audio_length = new_batch[0][0].samples.shape
    batch_size = len(new_batch)
    # 以最大的长度创建0张量
    inputs = np.zeros((batch_size, max_audio_length[0]), dtype='float32')
    input_lens_ratio = []
    labels = []
    for x in range(batch_size):
        sample = batch[x][0]
        tensor = sample.samples
        labels.append(float(batch[x][1]))
        seq_length = tensor.shape[0]
        # 将数据插入都0张量中，实现了padding
        inputs[x, :seq_length] = tensor[:]
        input_lens_ratio.append(seq_length/max_audio_length[0])
    input_lens_ratio = np.array(input_lens_ratio, dtype='float32')
    labels = np.array(labels, dtype='int64')
    return torch.tensor(inputs), torch.tensor(labels), torch.tensor(input_lens_ratio)

class SoundTrainer:
    def __init__(self, configs, use_gpu):
        if use_gpu:
            assert (torch.cuda.is_available()), 'GPU不可用'
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.use_gpu = use_gpu

        #读取配置文件
        if isinstance(configs, str):
            with open(configs, 'r', encoding='utf-8') as f:
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            print_arguments(configs=configs)

        self.config = dict_to_object(configs)

        assert self.config.use_model in ['ecapa_tdnn', 'EcapaTdnn', 'Res2Net', 'ResNetSE', 'TDNN'], "没有支持的模型"
        self.model = None
        self.test_loader = None

        # 获取特征器
        self.audio_featurizer = AudioFeaturizer(feature_conf=self.config.feature_conf, **self.config.preprocess_conf)
        self.audio_featurizer.to(self.device)

        if platform.system().lower() == 'windows':
            self.config.dataset_conf.num_workers = 0
            logger.warning('Windows系统不支持多线程读取数据，已自动关闭！')

        self.save_model_paths = []
        self.best_loss = 100.0
        self.best_acc = 0.0

        self.best_model_path = os.path.join("train_model",
                                       f'{self.config.use_model}_{self.config.preprocess_conf.feature_method}',
                                       'best_model').replace('\\', '/')
        self.model_path = os.path.join("train_model",
                                  f'{self.config.use_model}_{self.config.preprocess_conf.feature_method}').replace('\\', '/')
        if os.path.exists(self.model_path):
            paths = os.listdir(self.model_path)
            for path in paths:
                if path != "best_model":
                    self.save_model_paths.append(os.path.join(self.model_path, path).replace('\\', '/'))

    def __dataload(self, istrain = False):
        #读取训练集
        if istrain:
            self.train_data = AudioReader(data_list_path=self.config.dataset_conf.train_list,
                                          max_length=self.config.dataset_conf.max_length,
                                          min_length=self.config.dataset_conf.min_length,
                                          sample_rate=self.config.dataset_conf.sample_rate,
                                          normalization=self.config.dataset_conf.normalization,
                                          target_dB=self.config.dataset_conf.target_dB)

        self.train_loader = DataLoader(dataset=self.train_data,
                                       shuffle=True,
                                       batch_size=self.config.dataset_conf.batch_size,
                                       collate_fn=collate_fn,
                                       num_workers=self.config.dataset_conf.num_workers,
                                       drop_last=True)

        # 获取测试数据
        self.test_data = AudioReader(data_list_path=self.config.dataset_conf.test_list,
                                     max_length=self.config.dataset_conf.max_length,
                                     min_length=self.config.dataset_conf.min_length,
                                     sample_rate=self.config.dataset_conf.sample_rate,
                                     normalization=self.config.dataset_conf.normalization,
                                     target_dB=self.config.dataset_conf.target_dB,
                                     mode='eval')
        self.test_loader = DataLoader(dataset=self.test_data,
                                      batch_size=self.config.dataset_conf.batch_size,
                                      collate_fn=collate_fn,
                                      num_workers=self.config.dataset_conf.num_workers,
                                      drop_last=True)


    def __modelload(self, input_size, is_train):
        use_loss = self.config.get('use_loss', 'AAMLoss')
        # 获取模型
        if self.config.use_model == 'EcapaTdnn' or self.configs.use_model == 'ecapa_tdnn':
            backbone = EcapaTdnn(input_size=input_size, **self.config.model_conf)
        elif self.config.use_model == 'Res2Net':
            backbone = Res2Net(input_size=input_size, **self.config.model_conf)
        elif self.config.use_model == 'ResNetSE':
            backbone = ResNetSE(input_size=input_size, **self.config.model_conf)
        elif self.config.use_model == 'TDNN':
            backbone = TDNN(input_size=input_size, **self.config.model_conf)
        else:
            raise Exception(f'{self.config.use_model} 模型不存在！')

        self.model = SpeakerIdetification(backbone=backbone,
                                          num_class=self.config.dataset_conf.num_speakers,
                                          loss_type=use_loss)
        self.model.to(self.device)
        #summary(self.model, (1, 98, self.audio_featurizer.feature_dim))
        # print(self.model)
        # 获取损失函数
        if use_loss == 'AAMLoss':
            self.loss = AAMLoss()
        elif use_loss == 'AMLoss':
            self.loss = AMLoss()
        elif use_loss == 'ARMLoss':
            self.loss = ARMLoss()
        elif use_loss == 'CELoss':
            self.loss = CELoss()
        else:
            raise Exception(f'没有{use_loss}损失函数！')
        if is_train:
            # 获取优化方法
            optimizer = self.config.optimizer_conf.optimizer
            if optimizer == 'Adam':
                self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                                  lr=float(self.config.optimizer_conf.learning_rate),
                                                  weight_decay=float(self.config.optimizer_conf.weight_decay))
            elif optimizer == 'AdamW':
                self.optimizer = torch.optim.AdamW(params=self.model.parameters(),
                                                   lr=float(self.config.optimizer_conf.learning_rate),
                                                   weight_decay=float(self.config.optimizer_conf.weight_decay))
            elif optimizer == 'SGD':
                self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                                 momentum=self.config.optimizer_conf.momentum,
                                                 lr=float(self.config.optimizer_conf.learning_rate),
                                                 weight_decay=float(self.config.optimizer_conf.weight_decay))
            else:
                raise Exception(f'不支持优化方法：{optimizer}')

    def __save_model(self, save_model_path, batch_id, epoch_id, best_model=False):
        state_dict = self.model.state_dict()

        if best_model:
            model_path = self.best_model_path
        else:
            model_path = os.path.join(self.model_path, f'epoch_{epoch_id}_batch_{batch_id}').replace('\\', '/')
            if model_path not in self.save_model_paths:
                self.save_model_paths.append(model_path)

        os.makedirs(model_path, exist_ok=True)

        if len(self.save_model_paths) >= 6:
            shutil.rmtree(self.save_model_paths[0])
            del self.save_model_paths[0]


        torch.save(self.optimizer.state_dict(), os.path.join(model_path, 'optimizer.pt').replace('\\', '/'))
        torch.save(state_dict, os.path.join(model_path, 'model.pt').replace('\\', '/'))
        logger.info(f'\n已保存模型：{model_path}')

    """
            每一轮次的训练方法
            每一次训练长度为batch的音频数据
            记录训练时间，准确率及损失
    """
    def __train_epoch(self, epoch_id, save_path, writer):
        train_times, accs, loss = [], [], []
        start = time.time()
        sum_batch = len(self.train_loader) * self.config.train_conf.max_epoch

        for batch_id, (audio, label, length) in enumerate(self.train_loader):
            # print(batch_id)
            # print(f"audio = {audio}\nlabel = {label}\nlength = {length}")

            # """
            #         将数据送至显卡
            # """

            audio = audio.to(self.device)
            label = label.to(self.device).long()
            length = length.to(self.device)

            # """
            #         将音频进行一轮优化后送入模型进行训练
            # """
            features = self.audio_featurizer(audio, length)
            output = self.model(features[0])

            # 计算损失值
            los = self.loss(torch.sigmoid(output), label)
            self.optimizer.zero_grad()
            los.backward()
            self.optimizer.step()

            # 计算准确率
            output = torch.nn.functional.softmax(output, dim=-1)
            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()
            acc = np.mean((output == label).astype(int))
            accs.append(acc)
            loss.append(los)
            train_times.append((time.time() - start) * 1000)

            if batch_id % self.config.train_conf.log_interval == 0:
                # 计算每秒训练数据量
                train_speed = self.config.dataset_conf.batch_size / (sum(train_times) / len(train_times) / 1000)
                # 计算剩余时间
                eta_sec = (sum(train_times) / len(train_times)) * (
                        sum_batch - (epoch_id - 1) * len(self.train_loader) - batch_id)
                eta_str = str(timedelta(seconds=int(eta_sec / 1000)))
                logger.info('\n' + '='*50 +
                            f'\n训练轮数: [{epoch_id + 1}/{self.config.train_conf.max_epoch}], \n'
                            f'批次: [{batch_id}/{len(self.train_loader)}], \n'
                            f'损失: {sum(loss) / len(loss):.5f}, \n'
                            f'准确率: {sum(accs) / len(accs):.5f}, \n'
                            f'速度: {train_speed:.2f} data/sec, eta: {eta_str}\n' +
                            '=' * 50)

                train_times = []

            # 固定步数也要保存一次模型
            if batch_id % 30 == 0 and batch_id != 0:
                self.__save_model(save_model_path=save_path,batch_id=batch_id, epoch_id=epoch_id)
            start = time.time()



    def train(self, save_model_path = "train_model/"):
        gpus = torch.cuda.device_count()
        local = 0
        writer = None
        if local == 0:
            writer = LogWriter(logdir="log")
        self.__dataload(True)
        self.__modelload(input_size=self.audio_featurizer.feature_dim, is_train=True)
        logger.info(f"共有{len(self.train_data)}个训练集数据")

        best_eer = 1

        for epoch in range(0, self.config.train_conf.max_epoch):
            self.__train_epoch(epoch, save_model_path, writer)
            if epoch % 3 == 2 and epoch != 0:
                start_time = time.time()
                tpr, fpr, threshold, eer = self.__evaluate()
                logger.info('\n' + '=' * 70 +
                            f'测试轮: {epoch + 3}\n'
                            f'测试时间: {str(timedelta(seconds=(time.time() - start_time)))}\n'
                            f'阈值: {threshold:.2f}\n'
                            f'真阳性率: {tpr:.5f}\n'
                            f'假阳性率: {fpr:.5f},\n'
                            f'等错误率: {eer:.5f}\n' +
                            '=' * 70)

                if eer < best_eer:
                    self.__save_model(save_model_path=save_model_path, batch_id=0, epoch_id=epoch, best_model=True)
                    best_eer = eer
                else:
                    self.__save_model(save_model_path=save_model_path, batch_id=0, epoch_id=epoch)
                self.model.train()



    def __evaluate(self):
        print("开始模型测试")
        test_tprs, test_fprs, test_thresholds, test_eers = [], [], [], []
        for resume_model in self.save_model_paths:
            resume_model = os.path.join(resume_model, "model.pt").replace('\\', '/')
            assert os.path.exists(resume_model), "模型不存在"
            model_state_dict = torch.load(resume_model)
            self.model.load_state_dict(model_state_dict)
            eval_model = self.model

            print(f"当前模型:{resume_model}")
            print("开始加载测试集...")
            self.model.eval()
            features, labels = None, None
            with torch.no_grad():
                for batch_id, (audio, label, length) in enumerate(tqdm(self.test_loader)):
                    audio = audio.to(self.device)
                    length = length.to(self.device)
                    label = label.to(self.device).long()
                    audio_features, _ = self.audio_featurizer(audio, length)
                    feature = eval_model.backbone(audio_features).data.cpu().numpy()
                    label = label.data.cpu().numpy()
                    # 存放特征
                    features = np.concatenate((features, feature)) if features is not None else feature
                    labels = np.concatenate((labels, label)) if labels is not None else label

            self.model.train()
            metric = TprAtFpr()
            labels = labels.astype(np.int32)
            print('开始两两对比音频特征...')
            for i in tqdm(range(len(features))):
                feature_1 = features[i]
                feature_1 = np.expand_dims(feature_1, 0).repeat(len(features) - i, axis=0)
                feature_2 = features[i:]
                feature_1 = torch.tensor(feature_1, dtype=torch.float32)
                feature_2 = torch.tensor(feature_2, dtype=torch.float32)
                score = torch.nn.functional.cosine_similarity(feature_1, feature_2, dim=-1).data.cpu().numpy().tolist()
                y_true = np.array(labels[i] == labels[i:]).astype(np.int32).tolist()
                metric.add(y_true, score)
            tprs, fprs, thresholds, eer, index = metric.calculate()
            tpr, fpr, threshold = tprs[index], fprs[index], thresholds[index]
            test_tprs.append(tpr)
            test_fprs.append(fpr)
            test_thresholds.append(threshold)
            test_eers.append(eer)
        index = test_eers.index(min(test_eers))
        best_model = os.path.join(self.save_model_paths[index], "model.pt").replace('\\', '/')
        optimizer = os.path.join(self.save_model_paths[index], "optimizer.pt").replace('\\', '/')
        model_state_dict = torch.load(best_model)
        self.model.load_state_dict(model_state_dict)
        return test_tprs[index], test_fprs[index], test_thresholds[index], test_eers[index]

