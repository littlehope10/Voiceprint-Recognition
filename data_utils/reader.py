from torch.utils.data import Dataset
from data_utils.audio import AudioSegment


class AudioReader(Dataset):
    def __init__(self,
                 data_list_path,
                 max_length=3,
                 min_length=0.5,
                 augmentation_config='{}',
                 mode='train',
                 sample_rate=16000,
                 normalization=True,
                 target_dB=-20):
        """音频数据加载器

        Args:
            data_list_path: 包含音频路径和标签的数据列表文件的路径
            max_duration: 最长的音频长度，大于这个长度会裁剪掉
            min_duration: 过滤最短的音频长度
            augmentation_config: 用于指定音频增强的配置
            mode: 数据集模式。在训练模式下，数据集可能会进行一些数据增强的预处理
            sample_rate: 采样率
            normalization: 是否对音频进行音量归一化
            target_dB: 音量归一化的大小
        """
        with open(data_list_path, 'r') as f:
            self.datas = f.readlines()
        self.max_length = max_length
        self.min_length = min_length
        self.augmentation_config = augmentation_config
        self.mode = mode
        self.sample_rate = sample_rate
        self.normalization = normalization
        self.target_dB = target_dB

    def __getitem__(self, idx):
        data_path, label = self.datas[idx].replace('\n', '').split('\t')

        #获取声音
        sound = AudioSegment.from_file(data_path)

        #声音太小则不训练
        if self.mode == 'train':
            if sound.duration < self.min_length:
                return self.__getitem__(idx=idx + 1 if idx < len(self.datas) - 1 else 0)

        #重采样
        if sound.sample_rate != self.sample_rate:
            sound.resample(self.sample_rate)

        #正则化
        if self.normalization:
            sound.normalize(target_db=self.target_dB)

        #裁剪
        sound.crop(duration=self.max_length, mode=self.mode)

        return sound, label

    def __len__(self):
        return len(self.datas)

