import setup_data
from data_utils.reader import AudioReader
from data_utils.audio import AudioSegment

setup_data.main()

SoundData = AudioReader("dataset/train_list.txt")

sound = SoundData[1]