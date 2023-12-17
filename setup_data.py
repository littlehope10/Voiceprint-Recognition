# 这是一个示例 Python 脚本。
import os


# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
def get_data_all(voice_path):
    datas = []
    labels = os.listdir(voice_path)
    for label in labels:
        data_path = os.path.join(voice_path, label)
        data_all = os.listdir(data_path)
        for data in data_all:
            datas.append([os.path.join(data_path, data).replace('\\', '/'), label])
    return datas

def main(list_path = "dataset", voice_path = "dataset/voice"):
    datas = get_data_all(voice_path)
    train = open(os.path.join(list_path, "train_list.txt"), 'w')
    test = open(os.path.join(list_path, "test_list.txt"), 'w')
    for i, data in enumerate(datas):
        sound_path, label = data
        if i % 10 == 0:
            test.write(f"{sound_path}\t{label}\n")
        else:
            train.write(f"{sound_path}\t{label}\n")
    train.close()
    test.close()


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    main(list_path="dataset",
         voice_path="dataset/voice")

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
