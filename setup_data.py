# 这是一个示例 Python 脚本。
import os


# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
def rename():
    path = "dataset/voice"
    names = os.listdir(path)
    num_path = []
    else_path = []
    for name in names:
        try:
            to_num = int(name)
            if to_num >= 0:
                num_path.append(to_num)
            else:
                else_path.append(name)
        except ValueError:
            else_path.append(name)

    i = 0
    num_path.sort()
    for num in num_path:
        if num != i:
            old_path = os.path.join(path, str(num)).replace('\\', '/')
            new_path = os.path.join(path, str(i)).replace('\\', '/')
            os.rename(old_path, new_path)
        i = i + 1

    for else_p in else_path:
        old_path = os.path.join(path, else_p).replace('\\', '/')
        new_path = os.path.join(path, str(i)).replace('\\', '/')
        os.rename(old_path, new_path)
        i = i + 1

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
    rename()
    datas = get_data_all(voice_path)
    train = open(os.path.join(list_path, "train_list.txt"), 'w')
    test = open(os.path.join(list_path, "test_list.txt"), 'w')
    for i, data in enumerate(datas):
        sound_path, label = data
        if i % 20 == 0:
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
