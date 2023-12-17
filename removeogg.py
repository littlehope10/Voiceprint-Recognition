import os

path = "dataset/voice"

labels = os.listdir(path)

null_paths = []

for label in labels:
    label_path = os.path.join(path, label).replace('\\', '/')
    voices = os.listdir(label_path)
    for voice in voices:
        full_path = os.path.join(label_path, voice).replace('\\', '/')
        if voice.find('.ogg') > -1:
            os.remove(full_path)

    new_voices = os.listdir(label_path)
    if len(new_voices) == 0:
        null_paths.append(label_path)

for label in labels:
    if label not in null_paths:
        old_path = os.path.join(path, label)
        new_label = label.replace('JP_', '')
        new_path = os.path.join(path, new_label)

        os.rename(old_path, new_path)

