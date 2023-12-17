import os
import json

def keystoint(x):
    return {int(k): v for k, v in x.items()}

class JsonName:
    def __init__(self, name_path = "configs/name.json"):
        self.name_path = name_path
        self.json_data = json.loads(name_path, object_hook=keystoint)

    def return_name(self, target):
        return self.json_data[target]

# path = "dataset/voice"
# labels = os.listdir(path)
# i = 0
# for label in labels:
#     old_path = os.path.join(path, label).replace('\\', '/')
#     new_path = os.path.join(path, str(i)).replace('\\', '/')
#     i = i + 1
#     os.rename(old_path, new_path)



