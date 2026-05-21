import json

def read_config(path):
    """"读取配置"""
    with open(path) as json_file:
        config = json.load(json_file)
    return config
