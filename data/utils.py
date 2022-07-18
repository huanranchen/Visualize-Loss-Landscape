import json
import os


def reverse_dic(dic):
    return dict([(v, k) for (k, v) in dic.items()])


def get_label2id(file_name):
    with open(file_name) as file:
        dict = json.load(file)

    return dict


def get_train_set_dic_track1(path):
    """
    :param path:  example './data/' last character must be '/'
    :return:
    """
    dict = {}
    list_file = os.listdir(path)
    for item in list_file:
        list_content = os.listdir(path + item)
        inside_dict = {}
        for key in list_content:
            inside_dict[key] = os.listdir(path + item + "/" + key)
        dict[item] = inside_dict

    return dict


def get_train_set_dic_track2(path):
    """
    :param path:  example './data/' last character must be '/'
    :return:
    """
    dict = {}
    list_file = os.listdir(path)
    for item in list_file:
        list_content = os.listdir(path + item)
        dict[item] = list_content
    return dict


def get_test_set_images(path):
    dir = os.listdir(path)
    return dir


def write_result(result, path="prediction.json"):
    with open(path, "w") as f:
        json.dump(result, f)
    print("managed to save result!")
    print("-" * 100)
