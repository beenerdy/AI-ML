from torchvision import models


def get_model_by_name(model_name):
    models_list = {"vgg16": models.vgg16, "vgg13": models.vgg13}
    return models_list[model_name]

