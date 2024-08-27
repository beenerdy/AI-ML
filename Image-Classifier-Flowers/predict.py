import json

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch import nn

from get_input_args_test import get_input_args_test
from get_model_by_name import get_model_by_name


def preprocess_image(image_pil):
    """Scales, crops, and normalizes a PIL image for a PyTorch model, returns
    an Numpy array."""
    image_pil.thumbnail((256, 256))
    width, height = image_pil.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = left + 224
    bottom = top + 224
    image_pil = image_pil.crop((left, top, right, bottom))
    np_image = np.array(image_pil) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    return torch.from_numpy(np_image).type(torch.FloatTensor)


with open("cat_to_name.json", "r") as f:
    cat_to_name = json.load(f)


def classifier(img_path, model, device, top_k=3):
    img_pil = Image.open(img_path)
    img_tensor = preprocess_image(img_pil)

    in_arg = get_input_args_test()

    img_tensor.unsqueeze_(0)
    img_tensor.requires_grad = False
    img_tensor = img_tensor.to(device)

    model = model.eval()
    output = torch.exp(model(img_tensor))
    top_p, top_k = output.topk(top_k, dim=1)
    top_p = top_p.data.to(device).numpy().tolist()[0]
    top_k = top_k.data.to(device).numpy().tolist()[0]

    return top_p, top_k


def get_device(use_gpu):
    return torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")


def load_checkpoint(PATH, device):
    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = "cpu"

    checkpoint = torch.load(PATH, map_location=map_location)
    model = get_model_by_name(checkpoint["model_name"])
    model = model(pretrained=True)
    model.classifier = checkpoint["classifier"]

    model.load_state_dict(checkpoint["model_state_dict"])
    model.class_to_idx = checkpoint["train_images_class_idx"]
    epochs = checkpoint["epochs"]
    model.eval()

    model.to(device)
    return model


def get_top_names(CAT_TO_NAME_PATH, class_to_idx, top_k):
    with open(CAT_TO_NAME_PATH, "r") as f:
        cat_to_name = json.load(f)
        classes = []
        for top_klass in top_k:
            for k, v in model.class_to_idx.items():
                if v == top_klass:
                    classes.append(k)
        return [cat_to_name[str(_class)] for _class in classes]


if __name__ == "__main__":
    in_arg = get_input_args_test()

    run_on_gpu = True
    if in_arg.gpu == "False":
        run_on_gpu = False
    device = get_device(run_on_gpu)

    model = load_checkpoint(in_arg.checkpoint_path, device)
    top_p, top_k = classifier(in_arg.path_to_image, model, device, in_arg.top_k)
    print(f"top_p: {top_p}")
    print(f"top_k: {top_k}")

    if in_arg.category_names:
        flower_num = in_arg.path_to_image.split("/")[3]
        title_ = cat_to_name[flower_num]
        print("Real label: {}".format(title_))

        labels = get_top_names(in_arg.category_names, model.class_to_idx.items(), top_k)
        print(f"labels: {labels}")
