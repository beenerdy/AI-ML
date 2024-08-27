import argparse


def get_input_args_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_image", type=str, help="path to image to predict")
    parser.add_argument("checkpoint_path", type=str, help="path to model checkpoint")

    parser.add_argument(
        "--top_k", type=int, default=3, help="number of top_k to return"
    )
    parser.add_argument("--category_names", type=str, help="path to json cat_to_names")
    parser.add_argument("--gpu", type=str, default="True", help="use gpu to run model")
    in_arg = parser.parse_args()
    return in_arg


get_input_args_test()
