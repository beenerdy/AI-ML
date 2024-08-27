import argparse


def get_input_args_train():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir", type=str, help="path to the folder of flowers images"
    )

    parser.add_argument(
        "--save_dir", type=str, default="./", help="path to chekpoint file"
    )
    parser.add_argument(
        "--arch", type=str, default="vgg16", help="feature architecture"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument("--hidden_units", type=int, default=1000, help="hidden units")
    parser.add_argument("--epochs", type=int, default=6, help="epochs")
    parser.add_argument("--gpu", type=str, default="True", help="use gpu to run model")
    in_arg = parser.parse_args()
    return in_arg


get_input_args_train()
