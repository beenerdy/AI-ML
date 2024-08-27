import sys, time, threading
import torch
from torchvision import datasets
from torch import nn
from torch import optim


from get_input_args_train import get_input_args_train
from workspace_utils import active_session
from get_transforms import get_test_transforms, get_train_transforms
from get_model_by_name import get_model_by_name;


def animated_loading(id, stop):
    chars = "/—|"
    print("Epoch", id)
    while True:
        for char in chars:
            sys.stdout.write("\r" + "|" + char + "| ")
            time.sleep(0.1)
            sys.stdout.flush()

        if stop():
            break


def get_device(use_gpu):
    return torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")


def train(model, data_dir, device, learning_rate=0.001, hidden_units=1000, epochs=6):
    batch_size = 64
    start_time = time.time()
    train_transforms = get_train_transforms()
    test_transforms = get_test_transforms()

    train_datasets = datasets.ImageFolder(
        data_dir + "/train", transform=train_transforms
    )
    trainloader = torch.utils.data.DataLoader(
        train_datasets, batch_size=batch_size, shuffle=True
    )
    test_datasets = datasets.ImageFolder(data_dir + "/test", transform=test_transforms)
    testloader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch_size, shuffle=True
    )

    model.classifier = nn.Sequential(
        nn.Linear(25088, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 500),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(500, 102),
        nn.LogSoftmax(dim=1),
    )

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device)

    running_loss = 0
    testing_loss = 0
    test_accuracy = 0
    training_step = 0
    testing_step = 0

    total_iterations = (len(trainloader) + len(testloader)) * epochs

    stop_thread_training_animation = False
    stop_thread_testing_animation = False

    with active_session():
        for epoch in range(1, epochs + 1):
            training_animation_thread = threading.Thread(
                target=animated_loading,
                args=(epoch, lambda: stop_thread_training_animation),
            )
            testing_animation_thread = threading.Thread(
                target=animated_loading,
                args=(epoch, lambda: stop_thread_testing_animation),
            )
            stop_thread_training_animation = False
            training_animation_thread.start()
            model.train()
            for inputs, labels in trainloader:
                total_steps = training_step + testing_step
                print(
                    "{:.3f}% |training| steps: {}/{}".format(
                        total_steps * 100 / total_iterations,
                        total_steps,
                        total_iterations,
                    )
                )
                training_step += 1
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                logps = model.forward(inputs)
                loss = criterion(logps, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            stop_thread_training_animation = True

            accuracy = 0
            model.eval()
            with torch.no_grad():
                stop_thread_testing_animation = False
                testing_animation_thread.start()
                for inputs, labels in testloader:
                    total_steps = training_step + testing_step
                    print(
                        "{:.3f}% |testing| steps: {}/{}".format(
                            total_steps * 100 / total_iterations,
                            total_steps,
                            total_iterations,
                        )
                    )
                    testing_step += 1
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    loss = criterion(logps, labels)

                    testing_loss += loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                stop_thread_testing_animation = True

            print(f"Training Loss {running_loss/training_step}")
            print(f"Validation Loss {testing_loss/training_step}")
            print(f"Accuracy {accuracy/len(testloader)}")

    end_time = time.time()
    tot_time = (
        start_time - end_time
    )  # calculate difference between end time and start time
    print(
        "\n** Total Elapsed Runtime:",
        str(int((tot_time / 3600)))
        + ":"
        + str(int((tot_time % 3600) / 60))
        + ":"
        + str(int((tot_time % 3600) % 60)),
    )
    return optimizer, train_datasets


def save_checkpoint(model_name, model, optimizer, train_datasets, epochs, save_dir):
    PATH = save_dir + "checkpoint.pth"
    checkpoint = {
        "train_images_class_idx": train_datasets.class_to_idx,
        "epochs": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_name": model_name,
        "classifier": model.classifier,
    }
    torch.save(checkpoint, PATH)
    print("checkpoint saved")


if __name__ == "__main__":
    in_arg = get_input_args_train()
    model = get_model_by_name(in_arg.arch)
    model = model(pretrained=True)

    run_on_gpu = True
    if in_arg.gpu == "False":
        run_on_gpu = False
    device = get_device(run_on_gpu)
    optimizer, train_dataset = train(
        model,
        in_arg.data_dir,
        device,
        in_arg.learning_rate,
        in_arg.hidden_units,
        in_arg.epochs,
    )
    save_checkpoint(
        in_arg.arch, model, optimizer, train_dataset, in_arg.epochs, in_arg.save_dir
    )
