import os
import torch
import argparse
from torch.backends import cudnn
from models.ConvIR import build_net
from train import _train
from eval import _eval
from infer import infer_single_image


def main(args):
    # CUDNN
    cudnn.benchmark = True

    if not os.path.exists("results/"):
        os.makedirs(args.model_save_dir)
    if not os.path.exists("results/" + args.model_name + "/"):
        os.makedirs("results/" + args.model_name + "/")
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = build_net(args.type)
    # print(model)

    if torch.cuda.is_available():
        model.cuda()
    if args.mode == "train":
        _train(model, args)

    elif args.mode == "test":
        _eval(model, args)

    elif args.mode == "infer":
        infer_single_image(model, args.test_model, args.input_image, args.output_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument("--model_name", default="ConvIR", type=str)
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument(
        "--mode", default="test", choices=["train", "test", "infer"], type=str
    )
    parser.add_argument(
        "--type", default="small", choices=["small", "base", "large"], type=str
    )

    # Train
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--num_epoch", type=int, default=30)
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--valid_freq", type=int, default=1)
    parser.add_argument("--resume", type=str, default="")

    # Test
    parser.add_argument("--test_model", type=str, default="")
    parser.add_argument("--save_image", type=bool, default=False, choices=[True, False])

    # Infer
    parser.add_argument("--input_image", type=str, default="")
    parser.add_argument("--output_image", type=str, default="")

    args = parser.parse_args()
    args.model_save_dir = os.path.join("results/", "ConvIR", "OTS/")
    args.result_dir = os.path.join("results/", args.model_name, "test")
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    command = "cp " + "models/layers.py " + args.model_save_dir
    os.system(command)
    command = "cp " + "models/ConvIR.py " + args.model_save_dir
    os.system(command)
    command = "cp " + "train.py " + args.model_save_dir
    os.system(command)
    command = "cp " + "main.py " + args.model_save_dir
    os.system(command)
    print(args)
    main(args)
