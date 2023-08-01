import argparse
import json
import os
import sys
from shutil import copyfile

import wandb

from parity_model_trainer import ParityModelTrainer

from datetime import datetime, timedelta

DEF_NUM_EPOCH = 500
DEF_LR = 1e-3
DEF_WEIGHT_DECAY = 1e-5

from datasets.code_dataset import SEQ_LEN, BATCH_SIZE


def get_config(lr, w_decay, num_epoch, ec_k, loss, encoder, decoder, base_model_file,
               base, dataset, save_dir, base_model_input_size, parity_model,
               only_test, loss_from_true_labels, cfg):
    # if ec_k == 2:
    #     mb_size = 5
    # else:
    #     mb_size = 1
    return {
        "save_dir": save_dir,
        "final_epoch": num_epoch,
        "ec_k": ec_k,
        "ec_r": 1,
        "batch_size": BATCH_SIZE,
        "only_test": only_test,

        "Loss": loss,

        "Optimizer": {
            "class": "torch.optim.Adam",
            "args": {
                "lr": lr,
                "weight_decay": w_decay
            }
        },

        "Encoder": {"class": "coders.summation." + encoder},
        "Decoder": {"class": "coders.summation." + decoder},

        "base_model_file": base_model_file,
        "base_model_input_size": base_model_input_size,
        "BaseModel": base,

        "ParityModel": parity_model,
        "Dataset": dataset,

        "loss_from_true_labels": loss_from_true_labels,
        "train_encoder": cfg["train_encoder"],
        "train_decoder": cfg["train_decoder"],
        "train_parity_model": cfg["train_parity_model"],
    }


def get_loss(loss_type, cfg):
    from_true_labels = False
    if "KLDivLoss" in loss_type: #or "CrossEntropy" in loss_type:
        if not cfg["train_encoder"] or not cfg["train_decoder"]:
            raise Exception(
                "{} currently only supported for learned encoders and decoders".format(loss_type))

    if "CrossEntropy" in loss_type:
        from_true_labels = True

    return {"class": "torch.nn." + loss_type}, from_true_labels


def get_base_model(dataset, base_model_type):
    base_path = "base_model_trained_files"

    model_file = os.path.join(
        base_path, dataset, base_model_type, "model.t7")

    num_classes = 10
    input_size = None

    if base_model_type == "base-mlp": #mnist
        base = {
            "class": "base_models.base_mlp.BaseMLP"
        }
        input_size = [-1, 784]
    elif base_model_type == "resnet18":
        if dataset == "cifar10":
            base = {
                "class": "base_models.resnet.ResNet18",
                "args": {
                    "size_for_cifar": True
                }
            }
            input_size = [-1, 3, 32, 32]
        elif dataset == "cifar100":
            base = {
                "class": "base_models.resnet.ResNet18",
                "args": {
                    "size_for_cifar": True,
                    "num_classes": 100
                }
            }
            input_size = [-1, 3, 32, 32]
        elif dataset == "cat_v_dog":
            base = {
                "class": "torchvision.models.resnet18",
                "args": {
                    "pretrained": False,
                    "num_classes": 2
                }
            }
            input_size = [-1, 3, 224, 224]

        else: #mnist
            base = {
                "class": "base_models.resnet.ResNet18",
                "args": {
                    "size_for_cifar": False
                }
            }
            input_size = [-1, 1, 28, 28]
    elif base_model_type == "resnet152":
        assert dataset == "cifar100", "ResNet152 only used for CIFAR-100"
        base = {
            "class": "base_models.resnet.ResNet152",
            "args": {
                "size_for_cifar": True,
                "num_classes": 100
            }
        }
        input_size = [-1, 3, 32, 32]
    elif base_model_type == "vgg11":
        assert dataset == "gcommands", "VGG currently only used for GCommands"
        base = {
            "class": "base_models.vgg.VGG11",
            "args": {
                "num_classes": 30
            }
        }
        input_size = [-1, 1, 161, 101]
    elif base_model_type == "lenet":
        assert dataset == "gcommands", "LeNet currently only used for GCommands"
        base = {
            "class": "base_models.lenet.LeNet",
            "args": {
                "num_classes": 30
            }
        }
        input_size = [-1, 1, 161, 101]

    elif base_model_type == "gpt2":
        assert dataset == "wikitext", "GPT2 currently only used for WikiText"
        base = {
            "class": "base_models.gpt2.GPTBase",
        }
        input_size = [-1, SEQ_LEN] #Todo what dims for this? batch size = 50 x seq len = 512
    else:
        raise Exception("Invalid base_model_type: {}".format(base_model_type))
    if dataset == "mnist":
        ds = {
            "class": "datasets.code_dataset.MNISTCodeDataset",
        }
    elif dataset == "fashion-mnist":
        ds = {
            "class": "datasets.code_dataset.FashionMNISTCodeDataset",
        }
    elif dataset == "cifar10":
        ds = {
            "class": "datasets.code_dataset.CIFAR10CodeDataset",
        }
    elif dataset == "cifar100":
        ds = {
            "class": "datasets.code_dataset.CIFAR100CodeDataset",
        }
    elif dataset == "cat_v_dog":
        ds = {
            "class": "datasets.code_dataset.CatDogCodeDataset",
        }
    elif dataset == "gcommands":
        ds = {
            "class": "datasets.gcommands_dataset.GCommandsCodeDataset",
        }
    elif dataset == "wikitext":
        ds = {
            "class": "datasets.code_dataset.WikiTextDataset",
        }
    else:
        raise Exception("Unrecognized dataset name '{}'".format(dataset))
    return model_file, base, input_size, ds


def get_parity_model(dataset, parity_model_type):
    if parity_model_type == "base-mlp":
        parity_model = {
            "class": "base_models.base_mlp.BaseMLP"
        }
        input_size = [-1, 784]
    elif parity_model_type == "resnet18":
        if dataset == "cifar10":
            parity_model = {
                "class": "base_models.resnet.ResNet18",
                "args": {
                    "size_for_cifar": True
                }
            }
            input_size = [-1, 3, 32, 32]
        elif dataset == "cat_v_dog":
            parity_model = {
                "class": "torchvision.models.resnet18",
                "args": {
                    "pretrained": False,
                    "num_classes": 2
                }
            }
            input_size = [-1, 3, 224, 224]
        else:
            parity_model = {
                "class": "base_models.resnet.ResNet18",
                "args": {
                    "size_for_cifar": False
                }
            }
            input_size = [-1, 1, 28, 28]
    elif parity_model_type == "resnet152":
        assert dataset == "cifar100", "ResNet152 only used for CIFAR-100"
        parity_model = {
            "class": "base_models.resnet.ResNet152",
            "args": {
                "size_for_cifar": True,
                "num_classes": 100
            }
        }
        input_size = [-1, 3, 32, 32]
    elif parity_model_type == "vgg11":
        assert dataset == "gcommands", "VGG currently only used for GCommand"
        parity_model = {
            "class": "base_models.vgg.VGG11",
            "args": {
                "num_classes": 30
            }
        }
        input_size = [-1, 1, 161, 101]
    elif parity_model_type == "lenet":
        assert dataset == "gcommands", "LeNet currently only used for GCommands"
        parity_model = {
            "class": "base_models.lenet.LeNet",
            "args": {
                "num_classes": 30
            }
        }
        input_size = [-1, 1, 161, 101]
    elif parity_model_type == "gpt2":
        assert dataset == "wikitext", "GPT2 currently only used for WikiText"
        parity_model = {
            "class": "base_models.gpt2.GPTBase"
        }
        input_size = [-1, SEQ_LEN]
    else:
        raise Exception("Unrecognized parity_model_type '{}'".format(parity_model_type))
    return parity_model, input_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str,
                        help="JSON file containing configuration parameters")
    parser.add_argument("overall_save_dir", type=str,
                        help="Directory to save logs and models to")
    parser.add_argument("--continue_from_file",
                        help="Path to file containing previous training state.")
    parser.add_argument("--checkpoint_cycle", type=int, default=1,
                        help="Number of epochs between model checkpoints")
    parser.add_argument("--only_test", action="store_true",
                        help="Run only the test. --continue_from_file option "
                             "must also be set")
    args = parser.parse_args()

    with open(args.config_file, 'r') as infile:
        cfg = json.load(infile)

    if not os.path.isdir(args.overall_save_dir):
        os.makedirs(args.overall_save_dir)

    unique_time = (datetime.utcnow() + timedelta(hours=2)).strftime(
        "%d_%m_%Y__%H_%M_%S")  # My computer has a 2-hour time delay for some reason

    tr_enc = cfg["train_encoder"]
    tr_dec = cfg["train_decoder"]
    tr_parity = cfg["train_parity_model"]

    # The following params have default values
    num_epoch = DEF_NUM_EPOCH
    lr = DEF_LR
    wd = DEF_WEIGHT_DECAY

    if "num_epoch" in cfg:
        num_epoch = cfg["num_epoch"]
    if "lr" in cfg:
        lr = cfg["lr"]
    if "weight_decay" in cfg:
        wd = cfg["weight_decay"]

    ##Wandb
    use_wandb = False
    if "use_wandb" in cfg:
        use_wandb = cfg["use_wandb"]

    # Before the training and testing phases
    print("Unique Time Identifier: " + unique_time)
    print("Encoder Training: " + str(tr_enc))
    print("Decoder Training: " + str(tr_dec))
    print("Parity Model Training: " + str(tr_parity))
    print("Number of Epochs: " + str(num_epoch))
    print("Learning Rate: " + str(lr))
    print("Weight Decay: " + str(wd))

    print()


    for dataset in cfg["datasets"]:
        for base_type, parity_type in cfg["base_parity_models"]:
            for ec_k in cfg["k_vals"]:
                for loss_type in cfg["losses"]:
                    for enc, dec in cfg["enc_dec_types"]:
                        print(dataset, base_type, parity_type,
                              ec_k, loss_type, enc, dec)
                        loss, loss_from_true_labels = get_loss(loss_type, cfg)
                        model_file, base, input_size, ds = get_base_model(
                            dataset, base_type)
                        parity_model, pm_input_size = get_parity_model(
                            dataset, parity_type)

                        suffix_dir = os.path.join(dataset,
                                                  "E_{}".format(tr_enc)[0:3],
                                                  "D_{}".format(tr_dec)[0:3],
                                                  "P_{}".format(tr_parity)[0:3],
                                                  "b_{}".format(base_type),
                                                  "p_{}".format(parity_type),
                                                  "k{}".format(ec_k),
                                                  "lr{}".format(lr),
                                                  "w{}".format(wd),
                                                  "ep{}".format(num_epoch),
                                                  "{}".format(loss_type),
                                                  "{}".format(enc),
                                                  "{}".format(dec))

                        save_dir = os.path.join(
                            args.overall_save_dir, suffix_dir)
                        config_map = get_config(lr, wd, num_epoch, ec_k, loss, enc,
                                                dec, model_file, base,
                                                ds, save_dir, input_size,
                                                parity_model, args.only_test,
                                                loss_from_true_labels,
                                                cfg)

                        ckpt_path = save_dir #os.path.join(args.results_base_folder, args.dataset, args.model, exp_name)
                        if not os.path.exists(ckpt_path):
                            os.makedirs(ckpt_path)
                        elif os.path.isfile(
                                os.path.join(ckpt_path, "summary.json")):  # the experiment was already completed
                            print(f"Already found experiment '{ckpt_path}'.\nSkipping.")
                            sys.exit(0)

                        if args.continue_from_file:
                            config_map["continue_from_file"] = args.continue_from_file

                        else:
                            possible_file = os.path.join(save_dir, "current.pth")
                            if os.path.exists(possible_file):
                                print("found prev file")
                                config_map["continue_from_file"] = possible_file

                        try:
                            trainer = ParityModelTrainer(config_map,
                                                         checkpoint_cycle=args.checkpoint_cycle)
                            if use_wandb:
                                exp_name = f"Encoder{tr_enc}_Decoder{tr_dec}_Par{tr_parity}_lr{lr}_bs{BATCH_SIZE}x{SEQ_LEN}"

                                wandb.init(project=cfg["wandb_project"], name=exp_name, config=cfg)

                            stats = trainer.train(wandb=use_wandb)
                            with open(f"{ckpt_path}/summary.json", "w") as fs:
                                json.dump(stats, fs)
                        except KeyboardInterrupt:
                            print("INTERRUPTED")
