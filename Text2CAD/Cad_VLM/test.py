import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))

from CadSeqProc.cad_sequence import CADSequence
from CadSeqProc.utility.macro import *
from CadSeqProc.utility.utils import chamfer_dist, normalize_pc
from CadSeqProc.utility.logger import CLGLogger
from Cad_VLM.models.text2cad import Text2CAD
from Cad_VLM.models.utils import print_with_separator
from Cad_VLM.dataprep.t2c_dataset import get_dataloaders
from loguru import logger
from rich import print
import torch
import argparse
from tqdm import tqdm
import os
import datetime
import argparse
import yaml
import warnings
import logging.config
import pickle

warnings.filterwarnings("ignore")
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)

t2clogger = CLGLogger().configure_logger(verbose=True).logger

# ---------------------------------------------------------------------------- #
#                              Text2CAD Test Code                              #
# ---------------------------------------------------------------------------- #


def parse_config_file(config_file):
    with open(config_file, "r") as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data


def save_yaml_file(yaml_data, filename, output_dir):
    with open(os.path.join(output_dir, filename), "w+") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)


@logger.catch()
def main():
    print_with_separator("😊 Text2CAD Inference 😊")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default="config/inference.yaml",
    )
    args = parser.parse_args()
    config = parse_config_file(args.config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t2clogger.info(
        "Current Device {}",
        torch.cuda.get_device_properties(device),
    )

    # -------------------------------- Load Model -------------------------------- #
    cad_config = config["cad_decoder"]
    cad_config["cad_seq_len"] = MAX_CAD_SEQUENCE_LENGTH
    text2cad = Text2CAD(text_config=config["text_encoder"], cad_config=cad_config).to(
        device
    )

    dim = config["cad_decoder"]["cdim"]
    nlayers = config["cad_decoder"]["num_layers"]
    ca_level_start = config["cad_decoder"]["ca_level_start"]

    # --------------------------- Prepare Log Directory -------------------------- #
    now = datetime.datetime.now()
    time_str = now.strftime("%H:%M")
    date_str = datetime.date.today()
    log_dir = os.path.join(
        config["test"]["log_dir"],
        f"{date_str}/{time_str}_d{dim}_nl{nlayers}_ca{ca_level_start}",
    )
    t2clogger.info(
        "Current Date {date_str} Time {time_str}\n",
        date_str=date_str,
        time_str=time_str,
    )

    # Create the log dir if it doesn't exist
    if not config["debug"]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        save_yaml_file(
            config, filename=args.config_path.split("/")[-1], output_dir=log_dir
        )

    # -------------------------------- Train Model ------------------------------- #

    test_model(
        model=text2cad,
        device=device,
        log_dir=log_dir,
        config=config,
        logger=t2clogger,
    )


def test_model(
    model,
    device,
    log_dir,
    config,
    logger,
):
    """
    Trains a deep learning model.

    Parameters:
        model (torch.nn.Module): The neural network model.
        device (str): Device to train on ('cuda' for GPU, 'cpu' for CPU).
        log_dir (str): Directory to save logs and checkpoints.
        config (dict): Additional configuration parameters.

    Returns:
        None
    """

    # Create the dataloader for train
    test_loader = get_dataloaders(
        cad_seq_dir=config["test_data"]["cad_seq_dir"],
        prompt_path=config["test_data"]["prompt_path"],
        split_filepath=config["test_data"]["split_filepath"],
        subsets=["test"],
        batch_size=config["test"]["batch_size"],
        num_workers=config["test"]["num_workers"],
        pin_memory=True,
        shuffle=False,  # If curriculum learning is enabled, set to False else it will automatically shuffle
        prefetch_factor=config["test"]["prefetch_factor"],
    )[0]

    if not config["debug"]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    logger.info(f"Saving results in {log_dir}.")
    if config["test"]["checkpoint_path"] is not None:
        checkpoint_file = config["test"]["checkpoint_path"]

        print(f"Using saved checkpoint at {checkpoint_file}.")
        checkpoint_file_name = checkpoint_file.split("/")[-1]
        checkpoint = torch.load(checkpoint_file, map_location=device)

        if "epoch" in checkpoint:
            print(f"Model was trained for epoch {checkpoint['epoch']}.")

        pretrained_dict = {}
        for key, value in checkpoint["model_state_dict"].items():
            if key.split(".")[0] == "module":
                pretrained_dict[".".join(key.split(".")[1:])] = value
            else:
                pretrained_dict[key] = value

        model.load_state_dict(pretrained_dict, strict=False)
        if not config["debug"]:
            save_yaml_file(
                config,
                filename=f"config_{checkpoint_file_name.split('.')[0]}.yaml",
                output_dir=log_dir,
            )

    # ---------------------------------- Inference ---------------------------------- #
    test_acc_uid = {}
    model.eval()
    if config["test"]["sampling_type"] == "max":
        TOPK = 1
    else:
        TOPK = 5

    with torch.no_grad():
        with tqdm(test_loader, ascii=True, desc=f"Inference✨") as pbar:
            for uid_level, vec_dict, prompt, _ in pbar:
                for key, value in vec_dict.items():
                    vec_dict[key] = value.to(device)

                for topk_index in range(1, TOPK + 1):
                    # Autoregressive Prediction (5 outputs per sample)
                    pred_cad_seq_dict = model.test_decode(
                        texts=prompt,
                        maxlen=MAX_CAD_SEQUENCE_LENGTH,
                        nucleus_prob=0,
                        topk_index=topk_index,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                    )

                    # Save the results batchwise
                    for i in range(vec_dict["cad_vec"].shape[0]):
                        uid, level = uid_level[i].split("_")
                        # if topk_index == 1:
                        if uid not in test_acc_uid:
                            test_acc_uid[uid] = {}

                        if level not in test_acc_uid[uid]:
                            test_acc_uid[uid][level] = {}  # {"response":response}

                        is_invalid = 0
                        try:
                            gt_cad = (
                                CADSequence.from_vec(
                                    vec_dict["cad_vec"][i].cpu().numpy(),
                                    bit=N_BIT,
                                    post_processing=True,
                                )
                                .create_cad_model()
                                .sample_points(n_points=8192)
                            )
                        except:
                            continue

                        try:
                            pred_cad = (
                                CADSequence.from_vec(
                                    pred_cad_seq_dict["cad_vec"][i].cpu().numpy(),
                                    bit=N_BIT,
                                    post_processing=True,
                                )
                                .create_cad_model()
                                .sample_points(n_points=8192)
                            )

                        except Exception as e:
                            is_invalid = 1
                            pred_cad = None

                        # Save the model prediction output
                        try:
                            test_acc_uid[uid][level]["pred_cad_vec"].append(
                                pred_cad_seq_dict["cad_vec"][i].cpu().numpy()
                            )
                        except:
                            test_acc_uid[uid][level]["pred_cad_vec"] = [
                                pred_cad_seq_dict["cad_vec"][i].cpu().numpy()
                            ]
                            # Adding Ground Truth Label
                            test_acc_uid[uid][level]["gt_cad_vec"] = (
                                vec_dict["cad_vec"][i].cpu().numpy()
                            )
                            test_acc_uid[uid][level]["cd"] = []

                        # If the model is valid, add the chamfer distance (Multiplied by 1000)
                        if is_invalid == 0:
                            cd = (
                                chamfer_dist(
                                    normalize_pc(gt_cad.points),
                                    normalize_pc(pred_cad.points),
                                )
                                * 1000
                            )
                        else:  # If the model is invalid, -1 chamfer distance (will be filtered in the evaluation stage)
                            cd = -1

                        test_acc_uid[uid][level]["cd"].append(cd)

                        pbar.set_postfix({"uid": uid, "cd": cd})

                        test_acc_uid[uid][level]["is_invalid"] = is_invalid

                # if not config['debug']:
                #     # Save the pkl files
                #     with open(log_dir+"/output.pkl", "wb") as f:
                #         pickle.dump(test_acc_uid, f,
                #                     protocol=pickle.HIGHEST_PROTOCOL)

    if not config["debug"]:
        # Save the pkl files (overwrites the previous file)
        with open(log_dir + "/output.pkl", "wb") as f:
            pickle.dump(test_acc_uid, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.success("Inference Complete")


if __name__ == "__main__":
    main()
