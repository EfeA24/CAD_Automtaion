import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))

from CadSeqProc.cad_sequence import CADSequence
from CadSeqProc.utility.macro import *
from CadSeqProc.utility.logger import CLGLogger
from Cad_VLM.models.text2cad import Text2CAD
from Cad_VLM.models.utils import print_with_separator, text_prompt
from loguru import logger
from rich import print
import torch
import argparse
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
#                    Generate CAD Sequence from User Inputs                    #
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
    print_with_separator("⚡Text2CAD Test from User Input⚡")

    # --------------------------------- Argument --------------------------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default="config/inference_user_input.yaml",
    )
    parser.add_argument("--prompt", type=str, default=None)
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
        prompt=args.prompt,
        model=text2cad,
        device=device,
        log_dir=log_dir,
        config=config,
    )


def test_model(
    prompt,
    model,
    device,
    log_dir,
    config,
):
    """
    Trains a deep learning model.

    Parameters:
        prompt (str): Text prompt for CAD sequence generation.
        model (torch.nn.Module): Model to be used for inference.
        device (str): Device to be used for training.
        log_dir (str): Directory to save the results.
        config (dict): Configuration dictionary.

    Returns:
        None
    """

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print(f"Saving results in {log_dir}")
    if config["test"]["checkpoint_path"] is not None:
        checkpoint_file = config["test"]["checkpoint_path"]

        print(f"Using saved checkpoint at {checkpoint_file}")
        # checkpoint_file_name = checkpoint_file.split("/")[-1]
        checkpoint = torch.load(checkpoint_file, map_location=device)
        pretrained_dict = {}
        for key, value in checkpoint["model_state_dict"].items():
            if key.split(".")[0] == "module":
                pretrained_dict[".".join(key.split(".")[1:])] = value
            else:
                pretrained_dict[key] = value
        if "epoch" in checkpoint:
            epoch = checkpoint["epoch"]
            t2clogger.info(f"Model was trained for epoch {epoch}.")

        model.load_state_dict(pretrained_dict, strict=False)

    # ---------------------------------- Testing ---------------------------------- #
    test_acc_uid = {}

    if config["test"]["sampling_type"] == "max":
        TOPK = 1
    else:
        TOPK = 5

    # Get the text prompts
    if prompt is None:
        text = text_prompt(config["test"]["prompt_file"])
    else:
        text = [prompt]
        t2clogger.info(f"Using the user input text prompt.")

    num_texts = len(text)
    if num_texts == 0:
        raise Exception(
            f'No text found in the prompt file 😥. Please check the prompt file in {config["test"]["prompt_file"]} 🔍.'
        )
    else:
        t2clogger.info(f"Found {num_texts} prompts in the prompt file.")

    model.eval()
    batch_size=min(config["test"]["batch_size"], num_texts)
    with torch.no_grad():
        t2clogger.info("Generating CAD Sequence.")
        for b in range(num_texts // batch_size):
            # Autoregressive Generation of CAD Sequences from Text Prompts
            pred_cad_seq_dict = model.test_decode(
                texts=text[
                    b
                    * batch_size : (b + 1)
                    * batch_size
                ],
                maxlen=MAX_CAD_SEQUENCE_LENGTH,
                nucleus_prob=0,
                topk_index=TOPK,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            # Save the results batchwise
            for i in range(
                len(
                    text[
                        b
                        * batch_size : (b + 1)
                        * batch_size
                    ]
                )
            ):
                index = i + b * batch_size
                try:
                    CADSequence.from_vec(
                        pred_cad_seq_dict["cad_vec"][i].cpu().numpy(),
                        bit=N_BIT,
                        post_processing=True,
                    ).save_stp(f"pred", os.path.join(log_dir, str(index)))

                except Exception as e:
                    print(f"Invalid Model Generated for example {index}")
                # Save the model prediction output
                test_acc_uid[index] = {}
                test_acc_uid[index]["pred_cad_vec"] = (
                    pred_cad_seq_dict["cad_vec"][i].cpu().numpy()
                )
                test_acc_uid[index]["text_prompt"] = text[index]
    # Save the pkl files
    with open(log_dir + "/output.pkl", "wb") as f:
        pickle.dump(test_acc_uid, f, protocol=pickle.HIGHEST_PROTOCOL)

    t2clogger.info(f"Cad Sequence Generation Complete. Results are saved in {log_dir}.")


if __name__ == "__main__":
    main()
