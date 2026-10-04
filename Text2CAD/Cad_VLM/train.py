import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))

import random
import numpy as np
from CadSeqProc.utility.macro import *
from CadSeqProc.utility.logger import CLGLogger
from Cad_VLM.models.text2cad import Text2CAD
from Cad_VLM.models.loss import CELoss
from Cad_VLM.models.metrics import AccuracyCalculator
from Cad_VLM.models.utils import print_with_separator
from Cad_VLM.dataprep.t2c_dataset import get_dataloaders
from loguru import logger
import torch
import argparse
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from torch.optim.lr_scheduler import ExponentialLR
import datetime
import gc
import argparse
import yaml
import warnings
import logging.config

warnings.filterwarnings("ignore")
logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)

t2clogger = CLGLogger().configure_logger(verbose=True).logger

# ---------------------------------------------------------------------------- #
#                            Text2CAD Training Code                            #
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
    print_with_separator("😊 Text2CAD Training 😊")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default="config/trainer.yaml",
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

    # Freeze the base text embedder
    for param in text2cad.base_text_embedder.parameters():
        param.requires_grad = False

    # text2cad = torch.nn.DataParallel(
    #     text2cad
    # )  # For Parallel Processing (during Training)

    optimizer = optim.AdamW(text2cad.parameters(), lr=config["train"]["lr"])
    scheduler = ExponentialLR(optimizer, gamma=0.999)
    criterion = CELoss(device=device)

    lr = config["train"]["lr"]
    dim = config["cad_decoder"]["cdim"]
    nlayers = config["cad_decoder"]["num_layers"]
    batch = config["train"]["batch_size"]
    ca_level_start = config["cad_decoder"]["ca_level_start"]

    # --------------------------- Prepare Log Directory -------------------------- #
    now = datetime.datetime.now()
    time_str = now.strftime("%H:%M")
    date_str = datetime.date.today()
    log_dir = os.path.join(
        config["train"]["log_dir"],
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

    train_model(
        model=text2cad,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        log_dir=log_dir,
        num_epochs=config["train"]["num_epochs"],
        checkpoint_name=f"lr{lr}_d{dim}_nl{nlayers}_b{batch}_ca{ca_level_start}",
        config=config,
    )


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    device,
    log_dir,
    num_epochs,
    checkpoint_name,
    config,
):
    """
    Trains a deep learning model.

    Parameters:
        model (torch.nn.Module): The neural network model.
        criterion: Loss function.
        optimizer: Optimization algorithm.
        scheduler: Learning rate scheduler.
        device (str): Device to train on ('cuda' for GPU, 'cpu' for CPU).
        log_dir (str): Directory to save logs and checkpoints.
        num_epochs (int): Number of training epochs.
        checkpoint_name (str): Name to save the checkpoints.
        config (dict): Additional configuration parameters.

    Returns:
        None
    """

    # Create the dataloader for train
    train_loader, val_loader = get_dataloaders(
        cad_seq_dir=config["train_data"]["cad_seq_dir"],
        prompt_path=config["train_data"]["prompt_path"],
        split_filepath=config["train_data"]["split_filepath"],
        subsets=["train", "validation"],
        batch_size=config["train"]["batch_size"],
        num_workers=min(config["train"]["num_workers"], os.cpu_count()),
        pin_memory=True,
        shuffle=False,  # If curriculum learning is enabled, set to False else it will automatically shuffle
        prefetch_factor=config["train"]["prefetch_factor"],
    )

    tensorboard_dir = os.path.join(log_dir, f"summary")
    # ---------------------- Resume Training from checkpoint --------------------- #
    checkpoint_file = os.path.join(log_dir, f"t2c_{checkpoint_name}.pth")
    checkpoint_only_model_file = os.path.join(
        log_dir, f"t2c_{checkpoint_name}_model.pth"
    )

    if config["train"]["checkpoint_path"] is None:
        old_checkpoint_file = checkpoint_file
    else:
        old_checkpoint_file = config["train"]["checkpoint_path"]

    if os.path.exists(old_checkpoint_file):
        t2clogger.info("Using saved checkpoint at {}", old_checkpoint_file)
        checkpoint = torch.load(old_checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch",0) + 1
        step = checkpoint.get("step",0)
    else:
        step = 0
        start_epoch = 1
    t2clogger.info("Saving checkpoint at {}", checkpoint_file)

    # Create the tensorboard summary writer
    writer = SummaryWriter(log_dir=tensorboard_dir, comment=f"{checkpoint_name}")
    
    model = torch.nn.DataParallel(
        model
    )  # For Parallel Processing (during Training)

    # ---------------------------------- Training ---------------------------------- #
    if start_epoch > config["train"]["curriculum_learning_epoch"]:
        t2clogger.warning("MIXED LEARNING...")
        random.shuffle(train_loader.dataset.uid_pair)
    else:
        t2clogger.info("CURRICULUM LEARNING...")

    # model=torch.compile(model)
    # Start training
    model.train()
    for epoch in range(start_epoch, num_epochs + 1):
        # ------------------------------- Single Epoch ------------------------------- #
        # Shuffle the data when curriculum learning stops
        if epoch == config["train"]["curriculum_learning_epoch"]:
            t2clogger.info("MIXED LEARNING...")
            optimizer = optim.AdamW(model.parameters(), lr=config["train"]["lr"])
            scheduler = ExponentialLR(optimizer, gamma=0.99)

        if epoch >= config["train"]["curriculum_learning_epoch"]:
            # Note: Works as dataloader(shuffle=True) for the current epoch
            random.shuffle(train_loader.dataset.uid_pair)

        # Train for one epoch
        train_loss = []
        train_loss_seq = {"seq": []}
        train_accuracy_seq = {"seq": []}
        val_accuracy_seq = {"seq": []}

        with tqdm(
            train_loader,
            ascii=True,
            desc=f"\033[94mText2CAD\033[0m: Epoch [{epoch}/{num_epochs+1}]✨",
        ) as pbar:
            for _, vec_dict, prompt, mask_cad_dict in pbar:
                step += 1

                for key, value in vec_dict.items():
                    vec_dict[key] = value.to(device)

                for key, value in mask_cad_dict.items():
                    mask_cad_dict[key] = value.to(device)

                # Padding mask for predicted Cad Sequence
                shifted_key_padding_mask = mask_cad_dict["key_padding_mask"][:, 1:]
                # Create Label for Training
                cad_vec_target = vec_dict["cad_vec"][:, 1:].clone()

                # ------------------ Forward pass by Teacher Forcing Method ------------------ #

                # Create training input by removing the last token
                for key, value in vec_dict.items():
                    vec_dict[key] = value[:, :-1]

                # Padding mask for input Cad Sequence
                mask_cad_dict["key_padding_mask"] = mask_cad_dict["key_padding_mask"][
                    :, :-1
                ]

                # Output from the model
                cad_vec_pred, _ = model(
                    vec_dict=vec_dict,
                    texts=prompt,
                    mask_cad_dict=mask_cad_dict,
                    metadata=False,
                )  # (B,N1,2,C1)

                # ----------------------------- Loss Calculation ----------------------------- #
                loss, loss_sep_dict = criterion(
                    {
                        "pred": cad_vec_pred,
                        "target": cad_vec_target,
                        "key_padding_mask": ~shifted_key_padding_mask,
                    }
                )

                # ------------------------------- Backward pass ------------------------------ #
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(), max_norm=0.9, norm_type=2.0
                )
                optimizer.step()

                # --------------------------- Log loss and accuracy -------------------------- #
                train_loss.append(loss.item())
                train_loss_seq["seq"].append(loss_sep_dict["loss_seq"])

                # Compute accuracy
                cad_accuracy = AccuracyCalculator(
                    discard_token=len(END_TOKEN)
                ).calculateAccMulti2DFromProbability(cad_vec_pred, cad_vec_target)

                # Add Accuracy report
                train_accuracy_seq["seq"].append(cad_accuracy)
                pbar_keys = ["Loss", "seq"]

                updated_dict = {
                    key: (
                        np.round(train_loss[-1], decimals=2)
                        if key == "Loss"
                        else np.round(train_accuracy_seq[key.lower()][-1], decimals=2)
                    )
                    for key in pbar_keys
                }
                # Update the progress bar
                pbar.set_postfix(updated_dict)

                # ---------------------------- Add to Tensorboard ---------------------------- #

                if not config["debug"]:
                    # Add Losses
                    writer.add_scalar(
                        "Seq Loss (Train)",
                        np.mean(train_loss_seq["seq"]),
                        step,
                        new_style=True,
                    )

                    # Add Accuracies
                    writer.add_scalar(
                        "Seq Accuracy (Train)",
                        np.mean(train_accuracy_seq["seq"]),
                        step,
                        new_style=True,
                    )

                    writer.add_scalar(
                        "Total Train Loss", np.mean(train_loss), step, new_style=True
                    )

        # Perform Validation
        val_cad_acc = validation_one_epoch(
            val_loader=val_loader,
            model=model,
            epoch=epoch,
            num_epochs=num_epochs,
            writer=writer,
            config=config,
            total_batch=config["val"]["val_batch"],
        )
        val_accuracy_seq["seq"].append(val_cad_acc)
        # ---------------- Save the model weights and optimizer state ---------------- #
        if not config["debug"]:
            # Save checkpoints
            if epoch % config["train"]["checkpoint_interval"] == 0:

                # Save only the model weights
                checkpoint_only_model_file = os.path.join(
                    log_dir, f"t2c_{checkpoint_name}_{epoch}_model.pth"
                )
                # torch.save(
                #     {
                #         "epoch": epoch,
                #         "model_state_dict": model.state_dict(),
                #         "step": step,
                #     },
                #     checkpoint_only_model_file,
                # )

                # Save the model weights and optimizer states
                # Only save trainable parameters


                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.module.get_trainable_state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step,
                    },
                    checkpoint_file,
                )

        scheduler.step()
        # Print epoch summary
        logger.info(
            f"Epoch [{epoch}/{num_epochs+1}]✅,"
            f" Train Loss: {np.round(np.mean(train_loss), decimals=2)},"
            f" Train Seq Acc: {np.round(np.mean(train_accuracy_seq['seq']), decimals=2)},"
            f" Val Seq Acc: {np.round(np.mean(val_accuracy_seq['seq']), decimals=2)}",
        )

    # Close the tensorboard summary writer
    writer.close()
    t2clogger.success("Training Finished.")


def validation_one_epoch(
    val_loader,
    model,
    epoch=0,
    num_epochs=0,
    writer=None,
    topk=5,
    config=None,
    total_batch=5,
):
    """
    Perform one validation epoch on the given validation loader.

    Args:
        val_loader (torch.utils.data.DataLoader): DataLoader for validation dataset.
        model (torch.nn.Module): The model to be validated.
        epoch (int, required): Current epoch number. Defaults to 0.
        num_epochs (int, required): Total number of epochs. Defaults to 0.
        writer (SummaryWriter, optional): TensorBoard SummaryWriter for logging. Defaults to None.
        topk (int, optional): Hybrid Sampling. Defaults to 5. Set to 1 for top-1
        config (dict, optional): Additional configuration parameters. Defaults to None.

    Returns:
        tuple: Mean Sequence Token Accuracy (mean_seq_token_acc)
    """
    seq_acc_all = []

    # Get available GPU ID and set the device to that GPU
    # gpu_id = get_available_gpu_ids()[0]
    device = torch.device(f"cuda")
    val_model = model.module.to(device)  # Move the model to the GPU
    val_model.eval()

    cur_batch = 0
    with torch.no_grad():
        # tqdm is used to create a progress bar for validation
        with tqdm(
            val_loader, ascii=True, desc=f"Epoch [{epoch}/{num_epochs+1}] Validation✨"
        ) as pbar:
            for _, vec_dict, prompt, mask_cad_dict in pbar:
                # If the number of batches specified by num_batch is reached, break the loop
                if cur_batch == total_batch:
                    break
                cur_batch += 1

                for key, val in vec_dict.items():
                    vec_dict[key] = val.to(device)

                for key, val in mask_cad_dict.items():
                    mask_cad_dict[key] = val.to(device)

                # Create a copy of the sequence dictionaries, and take only the start token
                sec_topk_acc = []
                for topk_index in range(1, topk + 1):
                    new_cad_seq_dict = vec_dict.copy()

                    for key, value in new_cad_seq_dict.items():
                        new_cad_seq_dict[key] = value[:, :1]
                        # new_mask_pc_seg_dict = {}

                    # Autoregressive Prediction (topk outputs per sample)
                    pred_cad_seq_dict = val_model.test_decode(
                        texts=prompt,
                        maxlen=MAX_CAD_SEQUENCE_LENGTH,
                        nucleus_prob=0,
                        topk_index=topk_index,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                    # Calculate accuracies
                    try:
                        cad_seq_acc = AccuracyCalculator(
                            discard_token=len(END_TOKEN)
                        ).calculateAccMulti2DFromLabel(
                            pred_cad_seq_dict["cad_vec"].cpu(),
                            vec_dict["cad_vec"].cpu(),
                        )
                    except Exception as e:
                        logger.error(f"Error: {e}")
                        cad_seq_acc = 0

                    # Update progress bar with current accuracy information
                    pbar.set_postfix({"Seq": np.round(cad_seq_acc, decimals=2)})

                    # Store accuracies for each batch
                    sec_topk_acc.append(cad_seq_acc)

                seq_acc_all.append(np.max(sec_topk_acc))
                sec_topk_acc = []

            # Calculate mean accuracies for sketches and extrusions
            mean_seq_acc = np.mean(seq_acc_all)
            gc.collect()
            torch.cuda.empty_cache()

            # If a writer is provided, log the mean accuracies to TensorBoard
            if writer is not None:
                writer.add_scalar(
                    "Seq Accuracy (Val)",
                    np.round(mean_seq_acc, decimals=2),
                    epoch,
                    new_style=True,
                )

            # Return mean accuracies for sketches and extrusions
            return mean_seq_acc


if __name__ == "__main__":
    main()
