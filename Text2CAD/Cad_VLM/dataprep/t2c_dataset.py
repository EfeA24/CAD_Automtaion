import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import pandas as pd
from tqdm import tqdm
import re
from concurrent.futures import ThreadPoolExecutor
import pickle

class Text2CAD_Dataset(Dataset):
    def __init__(
        self,
        cad_seq_dir: str,
        prompt_path: str,
        split_filepath: str,
        subset: str,
        max_workers: int,
    ):
        """
        Args:
            cad_seq_dir (string): Directory with all the .pth files.
            prompt_path (string): Directory with all the .npz files.
            split_filepath (string): Train_Test_Val json file path.
            subset (string): "train", "test" or "val"
        """
        super(Text2CAD_Dataset, self).__init__()
        self.cad_seq_dir = cad_seq_dir
        self.prompt_path = prompt_path
        self.prompt_df = pd.read_csv(prompt_path)
        self.prompt_df = self.prompt_df[
            self.prompt_df["abstract"].notnull()
            & self.prompt_df["beginner"].notnull()
            & self.prompt_df["intermediate"].notnull()
            & self.prompt_df["expert"].notnull()
        ]
        self.all_prompt_choices = ["abstract", "beginner", "intermediate", "expert"]
        self.substrings_to_remove = ["*", "\n", '"', "\_", "\\", "\t", "-", ":"]
        # open spilt json
        with open(os.path.join(split_filepath), "r") as f:
            self.split = json.load(f)

        self.uid_pair = self.split[subset]
        func = self._prepare_data
        
        if os.path.exists(f"{subset}_data.pkl"):
            # Load the prompt data from the pickle file
            with open(f"{subset}_data.pkl",'rb') as f:
                self.prompt_data = pickle.load(f)
        else:
            # Load the prompt data using ThreadPoolExecutor and _prepare_data function
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create a dictionary to store the prompt data
                self.prompt_data = {}
                # Use ThreadPoolExecutor to process the prompt data in parallel
                for data in tqdm(
                    executor.map(func, self.uid_pair),
                    total=len(self.uid_pair),
                    desc=f"Loading {subset} split",
                ):
                    if data is not None:
                        uid, cad_vec, prompt, mask_cad_dict = data
                        if isinstance(prompt, dict):
                            for key, val in prompt.items():
                                self.prompt_data[uid + f"_{key}"] = (
                                    cad_vec,
                                    val,
                                    mask_cad_dict,
                                )  # "0000/00001234" -> "0000/00001234_beginner"

        self.keys = list(self.prompt_data.keys())
        print(f"Found {len(self.prompt_data)} samples for {subset} split.")

    def __len__(self):
        return len(self.keys)

    def _prepare_data(self, uid):
        root_id, chunk_id = uid.split("/")
        if len(self.prompt_df[self.prompt_df["uid"] == uid]) == 0:
            return None
        cad_vec_dict = torch.load(
            os.path.join(self.cad_seq_dir, root_id, chunk_id, "seq", f"{chunk_id}.pth"),
            weights_only=True,
        )
        level_data = {}
        for prompt_choice in self.all_prompt_choices:
            prompt = self.prompt_df[self.prompt_df["uid"] == uid][prompt_choice].iloc[0]
            if isinstance(prompt, str):
                level_data[prompt_choice] = prompt
        if len(level_data) == 0:
            return None
        # Filter the prompt
        prompt = self.remove_substrings(prompt, self.substrings_to_remove).lower()
        return uid, cad_vec_dict["vec"], level_data, cad_vec_dict["mask_cad_dict"]

    def remove_substrings(self, text, substrings):
        """
        Remove specified substrings from the input text.

        Args:
            text (str): The input text to be cleaned.
            substrings (list): A list of substrings to be removed.

        Returns:
            str: The cleaned text with specified substrings removed.
        """
        # Escape special characters in substrings and join them to form the regex pattern
        regex_pattern = "|".join(re.escape(substring) for substring in substrings)
        # Use re.sub to replace occurrences of any substrings with an empty string
        cleaned_text = re.sub(regex_pattern, " ", text)
        # Remove extra white spaces
        cleaned_text = re.sub(" +", " ", cleaned_text)
        return cleaned_text

    def __getitem__(self, idx):
        uid = self.keys[idx]
        return uid, *self.prompt_data[uid]


def get_dataloaders(
    cad_seq_dir: str,
    prompt_path: str,
    split_filepath: str,
    subsets: list[str],
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
    num_workers: int,
    prefetch_factor: int,
):
    """
    Generate a DataLoader for the Text2CADDataset.

    Args:
    - cad_seq_dir (str): The directory containing the CAD sequence files.
    - prompt_path (str): The path to the CSV file containing the prompts.
    - split_filepath (str): The path to the JSON file containing the train/test/validation split.
    - subsets (list[str]): The subset to use ("train", "test", or "val").
    - batch_size (int): The batch size.
    - shuffle (bool): Whether to shuffle the data.
    - pin_memory (bool): Whether to pin memory.
    - num_workers (int): The number of workers.
    - prefetch_factor (int): The prefetch factor.

    Returns:
    - dataloader (torch.utils.data.DataLoader): The DataLoader object.
    """

    all_dataloaders = []

    for subset in subsets:
        # Create an instance of the Text2CADDataset
        dataset = Text2CAD_Dataset(
            cad_seq_dir=cad_seq_dir,
            prompt_path=prompt_path,
            split_filepath=split_filepath,
            subset=subset,
            max_workers=num_workers,
        )

        # Create a DataLoader with the specified parameters
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,  # You can set this to True if you want to shuffle the data
            num_workers=num_workers,
            pin_memory=pin_memory,  # Set to True if using CUDA
            prefetch_factor=prefetch_factor,
        )
        all_dataloaders.append(dataloader)

    return all_dataloaders


if __name__ == "__main__":
    pass
