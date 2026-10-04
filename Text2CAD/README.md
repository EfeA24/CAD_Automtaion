<div align="center">

# Text2CAD: Generating Sequential CAD Designs from Beginner-to-Expert Level Text Prompts


[Mohammad Sadil Khan*](https://scholar.google.com/citations?user=XIDQo_IAAAAJ&hl=en&authuser=1) · [Sankalp Sinha*](https://scholar.google.com/citations?user=QYcfOjEAAAAJ&hl=en&authuser=1&oi=ao) · [Talha Uddin Sheikh](https://scholar.google.com/citations?hl=en&authuser=1&user=yW7VfAgAAAAJ) · [Didier Stricker](https://scholar.google.com/citations?hl=en&authuser=1&user=ImhXfxgAAAAJ) · [Sk Aziz Ali](https://scholar.google.com/citations?hl=en&authuser=1&user=zywjMeMAAAAJ) · [Muhammad Zeshan Afzal](https://scholar.google.com/citations?user=kHMVj6oAAAAJ&hl=en&authuser=1&oi=ao)

_*equal contributions_

<h2> NeurIPS 2024 (Spotlight 🤩) </h2>

<a href="https://arxiv.org/abs/2409.17106">
  <img src="https://img.shields.io/badge/Arxiv-3498db?style=for-the-badge&logoWidth=40&logoColor=white&labelColor=2c3e50&borderRadius=10" alt="Arxiv" />
</a>
<a href="https://sadilkhan.github.io/text2cad-project/">
  <img src="https://img.shields.io/badge/Project-2ecc71?style=for-the-badge&logoWidth=40&logoColor=white&labelColor=27ae60&borderRadius=10" alt="Project" />
</a>
<a href="https://huggingface.co/datasets/SadilKhan/Text2CAD">
  <img src="https://img.shields.io/badge/Dataset-7D5BA6?style=for-the-badge&logoWidth=40&logoColor=white&labelColor=27ae60&borderRadius=10" alt="Dataset" />
</a>




</div>


# ⚙️ Installation

## 🌍 Environment

- 🐧 Linux
- 🐍 Python >=3.9

## 📦 Dependencies

```bash
$ conda env create --file environment.yml
```

# ✅ Todo List

- [x] Release Data Preparation Code
- [x] Release Training Code
- [x] Release Inference Code

# 📊 Data Preparation

Download the DeepCAD data from [here](https://github.com/ChrisWu1997/DeepCAD?tab=readme-ov-file#data).

**Generate Vector Representation from DeepCAD Json**

_You can also download the processed cad vec from [here](https://huggingface.co/datasets/SadilKhan/Text2CAD/blob/main/cad_seq.zip)._

```bash
$ cd CadSeqProc
$  python3 json2vec.py --input_dir $DEEPCAD_JSON --split_json $TRAIN_TEST_VAL_JSON --output_dir $OUTPUT_DIR --max_workers $WORKERS --padding --deduplicate
```


**Download the text annotations from [here](https://huggingface.co/datasets/SadilKhan/Text2CAD). Download the preprocessed [training](https://huggingface.co/datasets/SadilKhan/Text2CAD/blob/main/text2cad_v1.0/train_data.pkl) and [validation](https://huggingface.co/datasets/SadilKhan/Text2CAD/blob/main/text2cad_v1.0/validation_data.pkl) data and place it in** `Cad_VLM/dataprep` folder.

# 🚀 Training

In the `Cad_VLM/config/trainer.yaml`, provide the following path.

<details><summary>Required Updates in yaml</summary>
<p>

- `cache_dir`: The directory to load model weights from Huggingface.
- `cad_seq_dir`: The root directory that contains the ground truth CAD vector.
- `prompt_path`: Path for the text annotation.
- `split_filepath`: Json file containing the UIDs for train, test or validation.
- `log_dir`: Directory for saving _logs, outputs, checkpoints_.
- `checkpoint_path` (Optional): For resuming training after some epochs.

</p>
</details> 

<br>

```bash
$ cd Cad_VLM
$ python3 train.py --config_path config/trainer.yaml
```


# 🤖 Inference

### For Test Dataset

In the `Cad_VLM/config/inference.yaml`, provide the following path. Download the checkpoint for v1.0 [here](https://huggingface.co/datasets/SadilKhan/Text2CAD/blob/main/text2cad_v1.0/Text2CAD_1.0.pth).

<details><summary>Required Updates in yaml</summary>
<p>

- `cache_dir`: The directory to load model weights from Huggingface.
- `cad_seq_dir`: The root directory that contains the ground truth CAD vector.
- `prompt_path`: Path for the text annotation.
- `split_filepath`: Json file containing the UIDs for train, test or validation.
- `log_dir`: Directory for saving _logs, outputs, checkpoints_.
- `checkpoint_path`: The path to model weights. 

</p>
</details> 

<br>

```bash
$ cd Cad_VLM
$ python3 test.py --config_path config/inference.yaml
```

### Run Evaluation

```bash
$ cd Evaluation
$ python3 eval_seq.py --input_path ./output.pkl --output_dir ./output
```

### For Random Text Prompts

In the `Cad_VLM/config/inference_user_input.yaml`, provide the following path.

<details><summary>Required Updates in yaml</summary>
<p>

- `cache_dir`: The directory to load model weights from Huggingface.
- `log_dir`: Directory for saving _logs, outputs, checkpoints_.
- `checkpoint_path`: The path to model weights.
- `prompt_file` (Optional): For single prompt ignore it, for multiple prompts provide a txt file.

</p>
</details> 
<br>

  #### For single prompt
  
  ```bash
  $ cd Cad_VLM
  $ python3 test_user_input.py --config_path config/inference_user_input.yaml --prompt "A rectangular prism with a hole in the middle."
  ```

  #### For Multiple prompts

  ```bash
  $ cd Cad_VLM
  $ python3 test_user_input.py --config_path config/inference_user_input.yaml
  ```

# 💻 Run Demo


In the `Cad_VLM/config/inference_user_input.yaml`, provide the following path.

<details><summary>Required Updates in yaml</summary>
<p>

- `cache_dir`: The directory to load model weights from Huggingface.
- `log_dir`: Directory for saving _logs, outputs, checkpoints_.
- `checkpoint_path`: The path to model weights.

</p>
</details> 
<br>

```bash
$ cd App
$ gradio app.py
```



# 👥 Contributors
Our project owes its success to the invaluable contributions of these remarkable individuals. We extend our heartfelt gratitude for their dedication and support.


<a href="https://scholar.google.com/citations?hl=en&authuser=1&user=QYcfOjEAAAAJ">
  <img src="https://av.dfki.de/wp-content/uploads/avatars/162/1722545138-bpfull.png" width="50" height="50" style="border-radius: 50%;">
</a>
<a href="https://github.com/saali14">
  <img src="https://github.com/saali14.png" width="50" height="50" style="border-radius: 50%;">
</a>
<a href="https://scholar.google.de/citations?user=yW7VfAgAAAAJ&hl=en">
  <img src="https://scholar.google.de/citations/images/avatar_scholar_128.png" width="50" height="50" style="border-radius: 50%;">
</a>

<br>

# ✍🏻 Acknowledgement

We thank the authors of [DeepCAD](https://github.com/ChrisWu1997/DeepCAD) and [SkexGen](https://samxuxiang.github.io/skexgen/) and acknowledge the use of their code.

# 📜 Citation

If you use this dataset in your work, please consider citing the following publications.


```
@inproceedings{text2cad,
	author = {Khan, Mohammad Sadil and Sinha, Sankalp and Sheikh, Talha Uddin and Stricker, Didier and Ali, Sk Aziz and Afzal, Muhammad Zeshan},
	booktitle = {Advances in Neural Information Processing Systems},
	editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
	pages = {7552--7579},
	publisher = {Curran Associates, Inc.},
	title = {Text2CAD: Generating Sequential CAD Designs from Beginner-to-Expert Level Text Prompts},
	url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/0e5b96f97c1813bb75f6c28532c2ecc7-Paper-Conference.pdf},
	volume = {37},
	year = {2024},
	bdsk-url-1 = {https://proceedings.neurips.cc/paper_files/paper/2024/file/0e5b96f97c1813bb75f6c28532c2ecc7-Paper-Conference.pdf}}
```



