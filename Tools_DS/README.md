## Data Selection for Machine Translation 
A Python Tool for Selecting Domain-Specific Data in Machine Translation.
## How to use
1. If you don't have conda installed on your machine, install it by following the instructions at this link: <br> https://docs.conda.io/projects/conda/en/latest/user-guide/install/<br>
2. Create a new environment, next activate it and finally install the required packages by running the command:<br>
`conda create --name <env> --file requirements.txt`<br>
3. The tool's operation requires three inputs: (i) a parallel generic corpus, (ii) a monolingual domain-specific corpus, and (iii) the desired number of generated data to be selected. The (i) splits into two inputs: source and target. To start the data selection process, run the command:
**`python main.py -ood_src [i-source] -ood_tgt [ii-target] -id [iii] -n [optional]`**
## Details and Tips
- `usage: main.py [-h] -ood_src GENERIC_SRC -ood_tgt GENERIC_TGT -id SPECIFIC [-n NUMBER]`
- `-n` is an optional argument, while the rest are required. 
- If the value of `n` exceeds the number of available samples for the domain-specific corpus (iii), the generic corpora will be divided into equal parts. However, it is important to note that dividing the generic corpus in this way requires a large and diverse enough corpus to generate similar sentences.
- The original model had a word embedding dimension of 768, which might be too computationally intensive for some users. As a solution, we decreased the embedding dimensions to 32. You can access the final model at our HF repository using this [URL](https://huggingface.co/joyebright/stsb-xlm-r-multilingual-32dim). However, if your system can handle the model with 768 dimensions, we suggest using the original model that can be found at this link: https://huggingface.co/sentence-transformers/stsb-xlm-r-multilingual. The tool is configured to use the 32-dimensional model as the default option.
- We advise utilizing a single GPU instead of multiple GPUs to avoid potential conflicts that may arise from loading sentences across two separate GPUs. To specify CUDA device X, you would use this command: `export CUDA_VISIBLE_DEVICES=X`
## Cite the paper
Please cite both the tool's paper and the original research paper (find below) if you decide to use the tool.
```
@inproceedings{sharami2023python,
  title={A Python Tool for Selecting Domain-Specific Data in Machine Translation},
  author={Sharami, Javad Pourmostafa Roshan and Shterionov, Dimitar and Spronck, Pieter},
  booktitle={1st Workshop on Open Community-Driven Machine Translation},
  pages={29},
  year={2023}
}

@article{Pourmostafa Roshan Sharami_Sterionov_Spronck_2021, 
title={Selecting Parallel In-domain Sentences for Neural Machine Translation Using Monolingual Texts}, 
volume={11}, 
url={https://www.clinjournal.org/clinj/article/view/137}, 
journal={Computational Linguistics in the Netherlands Journal}, 
author={Pourmostafa Roshan Sharami, Javad and Sterionov, Dimitar and Spronck, Pieter}, 
year={2021}, 
month={Dec.}, 
pages={213–230} }}
```

