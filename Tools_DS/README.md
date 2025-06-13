## Data Selection for Machine Translation 
A Python Tool for Selecting Domain-Specific Data in Machine Translation.

---

## 🚀 New Repository

⚠️ **Note:** This tool now has its own dedicated repository. Please visit the main project page for the latest updates and detailed documentation:  
👉 [https://github.com/JoyeBright/domain-adapt-mt](https://github.com/JoyeBright/domain-adapt-mt)

---

## How to use
1. If you don't have conda installed on your machine, install it by following the instructions at this link: <br> https://docs.conda.io/projects/conda/en/latest/user-guide/install/<br>
2. Create a new environment, next activate it and finally install the required packages by running the command:<br>
`conda create --name <env> --file requirements.txt`<br>
3. The tool's operation requires three inputs: (i) a parallel generic corpus-source, (ii) a parallel generic corpus-target (iii) a monolingual domain-specific corpus. To start the data selection process, run the command: **`python main.py -ood_src [i-source] -ood_tgt [ii-target] -id [iii] -k [optional] -n [optional] -dis [optional] -fn [optional]`**
## Details and Tips
- `usage: main.py [-h] -ood_src GENERIC_SRC -ood_tgt GENERIC_TGT -id SPECIFIC [-k K] [-n NUMBER] [-dis] [-fn FILENAME]`
- `-k, -n, -dis, -fn` are optional arguments, while the rest are required. 
- If the value of `n` exceeds the number of available samples for the domain-specific corpus (iii), the generic corpora will be divided into equal parts. However, it is important to note that dividing the generic corpus in this way requires a large and diverse enough corpus to generate similar sentences.
- The original model had a word embedding dimension of 768, which might be too computationally intensive for some users. As a solution, we decreased the embedding dimensions to 32. You can access the final model at our HF repository using this [URL](https://huggingface.co/joyebright/stsb-xlm-r-multilingual-32dim). However, if your system can handle the model with 768 dimensions, we suggest using the original model that can be found at this link: https://huggingface.co/sentence-transformers/stsb-xlm-r-multilingual. The tool is configured to use the 32-dimensional model as the default option.
- We advise utilizing a single GPU instead of multiple GPUs to avoid potential conflicts that may arise from loading sentences across two separate GPUs. To specify CUDA device X, you would use this command: `export CUDA_VISIBLE_DEVICES=X`
## Cite the paper
https://aclanthology.org/2023.crowdmt-1.4/
```
@inproceedings{sharami-etal-2023-python,
    title = "A {P}ython Tool for Selecting Domain-Specific Data in Machine Translation",
    author = "Sharami, Javad Pourmostafa Roshan  and
      Shterionov, Dimitar  and
      Spronck, Pieter",
    editor = {Espl{\`a}-Gomis, Miquel  and
      Forcada, Mikel L.  and
      Kuzman, Taja  and
      Ljube{\v{s}}i{\'c}, Nikola  and
      van Noord, Rik  and
      Ram{\'\i}rez-S{\'a}nchez, Gema  and
      Tiedemann, J{\"o}rg  and
      Toral, Antonio},
    booktitle = "Proceedings of the 1st Workshop on Open Community-Driven Machine Translation",
    month = jun,
    year = "2023",
    address = "Tampere, Finland",
    publisher = "European Association for Machine Translation",
    url = "https://aclanthology.org/2023.crowdmt-1.4",
    pages = "29--30",
}

```

