## Data Selection for MT 
A Python Tool for Selecting Domain-Specific Data in Machine Translation.
## How to use
1. If you don't have conda installed on your machine, install it by following the instructions at this link: <br> https://docs.conda.io/projects/conda/en/latest/user-guide/install/<br>
2. Create a new environment, next activate it and finally install the required packages by running the command:<br>
`conda create --name <env> --file requirements.txt`<br>
3. The tool's operation requires three inputs: (i) a parallel generic corpus, (ii) a monolingual domain-specific corpus, and (iii) the desired number of generated data to be selected. The (i) splits into two inputs: source and target. To start the data selection process, run the command:
**`python main.py -ood_src [i-source] -ood_tgt [i-target] -id [ii] -n [iii]`**
## Details and Tips
- `usage: main.py [-h] -ood_src GENERIC_SRC -ood_tgt GENERIC_TGT -id SPECIFIC [-n NUMBER]`
- `-n` is an optional argument while the rest are required. 
- If the value of `n` exceeds the number of available samples for a specific ID, the tool will generate an exception. This is because the tool selects the top 5 closest sentences from out-of-domain data for each query sentence corresponding to that ID. If you require more domain-specific samples, you should consider dividing your generic corpus into smaller parts and selecting data for each part separately. However, keep in mind that the size and variety of your generic corpus should be sufficient for the tool to generate similar sentences.


