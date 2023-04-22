## Data Selection for MT 
A Python Tool for Selecting Domain-Specific Data in Machine Translation.
## How to use
1. If you don't have conda installed on your machine, install it by following the instructions at this link: <br> https://docs.conda.io/projects/conda/en/latest/user-guide/install/<br>
2. Create a new environment and install the required packages by running the command:<br>
`conda create --name <env> --file requirements.txt`<br>
3. The tool's operation requires three inputs: (i) a parallel generic corpus, (ii) a monolingual domain-specific corpus, and (iii) the desired number of generated data to be selected. The (i) itself requires two inputs: source and target. To start the data selection process, run the command:
`python main.py -ood_src <(i)-source> -ood_tgt <(i)-target> -id <ii> -n <iii>`
