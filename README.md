# Naturalness of Attention: Revisiting Attention in Code Language Models (ICSE'24 NIER)

## Abstract
> Language models like CodeBERT have advanced source code representation learning, but their opacity poses barriers to understanding what properties they capture. Recent attention analysis studies provide initial interpretability insights by focusing solely on attention weights rather than considering the complete context modeling of Transformers. 
The goal of this study is to shed some light on the previously ignored factors of the attention mechanism, beyond the attention weights.
We conduct an initial empirical study analyzing both attention distributions and transformed representations in CodeBERT. Across two programming languages, Java and Python, we find that the scaled transformation norms of the input better capture syntactic structure compared to attention weights alone. Our analysis reveals characterization of how CodeBERT embeds syntactic code properties. The findings demonstrate the importance of incorporating factors beyond just attention weights for rigorously understanding neural code models. This lays the groundwork for developing more interpretable models and effective uses of attention mechanisms in program analysis.

![Reformulating MHA](./figures/main.png)

## Replication Package

### Structure
This repository contains the code and data needed to replicate the study. The main folders of this repository is described as follow,
- `custom_transformers`: This folder contains the modified implementation of `modelling_roberta.py` from the `transformers` library, based on Kobayashi _et al._ [work](https://github.com/gorokoba560/norm-analysis-of-transformer). It is used to generate maps of $||f(x)||$ and $||\alpha f(x)||$ along with the attention maps from each head.
- `data`: Contains the data that was used in this study. It is generated after performing some preprocessing steps. We directly include the preprocessed data for convenience. You can also generate it from scratsh following the steps that will be mentioned next.
- `notebooks`: Notebooks that were used to preprocess the data are located here.
- `raw_data`: Data that will be preprocessed before conducting the study. As mentioned, we used two corpora (Python and Java) from CodeSearchNet. The raw data for the Java corpus is seeded by randomly sampling 65K samples. You can download the set from the original repository [here](https://github.com/github/CodeSearchNet#data), or you can download our initial seed of 65K from [here](https://drive.google.com/drive/folders/1yalXZDtI055XtPhMyqTw4SlKuJannFZ4?usp=sharing). As for the Python dataset, we used Wan _et al._'s given that it already contains the property map ([link](https://drive.google.com/file/d/1FCDcl7eRm_H30-huqnWe7rVSCd7Jx0nl/view))
- `results`: The destination folder where the results of RQ1 and RQ2 will be saved. `java` and `python` will contain the results of RQ1, whereas the root folder will contain the results of RQ2. We also provide the results data direclty: RQ1 results can be downloaded from [here](https://drive.google.com/drive/folders/1yalXZDtI055XtPhMyqTw4SlKuJannFZ4) and RQ2 results are already included there (the `.pickle` files).
- `scripts`: The implementation of the logic needed to extract data to answer RQ1 and RQ2.
- `visualization`: Includes the notebooks needed to generate figures 2, 3 and 4 in the manuscript.

### Dependencies
First, we recommend creating a virtual environment to install the dependencies,  
`python3 -m venv norm_analysis`  
`source ~/norm_analysis/bin/activate`  

Then, these dependencies can be installed by executing,   
`pip install -r requirements.txt`  
Note that in this study we used Python 3.9.6.

### Data Preprocessing
In this step we performed some minor data preprocessing. The major preprocessing steps are: removing comments (line and block), and discarding samples that cannot be parsed by [code_tokenize](https://github.com/cedricrupb/code_tokenize) which constituted $< 7.7\%$ of the total set of instances in the raw datasets. We used code_tokenize to tokenize source code into tokens and group them by category.  
Execute the notebooks `Preparing_CSN_Java.ipynb` and `Preparing_CSN_Python.ipynb` to generate the data needed for the study. Upon successfull execution, `5k_csn_java.jsonl` and `5k_csn_python.jsonl` will be generated and stored in the `data` folder. Note that these files are already included inf you want to skip this step.

### Generating Data for RQs
To generate the data needed to answer RQ1 and RQ2, execute the following command,  
`python3 scripts/RQx.py`  
where `x` is `1` or `2` depending on the RQ. The results of RQ1 will be stored in `results/java` and `results/python`, whereas the results of RQ2 will be directly saved as `.pickle` files in `results`.  
Finally, to generate the visualizations that were used to study the difference between the components of the MHA (_i.e.,_ $\alpha$, $f(x)$ and $\alpha f(x)$), execute the Jupyter notebooks `RQ1.ipynb` and `RQ2.ipynb` in the `visualization` folder.