SDM 2022 submission's code

# FAME - Fragment-based Conditional Molecular Generation for Phenotypic Drug Discovery
-----------------------------------------------------------------

## 1. Introduction
**FAME** is a Python implementation of the deep graph generative model targeting phenotypic molecular design, in particular gene expression-based molecular design. **FAME** leverages a conditional variational autoencoder framework to learn the conditional distribution generating molecules from gene expression profiles. To tackle the difficulties when learning this distribution due to the complexity of the molecular space and the noisy phenomenon in gene expression data, first, a gene expression denoising (GED) model using constrative objective function is proposed to reduce noise from gene expression data before inputting to **FAME**. Second, **FAME** is designed to treat molecules as the sequences of fragments and then learn to generate these fragments in autoregressive manner. By leveraging this fragment-based generation strategy and the denoised gene expression profiles, **FAME** can generate novel molecules with a high validity rate and desired biological activity. 


The experimental results show that **FAME** outperforms existing methods including both SMILES-based and graph-based deep generative models for phenotypic molecular design. Furthermore, the effective mechanism for reducing noise in gene expression data proposed in our study can be applied and then adds more values to other phenotypic drug discovery applications.

## 3. FAME

![alt text](docs/fame.png "FAME")

Figure 1: Overall architecture of **FAME**

## 4. Installation

**FAME** depends on numpy, scipy, pandas, tqdm, scikit-learn, rdkit, PyTorch Geometric, and PyTorch (CUDA toolkit if use GPU).
You must have them installed before using **FAME**.

The simple way to install them is using conda:

```sh
	$ conda install numpy scipy pandas tqdm scikit-learn rdkit pyg pytorch
```
## 5. Usage

### 5.1. Data

We do not provide the insurance claim data itself due to copyright issue.

### 5.2. Training FAME

The training script for **FAME** is located at the main folder.

```sh
    $ python main_fame.py 
```

Arguments in this scripts:

* ``--fold``:       data fold in cross-validation setting
* ``--batch_size``:         batch size used for training
* ``--max_epoch``:        maximum number of training iterations
* ``--model_name``:        name of model
* ``--gpu``:      gpu id
* ``--warm_start``: train model from pretrained weights
* ``--inference``:       use model in inference stage
