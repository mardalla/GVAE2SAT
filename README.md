# GVAE2SAT
Code for the paper "SAT Instances Generation Using Graph Variational Autoencoders", by D. Crowley, M. Dalla, B. O'Sullivan, A. Visentin (2022)

# Abstract
This paper presents a SAT instance generator using a Graph Variational Autoencoder (\textit{GVAE2SAT}) architecture that outperforms existing generative deep learning models in speed and requires minimal post-processing. Our computational analyses benchmark this model against current deep learning techniques, introducing advanced metrics for more accurate evaluation. This new model is unique in its ability to maintain partial satisfiability of SAT instances while significantly reducing computational time. Although no method perfectly addresses all challenges in generating SAT instances, our approach marks a significant step forward in the efficiency and effectiveness of SAT instance generation.


**Dependencies**
- Python 3.11  
- Numpy 1.25  
- Pytorch-gpu 
- Pythorch-geometric 2.0  


# How to run
```cd Experiments/``` 

```python <name of python file in the directory> --src <absolute path to directory with files that all models generate from> --gvae2sat <absolute path to directory of files that GVAE2SAT trains on>```.
