# Midterm-All-Conv-Net
## Overview:
An Pytorch implementation of All convolution nets on cifar-10, and transfer learning, followed by [1](https://arxiv.org/abs/1412.6806)

## Report in Latex
- 'egpaper_final.tex' is the main file, where we do modification.
- PDF version is the generated file 'egpaper_final.pdf'.
- 'seolen_fig' is a self-created directory, for saving figures used in the report. To load the file, replace common 'a.png' as relative path 'seolen_fig/a.png'
- To write references, add a .bib format reference in 'egbib.bib'.       Note: only if you cite the reference tag, the citation show in final 'Reference' part.       Example: for the first reference in my egbib file, to cite it in the paper: **\cite{striving}**

## Code
- models dir: 
   - convNet.py:  all 6 models used in this experiment (A, B, C, SCC, CCC, ACC)
   - scifar.py: A file to load given data 'class1', class '2', based on torchvision.dataset.cifar10().
   - utils.py: provide a class encapsulting visdom plotting.

- dataset dir: datasets used.

- cifar10.py: Traning on cifar-10 dataset. All parameters are from 'config.py'.
- cifar100.py: Traning on cifar-100 dataset. Two optional training method: transfer learning and from scratch. All parameters are from 'config.py'.
- config.py: options of models, datasets, traning_strategy, load_model_path and hyperparameters used in BP.

## Important Material
- [How to organize a project in Deep Learning](https://zhuanlan.zhihu.com/p/29024978) 
- [How to generate a table for Latex online ](http://www.tablesgenerator.com/#)
