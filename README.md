# Predicting Strategic Behavior from Free Text
Omer Ben-Porat, Sharon Hirsch, Lital Kuchy, Guy Elad, Roi Reichart and Moshe Tennenholtz.

## Overview

This repository contains data and code for [our paper](https://www.jair.org/index.php/jair/article/view/11849). All data files appear under the `data` folder. In particular, 

1. `human_text_games.csv` - the texts, games and demographics.
2. `traits_features.csv` - the collected attributes.
3. `LIWC_features.csv` - the LIWC features
4. `bluemix-tone-analyzer_features.csv` - Bluemix representation for the (uncensored) texts.
5. `tfidf_features.csv`	- tfidf feature for the (uncensored) texts.

Beyond the data files, you will find our implementation of TAC and the baselines we used in the paper, as well as the `replicate_paper.py` file that can be used to replicate our results.

## Data Censoring
Despite instructing participants to omit any identifying details, some of the texts included sensitive information like names. We replaced such information with tags. For instance, in text 57 we replaced the actual name with `<PERSONNAME57>`. If the text included several names (like text 28), we used `<PERSONNAME28-1>` and `<PERSONNAME28-2>`. Some of the participants used seemingly fake names. We removed those too since we could not verify these are fake. Some texts received special attention, since censoring names would have twisted the spirit of the essay.

## Post-processing Comments
After the paper was published, we discovered that texts 203 and 249 are not genuine---they were copied from the web. We do not consider this as a significant issue, since
1. these texts do reflect commonsensical features, which we attribute to the authors despite not writing them, and 
2. these are only two texts and thus will not change the results significantly.

## Citation
If you make use of our data or code for research purposes, we would appreciate citing the following:
```
@article{DBLP:journals/jair/Ben-PoratHKERT20,
  author    = author={Ben{-}Porat, Omer and 
                      Hirsch, Sharon and 
                      Kuchi, Lital and 
                      Elad, Guy and 
                      Reichart, Roi 
                      and Tennenholtz, Moshe},
  title     = {Predicting Strategic Behavior from Free Text},
  journal   = {Journal of Artificial Intelligence Research},
  volume    = {68},
  pages     = {413--445},
  year      = {2020},
  url       = {https://doi.org/10.1613/jair.1.11849},
  doi       = {10.1613/jair.1.11849},
}
```
