# People Who Liked This Also Liked ...A Publication Analysis of Three Decades of Recommender Systems Research

Barry Smyth, Insight Research Ireland Centre for Data Analytics, University College Dublin, Ireland

May 2025

barry.smyth@ucd.ie

## Introduction

This repository contains the code and data needed to reproduce a publication analysis of recommender systems (RS) published in ACM Transactions on Recommender Systems. The analysis uses a large collection of papers (>2M) includeing >50k papers identified as core RS papers. These papers were collected using the Semantic Scholar (SS) API. The terms of use of this API do not permit the republishing of raw SS data; for further information on SS see https://www.semanticscholar.org/product/api.

## License

This project is licensed under the [CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/). Non-commercial use only.

## Directory Structure

### notebooks/
This are the main jupyter notebooks (named in sequence) needed to rerun the analysis. They include data collection notebooks for reproducing the data used in this analysis. However, it should be noted that re-running the data collection is liable to produce a different dataset due to updates in the SS database; see below for how to reproduce the original datasets exactly.

* `1000` - Constructs an initial seed set of RS paper ids.
* `2000` - Uses the seed paper ids as the basis of an expanded dataset of RS papers and related papers (author publications, citations, references etc.) downloaded from SS.
* `2100` - This notebook further refines the expanded dataset to identify the papers that can be viewed as core RS papers.
* `2200` - Cleans the refined dataset and adds some addiitonal columns.
* `2300` - Identifies the inside and outside communities as described in the analysis and produces various Venn diagrams used in the main analysis.
* `2400` - This notebook include the BERT topic discovery code uses to identify the $U_p$ and $R_p$ topics and saves the topic models to file. These topic models are available in _data/model_ to avoid the need to re-run the BERT code, which requires significant memory and computational resources.
* `2410` - This notebook combines the relevant topic information to the main dataset.
* `3000` - Perfoms the basif paper, author, and venue analysis and prodices several graphs.
* `3100` - Comparesompares various aspects of the inside and outside communities (growth dynamics, engagement etc.).
* `3400` - An analysis of the $U_p$ topics and produces various graphs, tables, and wordclouds.
* `3410` - An analysis of the $R_p$ topics and produces various graphs, tables, and wordclouds.
* `3500` - Performs a citation analysis of the main RS papers to identify influential papers. various graphs and tables are also produced.
* `3600` - Identifes and visualises emerging topics with significant publication/citation momentum.

Notice that the notebooks are numerically labeled to indicate the esxecution order. In addition, any files produced by a notebook (e.g. dataset, graph etc) are labelled with the notebooks numerical code.

### data/
The data directory contains various different folders for different types of data used in this study. Most of these folders are empty but will be populated by the code provided. However, certain key files are provided, where feasible, to aid reproducibilitiy, as discussed below. Here is a brief summary of the folders and their uses:

* `data/raw` - as data collection proceeds this directory will be used to hold the raw data collected from SS. Note that it includes seed dataset _1000_recsys_paper_ids_52550.feather_ which is the orihginal ids of a seed set of RS related papers collected using _notebooks/1000_build_recsys_paper_ids_dataset.ipynb_.
* `data/orig` - this contains a set of data files containing the actual paper ids used in the original study.
* `data/models` - this directory contains the various topic model data used in the original study. These data can be used in the topic analysis notebooks (`2410`) as an alterative to re-generating the topic models from raw data, whihc is very expensive.
* `data/processed` - this directory contains the various datasets produced during the course of the analysis.

### graphs/
The graphs produced in the various analysis notebooks are saved here as png files. The original pngs from the paper are included in this directory for reference but they will be overwritten if the analysis notebooks are re-run.

### src/
This directory contains some wrapper code used to interact with the SS API. It uses multiprocessing to parallelise the download of large datasets while respecting rate limits and various techniques to manage API errors etc,.

## Reproducing the Original Study
As mentioned above, reproducing the original study precisely is non-trivial in the sense that re-running the data collection code will likely produce a different dataset because of the SS updates that have taken place since the data was originally collected in late 2024. However, the results should be substantially similar, especially since the analysis focuses on the period prior to 2024.

To aid reproducibility, the datasets `data/raw/2000_recsys_papers_ids_with_extra_columns.feather` and `data/raw/2000_recsys_authors_ids_with_extra_columns.feather` contain the full set of original paper/author ids collected from SS as the basis for the study; these datasets also contain some additional columns constructed during the data collection process in `notebooks/2000_build_recsys_ss_paper_datatset.ipynb`. It should be possible to re-collect the records for these paper/authors ids using the SS API (code is included in `src/` to help with this) and these records can then be joined with the id dataframes in order to recreate the exact dataframes (`2000_recsys_papers.feather` and `2000_recsys_authors.feather`) needed by this study, starting with `2100_refine_recsys_ss_paper_datatset.ipynb`


