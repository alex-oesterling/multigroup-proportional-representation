# Multi-Group Proportional Representation in Retrieval
---
Codebase for the paper ["Multi-Group Proportional Representation in Retrieval"](https://arxiv.org/abs/2407.08571) by Alex Oesterling*, Claudio Mayrink Verdun*, Carol Xuan Long, Alexander Glynn, Lucas Monteiro Paes, Sajani Vithana, Martina Cardone, and Flavio P. Calmon

## Installation
To use MPR clone the repository and install it as a package:

```
git clone https://github.com/alex-oesterling/multigroup-proportional-representation
cd multigroup-proportional-representation
pip install .
```

Note: we use Gurobi as our linear solver, which requires an academic or industrial license. See [here](https://www.gurobi.com/academia/academic-program-and-licenses/) for academic licenses.

## Usage

## Precomputing Embeddings and Probe Weights
To accelerate experimentation, we precomputed and stored CLIP and DebiasCLIP embeddings for our image datasets. Furthermore, we ran vanilla KNN retrieval on the top 10000 entries first before performing MOPR per-query. The following code takes a dataset, computes its CLIP embedding, and for each query, stores a .npy array of size min(10000, dataset_size) by 512 with the top most similar embeddings for the query. It also stores a .txt file with the corresponding image ids for downstream use.

```
python experiments/embed_datasets.py -model "clip" -dataset "celeba" -device "cuda" -query "queries.txt" --data-path "path/to/dataset" --embed-path "path/to/save/embeddings"
```

Then, to label these images with the attributes used in the paper (gender, age, and race), run the following:

```
python experiments/train_linear_probes.py -device "cuda" -dataset "celeba" --data-path "path/to/dataset" --embed-path "path/to/load/embeddings"
```

## Replicating Experiments

### Base Retrieval Code

To replicate Figure 1 (and corresponding Appendix figures) and run a hyperparamter sweep for each method for a specific experimental setting, run the following. Note that DebiasCLIP does not have a hyperparameter to sweep. We only recommend using "fairface" or no curation dataset. Also, we note that for MOPR, Gurobi isn't garbage collecting properly so even though we delete and start a new solver for each query, the memory consumption goes up and sometimes you run into OOM errors. In that case you could easily run a bash script to start a new python call for each query uniquely instead of running it out of a single call.

Also, because we precompute embeddings, these experiments can be run on CPU only. All that you have to do is uncomment lines 210-218 to embed the query and save it somewhere and then you can load all the embeddings on CPU to conduct retrieval. You may even be able to get away with just one call to CLIP on CPU for the query embedding in a reasonable amount of time. The precomputed queries will be saved to the `experiments` directory.

The two files we provide are `queries.txt` which contains the 10 professions we constructed with ChatGPT and `queries_occupations.txt` which contains the 45 professions in the Occupations dataset.

```
python experiments/benchmark_retrieval.py -device "cpu" -dataset "celeba" -query 'experiments/queries.txt' -curation_dataset "fairface" -k 50 -functionclass "linearregression" -method "mopr" --data-path "path/to/dataset" --embed-path "path/to/load/embeddings" --out-path "path/to/save/results"
```

To replicate Table 1, uncomment lines 180-192 and use synthetic, balanced data instead of the curation set.

### Comparing integer and linear programs

To compare the results when using the hard integer program and lineaer program solutions, run the following (using `-method "mopr_linear"` or `-method "mopr_integer"` accordingly)

```
python experiments/ip_vs_lp.py -device "cpu" -dataset "celeba" -curation_dataset "fairface" -query 'experiments/queries.txt' -k 50 -functionclass "linearregression" -method "mopr_linear" --data-path "path/to/dataset" --embed-path "path/to/load/embeddings" --out-path "path/to/save/results"
```

### Comparing closed form and MSE-oracle solutions to MPR in the linear case

To validate the performace of our iterative MOPR algorithm as well as the accuracy of estimating MPR with a regression model, run the following (using `-method "mopr_regression"` or `-method "mopr_closed"`). Note that the function class must be linear regression for this experiment.

```
python experiments/closed_vs_oracle.py -device "cpu" -dataset "celeba" -query 'experiments/queries.txt' -k 50 -functionclass "linearregression" -method "mopr_regression" --data-path "path/to/dataset" --embed-path "path/to/load/embeddings" --out-path "path/to/save/results"
```

### Recomputing MPR to accelerate results

For all methods but MOPR, the resulting retrievals are the same, independent of the function class used to measure MPR. We can accelerate our results by passing in an existing set of retrievals and just recomputing MPR over the other function classes we want to evaluate, rather than having to rerun the entire retrieval algorithm for each new measurement we want to take. This is especially important for the MMR algorithm which can take hours per-solve, saving many hours of experiment time.

Notes: The `-functionclass` flag determines what *new* function class to compute MPR over. The code assumes 1) the existing retrieval was computed over linear regression (to modify, see line 133) and 2) that the existing results were stored in the same `--out-path` as where the new results will be stored.

```
python experiments/recompute_mpr.py -device "cpu" -dataset "celeba" -curation_dataset "fairface" -query 'experiments/queries.txt' -k 50 -functionclass "decisiontree" -method "mmr" --data-path "path/to/dataset" --embed-path "path/to/load/embeddings" --out-path "path/to/save/results"
```
