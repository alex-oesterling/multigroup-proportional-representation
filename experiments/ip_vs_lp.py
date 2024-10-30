from mpr import *
from mpr.solvers import *
from datasets import *
import torch
import numpy as np
from tqdm.auto import tqdm
import argparse
import seaborn as sns
import clip
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
sns.set_style("whitegrid")
import pickle
import os
import debias_clip as dclip

def argmax_probe_labels(probe_embeds):
    age_vec = (probe_embeds[:, 1] > 0.5).astype(np.int64)
    race_vec = probe_embeds[:, 2:]
    race_argmax = np.argmax(race_vec, axis=1)
    race_onehot = np.zeros((race_argmax.size, probe_embeds.shape[1]-2))
    race_onehot[np.arange(race_argmax.size), race_argmax] = 1
    return np.concatenate((probe_embeds[:, 0].reshape(-1, 1), age_vec.reshape(-1,1), race_onehot), axis=1)

def get_top_embeddings_labels_ids(dataset, query, embedding_model, datadir, args):
    if datadir == "occupations": ## as occupations only has 3000 images, each query has the same 3000 image embeddings so we only compute probe labels for one query. Whichever query you compute the probe labels for in train_linear_probes.py you need to ensure is the same one being accessed here.
        embeddings = []
        filepath = os.path.join(args.embed_path, "occupations/")
        embeddingpath = glob.glob(os.path.join(filepath, embedding_model, "*/embeds.npy"))[0]
        embeddings = np.load(os.path.join(os.path.split(embeddingpath)[1],"/embeds.npy"))
        probe_labels = np.load(os.path.join(os.path.split(embeddingpath)[1],"/probe_labels.npy"))
        indices = []
        labels = np.zeros((embeddings.shape[0], dataset.labels.shape[1]))
        with open(os.path.join(os.path.split(embeddingpath)[1],"/images.txt"), "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                idx = dataset.img_paths.index(line+".jpg")
                labels[i] = dataset.labels[idx].numpy()
                indices.append(idx)   
    else:
        retrievaldir = os.path.join(args.embed_path, datadir, embedding_model,query)
        embeddings = np.load(os.path.join(retrievaldir, "embeds.npy"))
        probe_labels = np.load(os.path.join(retrievaldir, "probe_labels.npy"))
        # embeddings /= np.linalg.norm(embeddings, axis=1)
        labels = np.zeros((embeddings.shape[0], dataset.labels.shape[1]))
        indices = []
        with open(os.path.join(retrievaldir, "images.txt"), "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                idx = dataset.img_paths.index(line+".jpg")
                labels[i] = dataset.labels[idx].numpy()
                indices.append(idx)
    
    return embeddings, labels, indices, probe_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-method', default="mmr", type=str)
    parser.add_argument('-device', default="cuda", type=str)
    parser.add_argument('-dataset', default="celeba", type=str)
    parser.add_argument('-curation_dataset', default=None, type=str)
    parser.add_argument('-query', default="queries.txt", type=str)
    parser.add_argument('-k', default=10, type=int)
    parser.add_argument('-functionclass', default="randomforest", type=str)
    parser.add_argument('-use_clip', action="store_true")
    parser.add_argument('--embed-path', default="datasets/embeds", type=str)
    parser.add_argument('--data-path', default="datasets/", type=str)
    parser.add_argument('--out-path', default="results/", type=str)
    args = parser.parse_args()
    print(args)

    if args.method != "debiasclip":
        embedding_model = "clip"
        model, preprocess = clip.load("ViT-B/32", device=args.device)
        # model = model.to(args.device)

    else:
        embedding_model = "debiasclip"
        model, preprocess = dclip.load("ViT-B/16-gender", device =args.device) # DebiasClip for gender, the only publicly available model
        # model = model.to(args.device)

    # Load the dataset
    if args.dataset == "fairface":
        dataset = FairFace(args.data_path, train=True, transform=preprocess, embedding_model=embedding_model)
    elif args.dataset == "occupations":
        dataset = Occupations(args.data_path, transform=preprocess, embedding_model=embedding_model)
    elif args.dataset == "utkface":
        dataset = UTKFace(args.data_path, transform=preprocess, embedding_model=embedding_model)
    elif args.dataset == "celeba":
        dataset = CelebA(args.data_path, attributes=None, train=True, transform=preprocess, embedding_model=embedding_model)
    else:
        print("Dataset not supported!")
        exit()

    if args.curation_dataset:
        if args.curation_dataset == "fairface":
            curation_dataset = FairFace(args.data_path, train=True, transform=preprocess, embedding_model=embedding_model)
        elif args.curation_dataset == "occupations":
            curation_dataset = Occupations(args.data_path, transform=preprocess, embedding_model=embedding_model)
        elif args.curation_dataset == "utkface":
            curation_dataset = UTKFace(args.data_path, transform=preprocess, embedding_model=embedding_model)
        elif args.curation_dataset == "celeba":
            curation_dataset = CelebA(args.data_path, attributes=None, train=True, transform=preprocess, embedding_model=embedding_model)
        else:
            print("Curation set not supported!")
            exit()
    else:
        curation_dataset = None

    if args.functionclass == "randomforest":
        reg_model = RandomForestRegressor(max_depth=2, random_state=1017)
    elif args.functionclass == "linearregression":
        reg_model = LinearRegression(fit_intercept=False)
    elif args.functionclass == "decisiontree":
        reg_model = DecisionTreeRegressor(max_depth=3, random_state=1017)
    elif args.functionclass == "mlp":
        reg_model = MLPRegressor([64], random_state=1017)
    else:
        print("Function class not supported.")
        exit()

    with open(args.query, 'r') as f:
        queries = f.readlines()

    for query in tqdm(queries):
        q_org = query.strip()
        q = "A photo of "+ q_org
        q_tag = " ".join(q.split(" ")[4:])
        print(q_tag)
        q_emb = np.load("mpr/queries/{}_{}.npy".format(embedding_model, q_tag))
        # q_token = clip.tokenize(q).to(args.device)

        retrieval_features, retrieval_labels, retrieval_indices, retrieval_probe_labels = get_top_embeddings_labels_ids(
            dataset,
            q_tag,
            embedding_model,
            args.dataset,
            args
        )
        retrieval_features = retrieval_features.astype(np.float32)

        retrieval_probe_labels = argmax_probe_labels(retrieval_probe_labels)
        
        if curation_dataset is not None:
            curation_features, curation_labels, curation_indices, curation_probe_labels = get_top_embeddings_labels_ids(
                curation_dataset,
                q_tag,
                embedding_model,
                args.curation_dataset,
                args
            )
            curation_features = curation_features.astype(np.float32)
            curation_probe_labels = argmax_probe_labels(curation_probe_labels)

            if args.dataset == "utkface": ## remap races to utkface
                new_races = np.zeros((curation_probe_labels.shape[0], 5))
                new_races[:, 0] = curation_probe_labels[:, 5]
                new_races[:, 1] = curation_probe_labels[:, 4]
                new_races[:, 2] = np.logical_or(curation_probe_labels[:, 2], curation_probe_labels[:, 8]) ## Asian, SE Asian
                new_races[:, 3] = curation_probe_labels[:, 3]
                new_races[:, 4] = np.logical_or(curation_probe_labels[:, 6], curation_probe_labels[:, 7]) ## Middle Eastern, Latino
                curation_probe_labels = np.concatenate((curation_probe_labels[:, :2], new_races), axis=1)
            if args.curation_dataset == "utkface":
                new_races = np.zeros((retrieval_probe_labels.shape[0], 5))
                new_races[:, 0] = retrieval_probe_labels[:, 5]
                new_races[:, 1] = retrieval_probe_labels[:, 4]
                new_races[:, 2] = np.logical_or(retrieval_probe_labels[:, 2], retrieval_probe_labels[:, 8]) ## Asian, SE Asian
                new_races[:, 3] = retrieval_probe_labels[:, 3]
                new_races[:, 4] = np.logical_or(retrieval_probe_labels[:, 6], retrieval_probe_labels[:, 7]) ## Middle Eastern, Latino
                retrieval_probe_labels = np.concatenate((retrieval_probe_labels[:, :2], new_races), axis=1)

            curation_labels = curation_probe_labels
            retrieval_labels = retrieval_probe_labels
        else:
            curation_features = None
            curation_labels = None

        n = retrieval_labels.shape[0]

        if args.use_clip:
            curation_labels = curation_features
            retrieval_labels = retrieval_features 

        # compute similarities
        s = retrieval_features @ q_emb.T

        top_indices = np.zeros(n)
        selection = np.argsort(s.squeeze())[::-1][:args.k]
        top_indices[selection] = 1
        sim_upper_bound = s.T@top_indices
        rep_upper_bound = mpr(retrieval_labels, curation_labels, top_indices, reg_model)
        rep_upper_bound_1 = mpr(retrieval_labels, curation_labels, top_indices, reg_model)
        print(rep_upper_bound == rep_upper_bound_1, "MPR consistent across two calls") 
        print("KNN selection", selection)
        print("mpr for KNN", rep_upper_bound_1)
        print("sim for KNN", sim_upper_bound)


        torch.cuda.empty_cache()

        results = {}

        if args.method == "mapr_linear":
            solver = MAPRLinear(retrieval_labels, curation_labels, s, model=reg_model)

            num_iter = 50

            reps = []
            sims = []
            rounded_reps = []
            rounded_sims = []
            indices_list = []
            rounded_indices_list = []
            rhos = np.linspace(0.005, rep_upper_bound+1e-5, 50)

            for rho in tqdm(rhos, desc="rhos"):
                indices = solver.fit(args.k, num_iter, rho, top_indices)
                if indices is None: ## returns none if problem is infeasible
                    continue
                
                print("num_indices", np.sum(indices), flush=True)
                # rep, _ = mpr(indices, retrieval_labels, args.k, curation_set=curation_labels, model=reg_model)
                rep = mpr(retrieval_labels, curation_labels, indices, reg_model)
                print("Rep: ", rep, flush=True)
                sim = (s.T @ indices)
                print("Sim: ", sim, flush=True)

                reps.append(rep)
                sims.append(sim[0])
                indices_list.append(indices)

                indices_rounded = indices.copy()
                print(np.sort(indices_rounded)[::-1][:2*args.k], flush=True)
                print(np.argsort(indices_rounded)[::-1][:2*args.k], flush=True)
                
                indices_rounded[np.argsort(indices_rounded)[::-1][args.k:]] = 0.0
                indices_rounded[indices_rounded>0] = 1.0

                print(np.sort(indices_rounded)[::-1][:2*args.k], flush=True)
                print(np.argsort(indices_rounded)[::-1][:2*args.k], flush=True)

                print("num_rounded_indices", np.sum(indices_rounded), flush=True)

                print(np.sort(indices_rounded-indices)[::-1][:2*args.k], flush=True)
                print("diff:", np.sum(indices_rounded-indices))

                # rounded_rep, _ = mpr(indices_rounded, retrieval_labels, args.k, curation_set=curation_labels, model=reg_model)
                rounded_rep = mpr(retrieval_labels, curation_labels, indices_rounded, reg_model)
                print("Rounded Rep", rounded_rep, flush=True)
                rounded_sim = (s.T @ indices_rounded)
                print("rounded sim", rounded_sim, flush=True)
                print("diff sim:", s.T@(indices_rounded-indices), flush=True)

                rounded_reps.append(rounded_rep)
                rounded_sims.append(rounded_sim[0])

                rounded_indices_list.append(indices_rounded)

            results['MPR'] = reps
            results['sims'] = sims
            results['indices'] = indices_list
            results['rounded_MPR'] = rounded_reps
            results['rounded_sims'] = rounded_sims
            results['rounded_indices'] = rounded_indices_list
            results['rhos'] = rhos

            print(rounded_reps)

            if solver.problem:
                solver.problem.dispose()
            del solver
    
        elif args.method == "mapr_integer":
            solver = MAPRInteger(retrieval_labels, curation_labels, s, model=reg_model)

            num_iter = 50

            reps = []
            sims = []
            indices_list = []
            rhos = np.linspace(0.005, rep_upper_bound+1e-5, 50)
            for rho in tqdm(rhos[::-1], desc="rhos"):
                indices = solver.fit(args.k, num_iter, rho)
                
                if indices is None: ## returns none if problem is infeasible
                    continue
                sparsity = sum(indices>1e-4)
                
                # rep, _ = mpr(indices, retrieval_labels, args.k, curation_set=curation_labels, model=reg_model)
                rep = mpr(retrieval_labels, curation_labels, indices, reg_model)
                sim = (s.T @ indices)

                reps.append(rep)
                sims.append(sim[0])
                indices_list.append(indices)

            results['MPR'] = reps
            results['sims'] = sims
            results['indices'] = indices_list
            results['rhos'] = rhos
            solver.problem.dispose()
            del solver

        q_title = "_".join(q_org.split(" ")[1:])
        if args.use_clip:
            filename_pkl = "ipvslp_clip_{}_curation_{}_top10k_{}_{}_{}_{}.pkl".format(args.dataset, args.curation_dataset, args.method, args.k, args.functionclass, q_title)
        else:
            filename_pkl = "ipvslp_{}_curation_{}_top10k_{}_{}_{}_{}.pkl".format(args.dataset, args.curation_dataset, args.method, args.k, args.functionclass, q_title)
        print(filename_pkl)
        print(args.out_path+filename_pkl)
        if not os.path.exists(args.out_path):
            os.makedirs(args.out_path)
        with open(os.path.join(args.out_path,filename_pkl), 'wb') as f:
            pickle.dump(results, f)

    print(args)

if __name__ == "__main__":
    main()