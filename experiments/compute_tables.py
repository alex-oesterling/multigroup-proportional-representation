import sys
sys.path.append(sys.path[0] + "/..")
from mpr import *
import torch
import numpy as np
from tqdm.auto import tqdm
import argparse
import seaborn as sns
import clip
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
sns.set_style("whitegrid")
import pickle
import os
import debias_clip as dclip
import json
from PIL import Image
import cv2

def argmax_probe_labels(probe_embeds):
    age_vec = (probe_embeds[:, 1] > 0.5).astype(np.int64)
    race_vec = probe_embeds[:, 2:]
    race_argmax = np.argmax(race_vec, axis=1)
    race_onehot = np.zeros((race_argmax.size, 7))
    race_onehot[np.arange(race_argmax.size), race_argmax] = 1
    return np.concatenate((probe_embeds[:, 0].reshape(-1, 1), age_vec.reshape(-1,1), race_onehot), axis=1)

def get_top_embeddings_labels_ids(dataset, query, embedding_model, datadir):
    if datadir == "occupations": ## Occupations is so small no need to use all 10k
        embeddings = []
        filepath = "/n/holylabs/LABS/calmon_lab/Lab/datasets/occupations/"
        embeddings = np.load(os.path.join(filepath, embedding_model, "architect/embeds.npy"))
        probe_labels = np.load(os.path.join(filepath, embedding_model, "architect/probe_labels.npy"))
        indices = []
        labels = np.zeros((embeddings.shape[0], dataset.labels.shape[1]))
        with open(os.path.join(filepath, embedding_model, "architect/images.txt"), "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                idx = dataset.img_paths.index(line+".jpg")
                labels[i] = dataset.labels[idx].numpy()
                indices.append(idx)
        # labels = dataset.labels.numpy()
        # indices = torch.arange(embeddings.shape[0])        
    else:
        retrievaldir = os.path.join("/n/holylabs/LABS/calmon_lab/Lab/datasets", datadir, embedding_model,query)
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
    args = parser.parse_args()
    print(args)

    if args.method != "debiasclip":
        embedding_model = "clip"
        # model, preprocess = clip.load("ViT-B/32", device=args.device)
        _, preprocess = clip.load("ViT-B/32", device=args.device)
        # model = model.to(args.device)
    else:
        embedding_model = "debiasclip"
        _, preprocess = dclip.load("ViT-B/16-gender", device =args.device) # DebiasClip for gender, the only publicly available model
        # model = model.to(args.device)

    if args.dataset == "fairface":
        dataset = FairFace("/n/holylabs/LABS/calmon_lab/Lab/datasets/", train=True, transform=preprocess, embedding_model=embedding_model)
    elif args.dataset == "occupations":
        dataset = Occupations("/n/holylabs/LABS/calmon_lab/Lab/datasets/", transform=preprocess, embedding_model=embedding_model)
    elif args.dataset == "utkface":
        dataset = UTKFace("/n/holylabs/LABS/calmon_lab/Lab/datasets/", transform=preprocess, embedding_model=embedding_model)
    elif args.dataset == "celeba":
        dataset = CelebA("/n/holylabs/LABS/calmon_lab/Lab/datasets/", attributes=None, train=True, transform=preprocess, embedding_model=embedding_model)

    if args.curation_dataset:
        if args.curation_dataset == "fairface":
            curation_dataset = FairFace("/n/holylabs/LABS/calmon_lab/Lab/datasets/", train=True, transform=preprocess, embedding_model=embedding_model)
        elif args.curation_dataset == "occupations":
            curation_dataset = Occupations("/n/holylabs/LABS/calmon_lab/Lab/datasets/", transform=preprocess, embedding_model=embedding_model)
        elif args.curation_dataset == "utkface":
            curation_dataset = UTKFace("/n/holylabs/LABS/calmon_lab/Lab/datasets/", transform=preprocess, embedding_model=embedding_model)
        elif args.curation_dataset == "celeba":
            curation_dataset = CelebA("/n/holylabs/LABS/calmon_lab/Lab/datasets/", attributes=None, train=True, transform=preprocess, embedding_model=embedding_model)
        else:
            print("Curation set not supported!")
            exit()
        
    with open(args.query, 'r') as f:
        queries = f.readlines()


    gender_counts_total = {"male": 0, "female": 0}
    age_counts_total = {"old": 0, "young": 0}
    if args.dataset == "utkface":
        race_counts_total = {"white": 0, "black": 0, "asian": 0, "indian": 0, "others": 0}
    else:
        race_counts_total = {"East Asian": 0, "Indian": 0, "Black": 0, "White": 0, "Middle Eastern": 0, "Latino_Hispanic": 0, "Southeast Asian": 0}

    curation_gender_counts_total = {"male": 0, "female": 0}
    curation_age_counts_total = {"old": 0, "young": 0}
    if args.dataset == "utkface":
        curation_race_counts_total = {"white": 0, "black": 0, "asian": 0, "indian": 0, "others": 0}
    else:
        curation_race_counts_total = {"East Asian": 0, "Indian": 0, "Black": 0, "White": 0, "Middle Eastern": 0, "Latino_Hispanic": 0, "Southeast Asian": 0}
    race_gender_intersection_total = np.zeros((10, len(gender_counts_total), len(race_counts_total)))


    for query_idx, query in tqdm(enumerate(queries[::-1])):
        q_org = query.strip()
        q = "A photo of "+ q_org
        q_tag = " ".join(q.split(" ")[4:])
        print(q_tag)

        path = "/n/holyscratch01/calmon_lab/Users/aoesterling/mpr/final_results/"

        if args.method == "top":
            with open(path+f'synthetic_{args.dataset}_curation_{args.curation_dataset}_top10k_lp_{args.k}_linearregression_{q_tag}.pkl', 'rb') as f:
                results = pickle.load(f)
        elif args.method == "lp":
            with open(path+f'synthetic_{args.dataset}_curation_{args.curation_dataset}_top10k_lp_{args.k}_linearregression_{q_tag}.pkl', 'rb') as f:
                results = pickle.load(f)
        else:
            with open(path+f'synthetic_{args.dataset}_curation_{args.curation_dataset}_top10k_{args.method}_{args.k}_linearregression_{q_tag}.pkl', 'rb') as f:
                results = pickle.load(f)
        
        retrieval_features, retrieval_labels, retrieval_indices, retrieval_probe_labels = get_top_embeddings_labels_ids(
            dataset,
            q_tag,
            embedding_model,
            args.dataset
        )
        retrieval_probe_labels = argmax_probe_labels(retrieval_probe_labels)
        if curation_dataset is not None:
            curation_features, curation_labels, curation_indices, curation_probe_labels = get_top_embeddings_labels_ids(
                curation_dataset,
                q_tag,
                embedding_model,
                args.curation_dataset
            )
            curation_features = curation_features.astype(np.float32)
            curation_probe_labels = argmax_probe_labels(curation_probe_labels)
            curation_labels_full = curation_labels

            if args.dataset == "utkface": ## remap races to utkface
                new_races = np.zeros((curation_probe_labels.shape[0], 5))
                new_races[:, 0] = curation_probe_labels[:, 5]
                new_races[:, 1] = curation_probe_labels[:, 4]
                new_races[:, 2] = np.logical_or(curation_probe_labels[:, 2], curation_probe_labels[:, 8]) ## Asian, SE Asian
                new_races[:, 3] = curation_probe_labels[:, 3]
                new_races[:, 4] = np.logical_or(curation_probe_labels[:, 6], curation_probe_labels[:, 7]) ## Middle Eastern, Latino
                curation_probe_labels = np.concatenate((curation_probe_labels[:, :2], new_races), axis=1)

            curation_labels = curation_probe_labels
            retrieval_labels = retrieval_probe_labels
        else:
            curation_features = None
            curation_labels = None
            curation_labels_full = None

        curation_gender_counts = {"male": 50*np.mean(curation_probe_labels[:,0]), "female": 50*(1-np.mean(curation_probe_labels[:,0]))}
        curation_age_counts = {"old": 50*np.mean(curation_probe_labels[:,1]), "young": 50*(1-np.mean(curation_probe_labels[:,1]))}
        if args.dataset == "utkface":
            curation_race_counts = {"white": 50*np.mean(curation_probe_labels[:,2:], axis=0)[0], "black": 50*np.mean(curation_probe_labels[:,2:], axis=0)[1], "asian": 50*np.mean(curation_probe_labels[:,2:], axis=0)[2], "indian": 50*np.mean(curation_probe_labels[:,2:], axis=0)[3], "others": 50*np.mean(curation_probe_labels[:,2:], axis=0)[4]}
        else:
            curation_race_counts = {"East Asian": 50*np.mean(curation_probe_labels[:,2:], axis=0)[0], "Indian": 50*np.mean(curation_probe_labels[:,2:], axis=0)[1], "Black": 50*np.mean(curation_probe_labels[:,2:], axis=0)[2], "White": 50*np.mean(curation_probe_labels[:,2:], axis=0)[3], "Middle Eastern": 50*np.mean(curation_probe_labels[:,2:], axis=0)[4], "Latino_Hispanic": 50*np.mean(curation_probe_labels[:,2:], axis=0)[5], "Southeast Asian": 50*np.mean(curation_probe_labels[:,2:], axis=0)[6]}

        for i in curation_gender_counts.keys():
            curation_gender_counts_total[i] += curation_gender_counts[i]
        for i in curation_age_counts.keys():
            curation_age_counts_total[i] += curation_age_counts[i]
        for i in curation_race_counts.keys():
            curation_race_counts_total[i] += curation_race_counts[i]

        best_indices = results['indices'][np.argmin(results["MPR"])]

        if args.method == "top":
            best_indices = results['indices'][-1]

            indices_rounded = best_indices.copy()
            indices_rounded[np.argsort(indices_rounded)[::-1][args.k:]] = 0
            indices_rounded[indices_rounded>1e-5] = 1.0  

            best_indices = indices_rounded
        elif args.method == "debiasclip":
            best_indices = results['indices']
        elif args.method == "lp":
            indices_rounded = best_indices.copy()
            indices_rounded[np.argsort(indices_rounded)[::-1][args.k:]] = 0
            indices_rounded[indices_rounded>1e-5] = 1.0  

            best_indices = indices_rounded

        best_indices_numbers = np.argwhere(best_indices == 1.0)

        gender_counts = {"male": 0, "female": 0}
        age_counts = {"old": 0, "young": 0}
        if args.dataset == "utkface":
            race_counts = {"white": 0, "black": 0, "asian": 0, "indian": 0, "others": 0}
        else:
            race_counts = {"East Asian": 0, "Indian": 0, "Black": 0, "White": 0, "Middle Eastern": 0, "Latino_Hispanic": 0, "Southeast Asian": 0}
        race_gender_intersection = np.zeros((len(gender_counts), len(race_counts)))
        images = []

        for index in best_indices_numbers:
            index = index[0]
            retrieval_index = retrieval_indices[index]
            if args.dataset == "celeba":
                imagepath = os.path.join("/n/holylabs/LABS/calmon_lab/Lab/datasets/", args.dataset, "img_align_celeba/", dataset.img_paths[retrieval_index])
            elif args.dataset == "utkface":
                imagepath = dataset.img_paths[retrieval_index]
            elif args.dataset == "occupations":
                imagepath = dataset.img_paths[retrieval_index]
            elif args.dataset == "fairface":
                imagepath = os.path.join("/n/holylabs/LABS/calmon_lab/Lab/datasets/",args.dataset, dataset.image_paths[retrieval_index])
            images.append(Image.open(imagepath))

            if int(retrieval_probe_labels[index][0]) == 1:
                gender_counts["female"] += 1
                gender_key_idx = 1
            else:
                gender_counts["male"] += 1
                gender_key_idx = 0

            if int(retrieval_probe_labels[index][1]) == 1:
                age_counts["old"] += 1
            else:
                age_counts["young"] += 1

            race_key_idx = np.argmax(retrieval_probe_labels[index][2:])
            race_counts[list(race_counts.keys())[race_key_idx]] += 1


            race_gender_intersection[gender_key_idx, race_key_idx] += 1


        for i in gender_counts.keys():
            gender_counts_total[i] += gender_counts[i]
        for i in age_counts.keys():
            age_counts_total[i] += age_counts[i]
        for i in race_counts.keys():
            race_counts_total[i] += race_counts[i]
        race_gender_intersection_total[query_idx, :, :] = race_gender_intersection

        impath = "/n/holyscratch01/calmon_lab/Users/aoesterling/mpr/"

        if not os.path.isdir(os.path.join(impath, args.dataset, args.method)):
            os.mkdir(os.path.join(impath, args.dataset, args.method))
        if not os.path.isdir(os.path.join(impath, args.dataset, args.method, q_tag)):
            os.mkdir(os.path.join(impath, args.dataset, args.method, q_tag))

        im_concat = np.zeros((64*5, 64*10, 3))
        i = 0
        j = 0
        for idx, im in enumerate(images):
            im_crop = cv2.resize(np.array(im), (64,64))
            im_concat[i*64:i*64+64, j*64:j*64+64, :] = im_crop
            i+=1
            if i == 5:
                i = 0
                j += 1
            im_crop = Image.fromarray(im_crop)
            im_crop.save(os.path.join(impath, args.dataset, args.method, q_tag, "{}.pdf".format(idx)))
        # print(im_concat)
        im_concat = Image.fromarray(im_concat.astype(np.uint8))
        im_concat.save(os.path.join(impath, args.dataset, args.method, q_tag, "concat.pdf"))
    
    for i in gender_counts_total.keys():
        gender_counts_total[i] /= 10
    for i in age_counts_total.keys():
        age_counts_total[i] /= 10
    for i in race_counts_total.keys():
        race_counts_total[i] /= 10
    for i in curation_gender_counts_total.keys():
        curation_gender_counts_total[i] /= 10
    for i in curation_age_counts_total.keys():
        curation_age_counts_total[i] /= 10
    for i in curation_race_counts_total.keys():
        curation_race_counts_total[i] /= 10
    
    print(np.mean(race_gender_intersection_total, axis=0))
    print(np.std(race_gender_intersection_total, axis=0))
    print(np.mean(np.sum(race_gender_intersection_total, axis=1), axis=0))
    print(np.mean(np.sum(race_gender_intersection_total, axis=2), axis=0))
    print(np.std(np.sum(race_gender_intersection_total, axis=1), axis=0))
    print(np.std(np.sum(race_gender_intersection_total, axis=2), axis=0))

    print(args.method)
    print(gender_counts_total)
    print(age_counts_total)
    print(race_counts_total)
    print("curation:")
    print(curation_gender_counts_total)
    print(curation_age_counts_total)
    print(curation_race_counts_total)
    print("intersections race gender")
    print(race_gender_intersection_total)

    for i in gender_counts_total.keys():
        gender_counts_total[i] -= curation_gender_counts_total[i]
    for i in age_counts_total.keys():
        age_counts_total[i] -= curation_age_counts_total[i]
    for i in race_counts_total.keys():
        race_counts_total[i] -= curation_race_counts_total[i] 

    print("difference from curation:")
    print(gender_counts_total)
    print(age_counts_total)
    print(race_counts_total)


if __name__ == "__main__":
    main()