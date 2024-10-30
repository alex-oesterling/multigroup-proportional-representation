from mpr import *
from datasets import *
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
import argparse
import clip
import debias_clip as dclip
import faiss
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default="celeba", type=str)
    parser.add_argument('-device', default="cuda", type=str)
    parser.add_argument('-query', default="queries.txt", type=str)
    parser.add_argument('-model', default="clip", type=str)
    parser.add_argument('--data-path', default="datasets/", type=str)
    parser.add_argument('--embed-path', default="datasets/embeds", type=str) ##usually just inside data path
    args = parser.parse_args()
    print(args)

    if args.model=="clip":
        embedding_model = "clip"
        model, preprocess = clip.load("ViT-B/32", device=args.device)
    elif args.model == "debiasclip":
        embedding_model = "debiasclip"
        model, preprocess = dclip.load("ViT-B/16-gender", device =args.device) # DebiasClip for gender, the only publicly available model
        model.to(args.device)

    if args.dataset == "fairface":
        dataset = FairFace(args.data_path, train=True, transform=preprocess, embedding_model=None, return_paths=True)
    elif args.dataset == "occupations":
        dataset = Occupations(args.data_path, transform=preprocess, embedding_model=None, return_paths=True)
    elif args.dataset == "utkface":
        dataset = UTKFace(args.data_path, transform=preprocess, embedding_model=None, return_paths=True)
    elif args.dataset == "celeba":
        dataset = CelebA(args.data_path, attributes=None, train=True, transform=preprocess, embedding_model=None, return_paths=True)
    else:
        print("Dataset not supported!")
        exit()

    embeds = None
    with torch.no_grad():
        for images, _, path in tqdm(DataLoader(dataset, batch_size=512)):
            features = model.encode_image(images.to(args.device))
            if embeds is None:
                embeds = features.cpu()
            else:
                embeds = torch.cat((embeds, features.cpu()), dim=0)



    labels = dataset.labels
    embeds = torch.nn.functional.normalize(embeds, dim=0)
    print(labels.shape)
    print(embeds.shape)

    index = faiss.IndexFlatL2(512)   # build the index
    index.add(embeds.cpu())                  # add vectors to the index

    with open(args.query, 'r') as f:
        queries = f.readlines()
    q = None
    for query in queries:
        q_org = query.strip()
        qtext = "A photo of "+ q_org
        print(qtext)

        q_token = clip.tokenize(qtext).to(args.device)

        # ensure on the same device
        # q_token = q_token.to(args.device)

        with torch.no_grad():
            q_emb = model.encode_text(q_token).cpu().to(torch.float32)
        q_emb = q_emb/torch.nn.functional.normalize(q_emb, dim=0)
        
        if q is None:
            q = q_emb
        else:
            q = torch.cat((q, q_emb), dim=0)

    k = min(10000, embeds.shape[0])                        

    D, I = index.search(q, k)     # actual search

    
    for i, q in enumerate(queries):
        query = q.strip()
        querytag = query.split(" ")[-1]

        images = []
        retrievals = []
        for index in I[i]:
            path = dataset.img_paths[index]
            image_id = path.split(".")[0]
            images.append(image_id)
            retrievals.append(embeds[index].cpu().numpy())
        
        if not os.path.isdir(os.path.join(args.embed_path,args.dataset, embedding_model,querytag)):
            os.mkdir(os.path.join(args.embed_path,args.dataset, embedding_model,querytag))
        
        save_path = os.path.join(args.embed_path,args.dataset, embedding_model,querytag)

        retrievals = np.array(retrievals)
        np.save(os.path.join(save_path, "embeds.npy"), retrievals)
        with open(os.path.join(save_path, "images.txt"), "w") as f:
            for idd in images:
                f.write(idd + "\n")

if __name__ == "__main__":
    main()