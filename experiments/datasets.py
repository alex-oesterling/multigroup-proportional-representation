from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import torchvision
from PIL import Image
import os
import pandas as pd
import glob
from tqdm.auto import tqdm

class CelebA(torch.utils.data.Dataset):
    def __init__(self, path, attributes = None, train=True, transform=torchvision.transforms.ToTensor(), embedding_model=None, return_paths=False):
        self.filepath = os.path.join(path, "celeba/")
        self.return_paths = return_paths

        self.embedding_model = embedding_model
        self.transform = transform
        self.img_paths = []
        self.labels = []

        self.attr_to_idx = {
            '5_o_Clock_Shadow': 0, 
            'Arched_Eyebrows': 1, 
            'Attractive': 2, 
            'Bags_Under_Eyes': 3, 
            'Bald': 4,
            'Bangs': 5,
            'Big_Lips': 6, 
            'Big_Nose': 7, 
            'Black_Hair': 8, 
            'Blond_Hair': 9, 
            'Blurry': 10, 
            'Brown_Hair': 11, 
            'Bushy_Eyebrows': 12, 
            'Chubby': 13, 
            'Double_Chin': 14, 
            'Eyeglasses': 15, 
            'Goatee': 16, 
            'Gray_Hair': 17, 
            'Heavy_Makeup': 18, 
            'High_Cheekbones': 19, 
            'Male': 20, 
            'Mouth_Slightly_Open': 21, 
            'Mustache': 22, 
            'Narrow_Eyes': 23, 
            'No_Beard': 24, 
            'Oval_Face': 25, 
            'Pale_Skin': 26, 
            'Pointy_Nose': 27, 
            'Receding_Hairline': 28, 
            'Rosy_Cheeks': 29, 
            'Sideburns': 30, 
            'Smiling': 31, 
            'Straight_Hair': 32, 
            'Wavy_Hair': 33, 
            'Wearing_Earrings': 34, 
            'Wearing_Hat': 35, 
            'Wearing_Lipstick': 36, 
            'Wearing_Necklace': 37, 
            'Wearing_Necktie': 38, 
            'Young': 39, 
        }

        self.labeltags = self.attr_to_idx.keys()

        if attributes is None:
            attributes = self.attr_to_idx.keys()

        attr_indices = [self.attr_to_idx[attr] for attr in attributes]

        if train:
            target_idx = 0
        else:
            target_idx = 1

        target_filepaths = set()
        with open(path + "celeba/list_eval_partition.txt", "r") as f:
            split_lines = f.readlines()
            for line in split_lines[1:]:
                line = line.strip()
                line = line.split()
                if int(line[1]) == target_idx:
                    target_filepaths.add(line[0])

        with open(path + "celeba/list_attr_celeba.txt", "r") as f:
            lines = f.readlines()
            lines = lines[2:]
            # self.len = len(lines)
            for idx, line in enumerate(lines):
                line = line.strip()
                line = line.split()

                if line[0] not in target_filepaths:
                    continue

                self.img_paths.append(line[0])
                self.labels.append(torch.tensor([int(line[idx+1]) for idx in attr_indices]))

        print(list(self.labeltags))

        self.labels = torch.stack(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path = os.path.join(self.filepath, "img_align_celeba/", self.img_paths[idx])
        label = self.labels[idx]
        if self.embedding_model is not None:
            image_id = path.split(".")[0]
            embeddingpath = os.path.join(self.filepath, self.embedding_model, image_id+".pt")
            return torch.load(embeddingpath), label
        image = Image.open(path)
        if self.return_paths:
            return self.transform(image), label, path
        return self.transform(image), label
    
class Occupations(torch.utils.data.Dataset):
    def __init__(self, path, transform=torchvision.transforms.ToTensor(), embedding_model=None, return_paths=False):
        self.filepath = os.path.join(path, "occupations/")

        self.transform = transform
        self.images = []
        self.labels = []
        self.embedding_model = embedding_model
        self.return_paths = return_paths

        ## Labels is [gender, occupation_one_hot]
        self.labeltags = ["gender"]

        df = pd.read_csv(self.filepath + "gender_labelled_images.csv")

        occupation_to_idx = {}
        for i, occ in enumerate(df.search_term.unique()):
            occupation_to_idx[occ] = i
            self.labeltags.append(occ)

        gender_to_idx = {
            'man': 0,
            'woman': 1
        }

        occupation_one_hot = torch.nn.functional.one_hot(torch.tensor([occupation_to_idx[occ] for occ in df.search_term]))

        gender = torch.tensor([gender_to_idx[gen] for gen in df.image_gender])

        self.labels = torch.hstack([gender.unsqueeze(1), occupation_one_hot])

        construct_path = lambda x, y: os.path.join(self.filepath, "google", str(x), str(y)+".jpg")
        self.img_paths = [construct_path(*x) for x in tuple(zip(df['search_term'], df['order']))]

        for path in self.img_paths:
            self.images.append(self.transform(Image.open(path)))
        print(self.labeltags)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.embedding_model:
            path = self.img_paths[idx]
            image_id = os.path.relpath(path.split(".")[0], self.filepath)
            embeddingpath = os.path.join(self.filepath, self.embedding_model, image_id+".pt")
            return torch.load(embeddingpath), self.labels[idx]
        if self.return_paths:
            return self.images[idx], self.labels[idx], self.img_paths[idx]
        return self.images[idx], self.labels[idx]
    
class FairFace(torch.utils.data.Dataset):
    def __init__(self, path, train=True, transform=torchvision.transforms.ToTensor(), embedding_model=None, binarize_age=True, return_paths=False):
        self.filepath = os.path.join(path, "fairface/")

        self.transform = transform
        # self.images = []
        self.image_paths = []
        self.labels = []
        self.embedding_model = embedding_model
        self.return_paths = return_paths

        self.labeltags = [
            "gender",
            "age",
        ]

        if train:
            df = pd.read_csv(self.filepath + "fairface_label_train.csv")
        else:
            df = pd.read_csv(self.filepath + "fairface_label_val.csv")

        self.race_to_idx = {}
        for i, race in enumerate(df.race.unique()):
            self.labeltags.append(race)
            self.race_to_idx[race] = i

        self.gender_to_idx = {
            'Male': 0,
            'Female': 1
        }

        self.age_to_idx = {
            '0-2': 0,
            '3-9': 1,
            '10-19': 2,
            '20-29': 3,
            '30-39': 4,
            '40-49': 5,
            '50-59': 6,
            '60-69': 7,
            'more than 70': 8
        }

        one_hot = torch.nn.functional.one_hot(torch.tensor([self.race_to_idx[race] for race in df.race])).numpy()
        gender_idx = [self.gender_to_idx[gen] for gen in df.gender]
        age_idx = [self.age_to_idx[age] for age in df.age]

        if binarize_age:
            age_idx = [int(ag>4) for ag in age_idx]
        ## labels is [gender_binary, age_categorical, race_one_hot]

        self.labels = []
        for i in range(len(gender_idx)):
            self.labels.append([gender_idx[i], age_idx[i]] + list(one_hot[i]))

        self.labels = torch.tensor(self.labels)

        # construct_path = lambda x: os.path.join(self.filepath, x)
        self.img_paths = df.file.to_list()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.embedding_model is not None:
            path = self.img_paths[idx]
            image_id = path.split(".")[0]
            embeddingpath = os.path.join(self.filepath, self.embedding_model, image_id+".pt")
            return torch.load(embeddingpath), self.labels[idx]
        path = os.path.join(self.filepath, self.img_paths[idx])
        if self.return_paths:
            return self.transform(Image.open(path)), self.labels[idx], path
        return self.transform(Image.open(path)), self.labels[idx]
    
class UTKFace(torch.utils.data.Dataset):
    def __init__(self, path, transform=torchvision.transforms.ToTensor(), embedding_model=None, binarize_age=True, return_paths=False):
        self.filepath = os.path.join(path, "utkface/")

        self.transform = transform
        self.img_paths = []
        self.labels = []
        self.embedding_model = embedding_model
        self.return_paths=return_paths

        self.labeltags = [
            "gender",
            "age",
            "white",
            "black",
            "asian",
            "indian",
            "others"
        ]

        gender = []
        age = []
        race = []
        for path in glob.glob(os.path.join(self.filepath, "*/*.jpg")):

            attributes = path.split("/")[-1].split("_")
            # if len(attributes) < 3:
            #     continue
            # print(path.split("/")[-1])
            # print(attributes)
            try:
                gen, ag, rac = int(attributes[1]), int(attributes[0]), int(attributes[2])
                gender.append(gen)
                age.append(ag)
                race.append(rac)
                # self.labels.append([int(attributes[1]), int(attributes[0]), int(attributes[2])])
                self.img_paths.append(path)
            except:
                continue

        df = pd.DataFrame({'gender': gender,
            'age': age,
            'race': race
        })

        self.race_to_idx = {
            "white": 0,
            "black": 1,
            "asian": 2,
            "indian": 3,
            "others": 4
        }
        
        self.gender_to_idx = {
            'male': 0,
            'female': 1
        }

        self.age_to_idx = {
            '0-2': 0,
            '3-9': 1,
            '10-19': 2,
            '20-29': 3,
            '30-39': 4,
            '40-49': 5,
            '50-59': 6,
            '60-69': 7,
            'more than 70': 8
        }

        age_idx = []
        for i, age in enumerate(df.age):
            if age <=2:
                age_idx.append(0)
            elif age <= 9:
                age_idx.append(1)
            elif age <= 19:
                age_idx.append(2)
            elif age <= 29:
                age_idx.append(3)
            elif age <= 39:
                age_idx.append(4)
            elif age <= 49:
                age_idx.append(5)
            elif age <= 59:
                age_idx.append(6)
            elif age <= 69:
                age_idx.append(7)
            else:  
                age_idx.append(8)

        if binarize_age:
            age_idx = [int(ag>4) for ag in age_idx]

        one_hot = torch.nn.functional.one_hot(torch.tensor(df.race)).numpy()    

        self.labels = []
        for i in range(len(df)):
            self.labels.append([df.gender[i], age_idx[i]] + list(one_hot[i]))

        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.embedding_model:
            path = self.img_paths[idx]
            image_id = os.path.relpath(path.split(".")[0], self.filepath)
            embeddingpath = os.path.join(self.filepath, self.embedding_model, image_id+".pt")
            return torch.load(embeddingpath), self.labels[idx]
        if self.return_paths:
            return self.transform(Image.open(self.imagepaths[idx])), self.labels[idx], self.img_paths[idx]
        return self.transform(Image.open(self.imagepaths[idx])), self.labels[idx]