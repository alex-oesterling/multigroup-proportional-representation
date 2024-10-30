import random
import numpy as np
from .utils import fon


class PBM():
    ## As defined in the paper "Mitigating Test-Time Bias for Fair Image Retrieval" (Kong et. al. 2023)
    def __init__(self, features, similarities, pbm_labels, pbm_classes):
        self.features = features
        self.similarities = similarities
        self.m = features.shape[0]
        self.pbm_label = pbm_labels # predicted sensitive group label
        self.pbm_classes = pbm_classes

    def fit(self, k=10, eps=0):
        similarities_sorted = np.argsort(self.similarities.squeeze())[::-1]
        selections = []

        neutrals = [x for x in similarities_sorted if self.pbm_label[x] == 0]
        classes = [[x for x in similarities_sorted if self.pbm_label[x]== i] for i in range(1, len(self.pbm_classes))]

    
        while len(selections) < k:
            if random.random() < eps:
                try:
                    neutral_sim = self.similarities[neutrals[0]]
                except:
                    neutral_sim = -1
                
                max_class, idx = 0, 0
                for i, c in enumerate(classes):
                    try:
                        class_sim = self.similarities[c[0]]
                    except:
                        class_sim = -1
                    if class_sim > max_class:
                        max_class = class_sim
                        idx = i
                if max_class > neutral_sim:
                    selections.append(classes[idx][0])
                    classes[idx].pop(0)
                else:
                    selections.append(neutrals[0])
                    neutrals.pop(0)
                        
            else:
                best_neutral = neutrals[0]
                best_for_classes = [fon(c) for c in classes]
                best_for_classes_vals = [c for c in best_for_classes if c is not None]

                similarities_for_classes = [self.similarities[x] for x in best_for_classes_vals]
                avg_sim = np.mean(similarities_for_classes)
                neutral_sim = self.similarities[best_neutral]

                if avg_sim > neutral_sim:
                    if len(selections) + len(best_for_classes_vals) > k:
                        best_for_classes_vals = random.choices(best_for_classes_vals, k=k-len(selections))
                    selections += best_for_classes_vals

                    for i, x in enumerate(best_for_classes):
                        if x is not None:
                            classes[i].pop(0)
                else:
                    selections.append(best_neutral)
                    neutrals.pop(0)

        indices = np.zeros(self.m)
        indices[selections] = 1    
        AssertionError(np.sum(indices)==k)
        return indices, selections