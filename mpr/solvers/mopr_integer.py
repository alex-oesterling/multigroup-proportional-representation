import gurobipy as gp
import numpy as np
from gurobipy import GRB
from sklearn.linear_model import LinearRegression
from ..mpr import *
from tqdm import tqdm

class MOPRInteger():
    def __init__(self, dataset, curation_set, similarity_scores, model=None):
        print("using Gurobi IP...")
        self.n = dataset.shape[0]
        self.d = dataset.shape[1]
        self.dataset = dataset

        # if curation_set is None: ## If no curation set is provided, compute MPR over the retrieval set
        #     self.curation_set = self.dataset
        # else:
        self.curation_set = curation_set
        self.m = self.curation_set.shape[0]

        self.expanded_dataset = np.concatenate((self.dataset, self.curation_set), axis=0)

        self.similarity_scores = similarity_scores.squeeze()

        if model is None:
            print("No function class provided. Defaulting to SKLearn Linear Regression")
            self.model = LinearRegression()            
        else:
            self.model = model

    def fit(self, k, num_iter, rho):
        self.problem = gp.Model("mixed_integer_optimization")
        self.a = self.problem.addVars(self.m, vtype=GRB.BINARY, name="a")
        # self.problem.params.SoftMemLimit = 16

        obj = gp.quicksum(self.similarity_scores[i]*self.a[i] for i in range(self.m))
        self.problem.setObjective(obj, sense=GRB.MAXIMIZE)
        self.problem.addConstr(sum([self.a[i] for i in range(self.n)]) == k, "constraint_sum_a")
        self.problem.optimize()
       
        for index in tqdm(range(num_iter)):
            gurobi_solution = np.array([self.a[i].x for i in range(len(self.a))])

            # mpr, c = getMPR(gurobi_solution, self.dataset, k, self.curation_set, self.model)
            rep, c = mpr(self.dataset, self.curation_set, gurobi_solution, self.model, return_cx=True)

            # self.sup_function(gurobi_solution, k)
            # c = self.model.predict(self.expanded_dataset)
            # c /= np.linalg.norm(c)
            # c *= c.shape[0]
            # mpr = np.abs(np.sum((gurobi_solution/k)*c[:self.n])-np.sum((1/self.m)*c[self.n:]))
            if rep < rho:
                print("constraints satisfied, exiting early")
                print("\t", np.abs(np.sum((gurobi_solution/k)*c[self.n:])-np.sum((1/self.m)*c[self.n:])))
                print("\t", rho)
                break
            
            self.max_similarity(c, k, rho, index)

            if self.problem.status == 3:
                print("Constraints infeasible, rho = {}".format(rho))
                print(self.problem.NumConstrs)
                return None
            else:
                print(self.problem.ObjVal)
        return gurobi_solution


    def max_similarity(self, c, k, rho, linear_constraint_index):
        sum_a_c = gp.quicksum([self.a[i] * c[:self.n][i] for i in range(self.n)])
        sum_c = gp.quicksum(c[self.n:])
        self.problem.addConstr(((1/k)*sum_a_c - (1/self.m)*sum_c) <= rho, name="linear_constraint_{}".format(linear_constraint_index))
        self.problem.addConstr(((1/k)*sum_a_c - (1/self.m)*sum_c) >= -rho, name="neg_linear_constraint_{}".format(linear_constraint_index))
        self.problem.optimize()
        self.problem.update()