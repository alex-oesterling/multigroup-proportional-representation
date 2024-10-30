import cvxpy as cp
import numpy as np

class LinearClosedFormSolver():
    def __init__(self, retrieval_labels, curation_labels, similarity_scores):
        self.n = similarity_scores.shape[0]
        self.m = curation_labels.shape[0]
        d = retrieval_labels.shape[1]
        self.similarity_scores = similarity_scores
        self.a = cp.Variable(self.n)
        self.y = cp.Variable(d)
        self.rho = cp.Parameter(nonneg=True) #similarity value
        self.retrieval_labels = retrieval_labels

        self.expanded_dataset = np.vstack((self.retrieval_labels, curation_labels))
        self.curation_labels = curation_labels

        self.curation_mean = self.curation_labels.mean(axis=0)
        self.objective = None
        self.constraints = None
        self.problem = None
        U,_,_ = np.linalg.svd(self.expanded_dataset,full_matrices=False)
        self.U = U

    
    def get_lower_upper_bounds(self, k):
        objective = cp.Maximize(self.similarity_scores.T @ self.a)
        constraints = [(self.retrieval_labels.T @ self.a)/k == self.curation_mean, sum(self.a)==k, 0<=self.a, self.a<=1]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS, verbose=True)
        lb_opt = objective.value

        #  find global optimal
        objective = cp.Maximize(self.similarity_scores.T @ self.a)
        constraints = [(self.retrieval_labels.T @ self.a)/k == self.y, sum(self.a)==k, 0<=self.a, self.a<=1]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        global_opt = objective.value

        return lb_opt, global_opt
    def fit(self, k, p):
        self.rho.value = p
        if self.objective is None:
            # New variable t to represent the L2 norm
            t = cp.Variable()
            # The objective is now to minimize t
            self.objective = cp.Minimize(t)
            
            atilde = (cp.hstack((self.a/k, np.zeros(self.curation_labels.shape[0])))-cp.hstack((np.zeros(self.retrieval_labels.shape[0]), np.ones(self.curation_labels.shape[0])/self.curation_labels.shape[0])))
            self.constraints = [
                cp.SOC(t, self.U.T @ atilde)
            ]
            # Add the original linear constraints
            self.constraints += [
                cp.sum(self.a) == k,
                0 <= self.a,
                self.a <= 1,
                self.similarity_scores.T @ self.a == self.rho
            ]
        if self.problem is None:
            self.problem = cp.Problem(self.objective, self.constraints)
        # Solve the problem
        self.problem.solve(solver=cp.ECOS, warm_start=True, verbose=True)
        return self.a.value
    
    def getClosedMPR(self, indices, k):
        UTatilde = np.linalg.norm(self.U.T@(np.hstack((indices/k, np.zeros(self.curation_labels.shape[0])))-np.hstack((np.zeros(self.retrieval_labels.shape[0]), np.ones(self.curation_labels.shape[0])/self.curation_labels.shape[0]))))
        print(UTatilde)
        print(np.sqrt((self.m*k)/(self.m+k)))
        return np.sqrt(self.m*k/(self.m+k))*UTatilde