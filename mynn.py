import numpy as np


class MyNN:

    def __init__(self, l, q, d,
                 etall=.5, eta1=None, eta2=None,
                 ):
        self.eta1 = eta1 if eta1 is not None else etall
        self.eta2 = eta2 if eta2 is not None else etall

        self.l = l
        self.q = q
        self.d = d

        self.V = np.random.rand(d, q)
        self.W = np.random.rand(q, l)
        self.THETA = np.random.rand(l)
        self.GAMMA = np.random.rand(q)

        self.errs = list()

    def __f(self, X):
        return 1 / (1 + np.exp(-X))

    def estimate(self, X):
        X = np.asarray(X)
        ALPHA = np.matmul(X, self.V)
        B = self.__f(ALPHA - self.GAMMA)
        BETA = np.matmul(B, self.W)
        Y = self.__f(BETA-self.THETA)
        return Y

    def ek(self, X, Y):
        YB = self.estimate(X)

        Y = np.asarray(Y)
        YYB = Y - YB
        err = np.matmul(YYB, YYB)/2.0
        return err

    def bp(self, X, Y):

        X = np.asarray(X)
        Y = np.asarray(Y)
        e1 = self.eta1
        e2 = self.eta2

        ALPHA = np.matmul(X, self.V)
        B = self.__f(ALPHA - self.GAMMA)
        BETA = np.matmul(B, self.W)
        YB = self.__f(BETA-self.THETA)

        YYB = Y - YB
        err = np.matmul(YYB, YYB)/2.0
        self.errs.append(err)
        # print(err)

        G = YB * (1 - YB) * YYB
        E = B * (1 - B) * np.matmul(self.W, self.THETA)

        G1 = e1*G
        E2 = e2*E

        self.THETA = self.THETA - G1
        self.W = self.W + np.outer(B, G1)

        self.GAMMA = self.GAMMA - E2
        self.V = self.V + np.outer(X, E2)

