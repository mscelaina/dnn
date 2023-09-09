import numpy as np
import matplotlib.pyplot as plt


class MyDNN:

    def __init__(self, Q, etall=.5, ETA=None):
        if len(Q) < 3:
            raise ValueError
        self.Q = Q
        self.m = len(Q) - 2

        if ETA is None:
            self.ETA = [etall,] * (self.m+1)
        elif len(ETA) != self.m+1:
            raise ValueError
        else:
            self.ETA = ETA

        self.W = [0,]
        self.THETA = [0,]
        for c in range(1, self.m+2):
            self.W.append(np.random.rand(Q[c-1], Q[c]))
            self.THETA.append(np.random.rand(Q[c]))

        self.errs = list()

    def __f(self, X):
        return 1 / (1 + np.exp(-X))

    def estimate(self, X):
        B = np.asarray(X)
        for c in range(1, self.m+2):
            B = self.__f(np.matmul(B, self.W[c]) - self.THETA[c])
        return B

    def ek(self, X, Y):
        YB = self.estimate(X)
        YYB = YB - np.asarray(Y)
        return np.matmul(YYB, YYB) / 2.0

    def bp(self, X, Y):
        B = [0,] * (self.m+2)
        Z = [0,] * (self.m+2)
        G = [0,] * (self.m+2)
        ETA = [0,] + list(self.ETA)

        B[0] = np.asarray(X)

        for c in range(1, self.m+2):
            B[c] = self.__f(np.matmul(B[c-1], self.W[c]) - self.THETA[c])

        YB = B[self.m+1]
        YYB = YB - np.asarray(Y)
        err = np.matmul(YYB, YYB) / 2.0
        self.errs.append(err)

        Z[self.m+1] = YYB

        for c in range(self.m+1, 0, -1):
            G[c] = Z[c] * B[c] * (B[c] - 1)
            Z[c-1] = -np.matmul(self.W[c], G[c])
            GC = ETA[c] * G[c]
            self.THETA[c] -= GC
            self.W[c] += np.outer(B[c-1], GC)

    def draw(self, title='Ek', label='Ek', savefig=None, rate=None):

        lerrs = len(self.errs)
        plt.plot(range(lerrs), self.errs, label=label)

        if rate is not None:
            lrate = len(rate)
            rate2 = np.asarray(rate)
            nn = np.max(self.errs) // np.max(rate2)
            if nn > 1:
                plt.plot(np.arange(0, lerrs, lerrs/lrate),
                         rate2 * nn, label='rate*%d' % nn)
            else:
                plt.plot(np.arange(0, lerrs, lerrs/lrate),
                         rate, label='rate')

        plt.title(title + str(self.Q).replace(' ', ''))
        plt.legend()
        plt.grid()
        if savefig is None:
            plt.show()
        else:
            plt.savefig('./'+str(savefig)+'.png')
        plt.clf()
