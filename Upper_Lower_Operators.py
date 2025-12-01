import numpy as np
from numpy import where
from scipy.optimize import linear_sum_assignment
from sklearn import datasets
from sklearn.datasets import make_classification, make_blobs
import sklearn.cluster as sc
from sklearn.manifold import TSNE
from sklearn.metrics import pair_confusion_matrix, fowlkes_mallows_score
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import heapq
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from numpy import *
import time
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from sklearn import metrics
import time
import matplotlib.pyplot as plt
from pylab import *  # 支持中文
import numpy as np
from numpy import where
from sklearn import datasets
from sklearn.datasets import make_classification, make_blobs
import sklearn.cluster as sc
from sklearn.manifold import TSNE
from sklearn.metrics import fowlkes_mallows_score, pair_confusion_matrix
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
from numpy import *
from sklearn import metrics
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import numpy as np
from numpy import where
from sklearn import datasets
from sklearn.datasets import make_classification, make_blobs
import sklearn.cluster as sc
from sklearn.manifold import TSNE
from sklearn.metrics import pair_confusion_matrix, fowlkes_mallows_score
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import heapq
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from numpy import *
import time
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
def min_func(a,b):
    if a>=b:
        return b
    else:
        return a

def max_func(a,b):
    if a>=b:
        return a
    else:
        return b

def acc1(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def FCM_getCenters(U,X,m):
    N,D=np.shape(X)
    N,C=np.shape(U)

    um=U**m

    tile_X=np.tile(np.expand_dims(X,1),[1,C,1])
    tile_um=np.tile(np.expand_dims(um,-1),[1,1,D])
    temp=tile_X*tile_um

    new_C=np.sum(temp,axis=0)/np.expand_dims(np.sum(um,axis=0),axis=-1)
    return new_C

def kernel(X,sigma):                   # x1,x2为输入空间
       m,n = X.shape[0], X.shape[0]           #获取行数
       dist_matrix = np.zeros((m,n), dtype=float)   #全零核矩阵
       for i in range(m):
           for j in range(n):
               dist_matrix[i][j] = np.sum((X[i]-X[j])**2)    #向量差的平方和
       return np.exp(-0.5/sigma**2*dist_matrix)

def FCM_getClass(U):
    return np.argmax(U,axis=-1)

def initCentroids(dataSet, k):   #随机选取k个元素
	numSamples, dim = dataSet.shape   #矩阵的行数、列数
	centroids = zeros((k, dim))
	for i in range(k):
		index = int(random.uniform(0, numSamples))  #随机产生一个浮点数，然后将其转化为int型
		centroids[i, :] = dataSet[index, :]
	return centroids

# def nearest(point, cluster_centers):
#     '''
#     计算point和cluster_centers之间的最小距离
#     :param point: 当前的样本点
#     :param cluster_centers: 当前已经初始化的聚类中心
#     :return: 返回point与当前聚类中心的最短距离
#     '''
#     min_dist = 1000
#     m = np.shape(cluster_centers)[0]  # 当前已经初始化聚类中心的个数
#     for i in range(m):
#         # 计算point与每个聚类中心之间的距离
#         d = euclDistance(point, cluster_centers[i, ])
#         # 选择最短距离
#         if min_dist > d:
#             min_dist = d
#     return min_dist
#
# def initCentroids(points, k):
#     '''
#     kmeans++的初始化聚类中心的方法
#     :param points: 样本
#     :param k: 聚类中心的个数
#     :return: 初始化后的聚类中心
#     '''
#     m, n = np.shape(points)
#     cluster_centers = np.mat(np.zeros((k, n)))
#
#     # Acc_Test、随机选择一个样本点作为第一个聚类中心
#     index = np.random.randint(0, m)
#     cluster_centers[0,] = np.copy(points[index,])  # 复制函数，修改cluster_centers，不会影响points
#
#     # 2、初始化一个距离序列
#     d = [0.0 for _ in range(m)]
#
#     for i in range(1, k):
#         sum_all = 0
#         for j in range(m):
#             # 3、对每一个样本找到最近的聚类中心点
#             d[j] = nearest(points[j,], cluster_centers[0:i, ])
#             # 4、将所有的最短距离相加
#             sum_all += d[j]
#         # 5、取得sum_all之间的随机值
#         sum_all = random.uniform(0,sum_all)
#         # 6、获得距离最远的样本点作为聚类中心点
#         for j, di in enumerate(d):  # enumerate()函数用于将一个可遍历的数据对象（如列表、元组或字符串）组合为一个索引序列，同事列出数据和数据下标一般用在for循环中
#             sum_all -= di
#             if sum_all > 0:
#                 continue
#             cluster_centers[i] = np.copy(points[j,])
#             break
#     return cluster_centers

def euclDistance(vector1, vector2):
	return sqrt(sum(power(vector2 - vector1, 2)))  #求这两个矩阵的距离，vector1、2均为矩阵

def Upper_Lower_Operators(X,xi,ci,kernel_matrix,membership_matrix):
    upper = 0
    lower = 1
    for y in range(len(X)):
        upper_new=min_func(kernel_matrix[xi][y],membership_matrix[y][ci])
        if upper_new>upper:
            upper=upper_new

        lower_new = max_func(1 - (kernel_matrix[xi][y]), membership_matrix[y][ci])
        if lower_new < lower:
            lower = lower_new
    return upper/(lower+0.01)

def FCM_getCenters(U,X,m):
    N,D=np.shape(X)
    N,C=np.shape(U)

    um=U**m

    tile_X=np.tile(np.expand_dims(X,1),[1,C,1])
    tile_um=np.tile(np.expand_dims(um,-1),[1,1,D])
    temp=tile_X*tile_um

    new_C=np.sum(temp,axis=0)/np.expand_dims(np.sum(um,axis=0),axis=-1)

    return new_C

def FCM_dist(X,Centers):
    N, D = np.shape(X)
    C, D = np.shape(Centers)

    tile_x=np.tile(np.expand_dims(X,1),[1,C,1])
    tile_centers=np.tile(np.expand_dims(Centers,axis=0),[N,1,1])

    dist=np.sum((tile_x-tile_centers)**2,axis=-1)

    return np.sqrt(dist)

def FCM_getU(X,Centers,m):
    N,D=np.shape(X)
    C,D=np.shape(Centers)

    temp=FCM_dist(X,Centers)**float(2/(m-1))
    tile_temp=np.tile(np.expand_dims(temp,1),[1,C,1])

    denominator_=np.expand_dims(temp,-1)/(tile_temp)

    return 1/(np.sum(denominator_,axis=-1))

def Fuzzy_rough_fcm(X,n_centers,sigma,m,max_iter=100,theta=1e-5,seed=0):
    rng = np.random.RandomState(seed)
    N, D = np.shape(X)  # N为数据集对象数量，D为对象维度
    # 随机初始化关系矩阵
    U = rng.uniform(size=(N, n_centers))  # U为关系矩阵,维度:N×n_centers,N个样本点与n_centers个聚类中心的关系
    # 保证每行和为1
    U = U / np.sum(U, axis=1, keepdims=True)

    # centroids = initCentroids(X, n_centers)  # 在样本集中随机选取k个样本点作为初始质心
    centroids =FCM_getCenters(U, X, m)
    k_matrix=kernel(X,sigma)

    flag=False

    for i in range(max_iter):
        U_old = U.copy()
        clusterAssment = mat(zeros((N, 1)))  # 得到一个N*2的零矩阵
        for j in range(N):  # range
            ul_list=[]
            for g in range(n_centers):
                ul_list.append(Upper_Lower_Operators(X,j,g,k_matrix,U))

            minIndex=ul_list.index(min(ul_list))
            clusterAssment[j]=minIndex

        for j in range(n_centers):
            # clusterAssment[:,0].A==j是找出矩阵clusterAssment中第一列元素中等于j的行的下标，返回的是一个以array的列表，第一个array为等于j的下标
            pointsInCluster = X[nonzero(clusterAssment[:, 0].A == j)[0]]  # 将dataSet矩阵中相对应的样本提取出来
            if len(pointsInCluster)!=0:
                centroids[j, :] = mean(pointsInCluster, axis=0)  # 计算标注为j的所有样本的平均值




        U = FCM_getU(X, centroids, m)



        if np.linalg.norm(U-U_old)<theta:   #迭代终止条件：隶属矩阵变化小于阈值
            break


    if i==(max_iter-1):
        flag=True

    return centroids,U,flag



if __name__=="__main__":
    old_time = time.time()

    # iris = datasets.load_iris()
    # X=iris.data
    # y=iris.target
    #
    # Breast = datasets.load_breast_cancer()  # 乳腺癌数据集
    # X = Breast.data
    # y = Breast.target

    Wine = datasets.load_wine()  # 乳腺癌数据集
    X = Wine.data
    y = Wine.target

    # file_name = "appendicitis"
    # with open(file_name + ".txt", "r", encoding="utf-8") as f:
    #     liness = f.read().splitlines()
    # lines = [line.split(",")[:-1] for line in liness]
    # result = [line.split(",")[7:] for line in liness]
    # X = np.array(lines).astype(np.float32)
    # y = []
    # for i in range(len(result)):
    #     y.append(int(result[i][0]))
    y = np.array(y)

    X = MinMaxScaler().fit_transform(X)  # 最大最小归一化
    classes = len(np.unique(y))
    print(classes)

    p=1.5
    max_acc=0
    max_p=0.1


    for i in range(1):
        centers, U,isiternum = Fuzzy_rough_fcm(X, n_centers=classes, sigma=p, m=2, max_iter=200, theta=1e-4, seed=0)
        label = FCM_getClass(U)
        label = np.array(label)
        if isiternum:
            print("p=", p,"到达迭代次数")
            p+=0.1

            continue
        ACC = acc1(y, label)  # 聚类精度，取值范围0到1，值越大效果越好

        print("p=", p, "acc=", ACC)
        if ACC>max_acc:
            max_acc=ACC
            max_p=p


        p+=0.1

    print("最优p:",max_p,"acc=",max_acc)




    # ACC=acc1(y, label) # 聚类精度，取值范围0到1，值越大效果越好
    # LKXS = metrics.silhouette_score(X, label)  # 轮廓系数，数值越大越好
    # NMI = metrics.normalized_mutual_info_score(y, label)  # 标准化互信息，取值范围0到1，值越大效果越好
    # RI = metrics.rand_score(y, label)  # 兰德指数，取值范围0到1，值越大效果越好


    # for i in range(len(label)):
    #     if label[i]==0:
    #         label[i] = 1
    #     else:
    #         label[i]=0

    # file_name = "appendicitis"
    # with open(file_name + ".txt", "r", encoding="utf-8") as f:
    #     liness = f.read().splitlines()
    # lines = [line.split(",")[:-1] for line in liness]
    # result = [line.split(",")[7:] for line in liness]
    # X = np.array(lines).astype(np.float32)
    # y = []
    # for i in range(len(result)):
    #     y.append(int(result[i][0]))
    # y = np.array(y)

    Wine = datasets.load_wine()  # 乳腺癌数据集
    X = Wine.data
    y = Wine.target

    tsne = TSNE(n_components=2, method="exact", metric="euclidean", random_state=0)  # Tsne可视化降维到2维
    tsne.fit_transform(X)
    X = tsne.embedding_

    for cluster in np.unique(label):
            # 获取此群集的示例的行索引
        row_ix = where(label == cluster)
            # 创建这些样本的散布
        if cluster == 1:
            c = "y"
        elif cluster == 2:
            c = "g"
        # elif cluster == 3:
        #     c = "r"
        # elif cluster == 4:
        #     c = "c"
        # elif cluster == 5:
        #     c = "m"  #紫
        else:
            c = "b"
        plt.scatter(X[row_ix, 0], X[row_ix, 1], s=3, c=c)

    plt.show()




    current_time = time.time()
    print("运行时间为" + str((current_time - old_time) / 60) + "min")