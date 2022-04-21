import numpy as np

'''A class called matrix has been created, which has an attribute named array_2d which 
is a NumPy array containing numbers in two dimensions. In the class matrix, the parameters 
are in addition to self'''


class matrix:
    
    def __init__(self, file_name):
        self.array_2d = np.empty((0,0))
        self.load_from_csv(file_name)
        self.n  = self.array_2d.shape[0]
        self.m = self.array_2d.shape[1] 
        self.standardise()
    '''The first method in class matrix is load_from_csv which has one parameter, the file name. 
This method reads the CSV file and load its data to the array_2d of matrix. 
Each row in this file is a row in array_2d which will be displayed as comma separated values.'''    
    
    def load_from_csv(self,filename):
        self.array_2d =  np.loadtxt(filename, delimiter=',')
        
        
    ''' The next method in class matrix is standardise which has no parameters. It standardises the array_2d in the matrix calling this 
method. The standardisation is done using the formula, self.array_2d[n,m]=(self.array_2d[n,m]-avg_of_coloumn[m])/(max_val[m]-min_val[m])
which implements the average values, minimum and the maximum values of the coloumns'''
        
    def standardise(self):
        avg_of_coloumn = np.mean(self.array_2d, axis =0)
        max_val = np.max(self.array_2d, axis =0)
        min_val = np.min(self.array_2d, axis =0)
        for n in range(len(self.array_2d)):
            for m in range(self.array_2d.shape[1]):
                self.array_2d[n,m]=(self.array_2d[n,m]-avg_of_coloumn[m])/(max_val[m]-min_val[m])
        #print(self.array_2d)
        return self.array_2d
    
    ''' The next method in class matrix in get_distance. This method has three parameters, two matricese which is other_matrix and weights 
and a number. This function returns a matrix containing the weighted Euclidean distance between the 
row in the matrix calling this method and each of the rows in other_matrix.
The weighted Euclidean distance is calculated by the formula rowValue += weights[colomns]**beta *((other_matrix[colomns]- cent_rows[colomns]))**2 '''
    
    def get_distance(self, other_matrix, weights, beta):
        newWeight = []
        for cent_rows in self.centroid:
            rowValue = 0.0
            for colomns in range(len(cent_rows)):
                rowValue += weights[colomns]**beta *((other_matrix[colomns]- cent_rows[colomns]))**2
                newWeight.append(rowValue)
        return np.array(newWeight)
    
    ''' The next method in class matrix in get_count_frequency. This method has no parametes.
    This method returns a dictionary mapping each element of the 
array_2d to the number of times this element appears in array_2d.'''
        
    def get_count_frequency(self):
        unique, counts= np.unique(self.S, return_counts=True)
        return dict(zip(unique,counts))


''' Moving on to functions'''


''' The first function here is get_initial_weights. This function has one parameter, an integer m. This function returns a matrix with 1 
row and m columns containing random values, each between zero and one. The sum of these m 
values should be equal to one. which is performed by using random function in numpy array 
'''
 
def get_initial_weights(m):
    weights = np.random.random(m)
    weights /= np.sum(weights)
    
    return weights

'''The next function we use is get_centroids.This function has three parameters, a matrix containing the data, the S matrix and the 
value of K.This function updates the k row in centroids.Taking into consideration those rows whose value in S is 
equal to k. It returns a matrix containing K rows and the same number of columns as the matrix containing 
the data.'''

def get_centroids(array_2d, S_matrix, K):
    if np.all(S_matrix==0):
        centroid= array_2d[np.random.randint(array_2d.shape[0], size=K), :]
        return centroid
    else:
        newCentroid=[]
        for groups in range(K):
            rowIndex=[]
            for noofrows in range(len(S_matrix)):
                if int(S_matrix[noofrows][0])==groups:
                    rowIndex.append(list(array_2d[noofrows]))
            newCentroid.append(np.mean(np.array(rowIndex), axis=0))
            #print(np.array(newCentroid))
        return np.array(newCentroid)
                
'''This next function is get_groups. This has three parameters, a matrix containing the data, and the number of clusters 
to be created (K), and a number beta (for the distance calculation). It returns a matrix S. This function uses the other functions that has been written before.'''

def get_groups(m,no_of_clusters,Beta_value):
    K= no_of_clusters
    beta= Beta_value
    new_S=[]
    m.weights= get_initial_weights(m.m)
   
    m.S  = np.zeros((m.array_2d.shape[0],1))

    i=0
    while True:
        m.centroid = get_centroids(m.array_2d, m.S, K)
        new_S=m.S
        for eachrow in m.array_2d:
            dist = m.get_distance(eachrow, m.weights, beta)
            m.S[i]=np.where(dist==min(dist))[0][0]
            i=i+1
        if (new_S!=m.S).all:
            m.centroid = get_centroids(m.array_2d, m.S, K)
            m.weights=get_new_weights(m.array_2d, m.centroid, m.S)
        
        else:
            return m

'''The next function is get_new_weights. This function takes three parameters, a matrix containing the data,
a matrix containing the centroids, and a matrix S. This function returns a new matrix weights 
with 1 row and as many columns as the matrix containing the data and the matrix containing the 
centroids. This method finally updates the new weights replacing the old ones.'''
def get_new_weights(array_2d, centroid, S_matrix):
    n= array_2d.shape[0]
    m= array_2d.shape[1]
    new_weights=np.zeros((1,m))
    K,_=centroid.shape 
    delta_value=np.zeros((1,m))
    
    for k in range(K):
        for i in range(n):
            if(S_matrix[i,:]==k):
                delta_value +=(array_2d[i,:]-centroid[k,:])**2
    for j in range(m):
        if delta_value[0,j]==0:
            new_weights[0,j]=0
        else:
            value=0
            for t in range(m):
                for beta in range(11,12):
                    value+=(delta_value[0,j]/delta_value[0,t])**(1/(beta-1))
            new_weights[0,j]=value
    return new_weights


def run_test():
    m = matrix('Data.csv')
    for k in range(3,4):
        for beta in range(11,12):
            S = get_groups(m, k, beta/10)
            print(str(k)+'-'+str(beta)+'='+str(S.get_count_frequency()))


if __name__ == '__main__':         
    run_test()


