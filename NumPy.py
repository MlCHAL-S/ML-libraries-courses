
import numpy as np

#########################################   BASICS
l = [1,2,3]
a = np.array([1,2,3])
# check version
print(np.__version__)
#check dimension
print(a.shape)
#check data type
print(a.dtype)
#check number of dimensions
print(a.ndim)
#check number of elements
print(a.size)
#check number of size of elements
print(a.itemsize)



print()

print(a[0])
a[0] = 10
print(a[0])

print()

b = a * np.array([1,2,3])


#########################################   LISTS VS ARRAYS
l = [1,2,3]
a = np.array([1,2,3])
# two ways of concatenating lists in python
l.append(4) 
l = l + [5]

# doubles the list
l = l * 2

#
a = a + np.array([4])
a = np.log(a)
print(l)
print(a)


#########################################   DOT PRODUCT

l1 = [1,2,3]
l2 = [4,5,6]
a1 = np.array(l1)
a2 = np.array(l2)

# iloczyn skalarny
# manual way
dot = 0
for i in range(len(l1)):
    dot += l1[i] * l2[i]
print(dot)

# 1 one
dot = np.dot(a1, a2)
print(dot)

# another way
dot = a1 @ a2
print(dot)



#########################################   SPEED OF PYTHON LISTS VS NP ARRAYS
from timeit import default_timer as timer

# create a random array of a thousand elements
a = np.random.randn(1000)
b = np.random.randn(1000)

A = list(a)
B = list(b)

T = 1000

print(a)

def dot_manually():
    dot = 0
    for i in range(len(A)):
        dot += A[i] * B[i]
    return dot

def dot_numpy():
    return np.dot(a, b)

start = timer()
for t in range(T):
    dot_manually()
end = timer()

t1 = end - start

start = timer()
for t in range(T):
    dot_numpy()
end = timer()

t2 = end - start


print(f'List calculation: {t1}')
print(f'NumPy arrays calculation: {t2}')
print(f'Ratio: {t1/t2}')


#########################################   Multidimensional (nd) arrays

a = np.array([[1,2], [3,4]])
print(a.shape)
print(a)

# regular syntax
print(a[0][1])
# shorter syntax
print(a[0,1])

# print all the columns with index 2
print(a[:,1])
# it doesn't work the other way around print(a[1,:])


# transpose
print(a.T)

# inverse
print(np.linalg.inv(a))

# determinant
print(np.linalg.det(a))

# diagonal matrix
print(np.diag(a))

#########################################   Indexing/Slicing/Boolean Indexing

a = np.array([[1,2,3,4], [5,6,7,8]])
print(a)

# takes only first row and values from 1 to 2 with two excluded
b = a[0,1:2]
print(b)

# takes only column 3
b = a[:,3]
print(b)


# no idea what is does
b = a[-1, -1]
print(b)


# both elements in the first row aren't > 2 then it's FALSE
bool_idx = a > 2
print(bool_idx)

# print only the elements from a where the condition is met
print(a[bool_idx])

# and it all shorter
print(a[a > 2])

# array which still has the same size

# we create an array all the elements where the condition is met with -1
b = np.where(a > 2, a, -1)


a = np.array([10, 19, 30, 41, 50, 61])
print(a)

# fancy indexing 
print(a[[1,3,5]])

# flatten makes it one dimensional array
even = np.argwhere(a%2==0).flatten()

print(a[even])


#########################################   Reshaping

# create an array with 1-6 elements
a = np.arange(1,7)
print(a)
print(a.shape)

# reshaping that (dunno what for)
b = a[np.newaxis, :]
print(b)

b = a[:, np.newaxis]
print(b)

#########################################   CONCATENATION

a = np.array([[1,2], [3,4]])
print(a)
b = np.array([[5,6]])

c = np.concatenate((a, b.T), axis=1)
print(c)


a = np.array([1,2,3,4])
b = np.array([5,6,7,8])

# hstack, vstack
c = np.vstack((a, b))
print(c)
#########################################   BROADCASTING

x = np.array([[1,2,3], [4,5,6], [1,2,3], [4,5,6]])
print(x)

# you don't have to add the rest of the dimensions
a = np.array([0,1,0])

y = x + a
print(y)



a = np.array([[7,8,9,10,11,12,13], [17,18,19,20,21,22,23]])
print(a)
print(a.sum(axis=1)) # sum, mean, var, std, min, max


#########################################   DATATYPE


# you can specify the datatype
x = np.array([1, 2, 3], dtype=np.int64)
print(x)


#########################################   GENERATING ARRAYS

a = np.zeros((2,3), dtype=np.int32)
print(a)

a = np.ones((2,3), dtype=np.int32)
print(a)


# creating 2 by 3 matrix filled with fives
a = np.full((2,3), 5.0)

# identity matrix
a = np.eye(3)

# create an array from 0 to 19
a = np.arange(20)
print(a)

# create an array filled numbers starting from 0 to 10 and spacing the elements
a = np.linspace(0, 10, 5)
print(a)



#########################################   GENERATING ARRAYS


a = np.random.random((2,3)) # it's called the uniform distribution (random numbers from 0 to 1)

a = np.random.randn(3, 2) # normal/Gaussian distribution

a = np.random.randn(1000)
print(a.mean(), a.var())


a = np.random.randint(3, 10, size=(3,3))
print(a)

a = np.random.choice(5, size=10) # a = np.random.choice([-8, 5], size=10)
print(a)


#########################################   GENERATING ARRAYS

a = np.array([[1,2], [3,4]])
eigenvalues, eigenvectors = np.linalg.eig(a)

# e_vec * e_val = A * e_vec
b = eigenvectors[:,0] * eigenvalues[0]
print(b)

c = a @ eigenvectors[:,0]
print(b)

# when we compare two arrays
print(np.allclose(b, c))


#########################################   SOLVING A REAL PROBLEM

# Q: The admission fee at a small fair is $1.50 for children and $4.00 for adults. 
# On a certain day, 2200 people enter the fair and $5050 is collected. 
# How many children and how many adults attended?

A = np.array([[1,1], [1.5, 4.0]])
b = np.array([2200, 5050])

x = np.linalg.solve(A, b)
print(x)

#########################################   LOADING A CSV FILE

data = np.genfromtxt('spambase.csv', delimiter=',', dtype=np.float32)
