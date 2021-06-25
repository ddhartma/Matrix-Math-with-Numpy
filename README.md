[image1]: assets/1.png 
[image2]: assets/2.png 
[image3]: assets/3.png 
[image4]: assets/4.png 
[image5]: assets/5.png 
[image6]: assets/6.png 
[image7]: assets/7.png 
[image8]: assets/8.png 
[image9]: assets/9.png 
[image10]: assets/10.png 


# Matrix Math with NumPy

A short overview of important matrix operations. This repo is a useful reminder for Deep Learning matrix calculations.


## Content 
- [Dimensions of Tensors](#dim)
- [Vectors and Matrices](#vec_mat)
- [Basic Numpy concepts](#basic_numpy)
    - [Data Types and Shapes](#data_types)
    - [Scalars](#scalars)
    - [Vectors](#vectors)
    - [Matrices](#matrices)
    - [Tensors](#tensors)
    - [Reshaping](#reshape)
    - [Elementwise operations](#element_wise)
    - [Matrix product](#matrix_prod)
    - [Matrix transpose](#transpose)
- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)

## Dimensions of Tensors <a name="dim"></a>
- Matrix math techniques are important for Deep Learning approaches
- Tensors can be:
    - 0D: Scalars
    - 1D: Vectors
    - 2D: Matrices
    - 3D: Stack of matrices
    - 3D: List of Matrices
    - 3D: Matrix of vectors
    - 4D: Matrix of matrices
    - 4D: List of Stacks of matrices

    ![image1]

## Vectors and Matrices <a name="vec_mat"></a>
- Schematics of Vecors and Matrices 

    ![image2]

## Basic Numpy concepts <a name="basic_numpy"></a> 
### Data Types and Shapes <a name="data_types"></a>
- **ndarray** objects
- Similar to **Python lists** (but can have any number of dimensions)
- ndarray supports **fast** math operations
- Storage of any number of dimensions
- Storage of **scalars**, **vectors**, **matrices**, or **tensors**
- Types like **uint8**, **int8**, **uint16**, **int16**

### Scalars <a name="scalars"></a>

```
s = np.array(5)
print(s)
print(s.shape)
print(s + 3)

RESULTS:
------------
5
()
8
```

### Vectors <a name="vectors"></a>
```
v = np.array([1,2,3])
print(v)
print(v[1])
print(v[1:])

RESULTS:
------------
[1 2 3]
2
[2 3]
```

### Matrices <a name="matrices"></a>
```
m = np.array([[1,2,3], [4,5,6], [7,8,9]])
print(m)
print(m.shape)
print(m[1][2])

RESULTS:
------------
[[1 2 3]
 [4 5 6]
 [7 8 9]]
(3, 3)
6
```

### Tensors <a name="tensors"></a>
```
t = np.array([[[[1],[2]], [[3],[4]], [[5],[6]]],
              [[[7],[8]], [[9],[10]], [[11],[12]]], 
              [[[13],[14]], [[15],[16]], [[17],[17]]]])
print(t)
print(t.shape)
print(t[2][1][1][0])

RESULTS:
------------
[[[[ 1]
   [ 2]]

  [[ 3]
   [ 4]]

  [[ 5]
   [ 6]]]

 [[[ 7]
   [ 8]]

  [[ 9]
   [10]]

  [[11]
   [12]]]

 [[[13]
   [14]]

  [[15]
   [16]]

  [[17]
   [17]]]]
(3, 3, 2, 1)
16

```

### Reshaping <a name="reshape"></a>
```
v = np.array([1,2,3,4])
print(v)
print(v.shape)
print(v.reshape(1,4))
print(v.reshape(4,1))

RESULTS:
------------
[1 2 3 4]

(4,)

[[1 2 3 4]]

[[1]
 [2]
 [3]
 [4]]
```

### Elementwise operations <a name="element_wise"></a>
- Scalars etc. can be manipulated elementwise

    ![image3]

Elementwise Addition

```
values = [1,2,3,4,5]
print(values)
values = np.array(values) + 5
print(values)

RESULTS:
------------
[1, 2, 3, 4, 5]
[ 6  7  8  9 10]
```

Elementwise Multiplication

```
values = [1,2,3,4,5]
values = np.array(values)
x = np.multiply(values, 5)
print(x)
x = values * 5
print(x)

RESULTS:
------------
[ 5 10 15 20 25]
[ 5 10 15 20 25]
```

Elementwise matrix operations - Addition
```
a = np.array([[1,3], [5,7]])
b = np.array([[2,4], [6,8]])
print(a)
print(b)
print(a + b)

RESULTS:
------------
[[1 3]
 [5 7]]

[[2 4]
 [6 8]]

[[ 3  7]
 [11 15]]
```

Elementwise matrix operations - Multiplication
```
a = np.array([[1,3], [5,7]])
b = np.array([[2,4], [6,8]])
print(a)
print(b)
print(a * b)

RESULTS:
------------
[[1 3]
 [5 7]]

[[2 4]
 [6 8]]

[[ 2 12]
 [30 56]]
```

### Matrix product <a name="matrix_prod"></a>
- Number of columns in the left matrix must equal the number of rows in the right matrix
- The answer matrix always has the same number of rows as the left matrix and the same number of columns as the right matrix.
- Order matters. Multiplying A•B is not the same as multiplying B•A.
- Data in the left matrix should be arranged as rows., while data in the right matrix should be arranged as columns.

    ![image4]
```
a = np.array([[1,2,3,4],[5,6,7,8]])
print(a)
print(a.shape)

b = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
print(b)
print(b.shape)

c = np.matmul(a, b)
print(c)
print(c.shape)

RESULTS:
------------
[1 2 3 4]
 [5 6 7 8]]

(2, 4)

[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]]

(4, 3)

[[ 70  80  90]
 [158 184 210]]

(2, 3)
```
```
np.matmul(b, a)
# displays the following error:
# ValueError: shapes (4,3) and (2,4) not aligned: 3 (dim 1) != 2 (dim 0)
```
```
a = np.array([[1,2],[3,4]])
print(a)
print(a.shape)
print(np.dot(a,a))
print(a.dot(a))
print(np.matmul(a,a))

RESULTS:
------------
[[1 2]
 [3 4]]

(2, 2)

[[ 7 10]
 [15 22]]

[[ 7 10]
 [15 22]]

[[ 7 10]
 [15 22]]
```
### Matrix transpose <a name="transpose"></a>

![image5]

```
m = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(m)
print(m.T)

RESULTS:
------------
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]

[[ 1  5  9]
 [ 2  6 10]
 [ 3  7 11]
 [ 4  8 12]]
```
Addressing elements in the tr4ansposed matrix
```
m_t = m.T
m_t[3][1] = 200
print(m_t)
print(m)

RESULTS:
-------------
[[  1   5   9]
 [  2   6  10]
 [  3   7  11]
 [  4 200  12]]

[[  1   2   3   4]
 [  5   6   7 200]
 [  9  10  11  12]]
```


## Setup Instructions <a name="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a name="Prerequisites"></a>
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python 3.7 - 64 Bit

- Upgrade Anaconda via
```
$ conda upgrade conda
$ conda upgrade --all
```

- Optional: In case of trouble add Anaconda to your system path. Write in your CLI
```
$ export PATH="/path/to/anaconda/bin:$PATH"
```

### Clone the project <a name="Clone_the_project"></a>
- Open your Command Line Interface
- Change Directory to your project older, e.g. `cd my_github_projects`
- Clone the Github Project inside this folder with Git Bash (Terminal) via:
```
$ git clone https://github.com/ddhartma/Deep-Reinforcement-Learning-Theory-AlphaZero.git
```

- Change Directory
```
$ cd Deep-Reinforcement-Learning-Theory-AlphaZero
```

- Create a new Python environment, e.g. alpha_zero. Inside Git Bash (Terminal) write:
```
$ conda create --name alpha_zero
```

- Activate the installed environment via
```
$ conda activate alpha_zero
```

- Install the following packages (via pip or conda)
```
numpy = 1.12.1
pandas = 0.23.3
matplotlib = 2.1.0
seaborn = 0.8.1
```

- Check the environment installation via
```
$ conda env list
```

## Acknowledgments <a name="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Deep Reinforcement Learning'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a name="Further_Links"></a>

Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

## Further Links <a name="Further_Links"></a>
Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

Important web sites - Deep Learning
* Deep Learning - illustriert - [GitHub Repo](https://github.com/the-deep-learners/deep-learning-illustrated)
