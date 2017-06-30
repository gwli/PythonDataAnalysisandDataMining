# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# 基本的文件读写，目录操作

# <codecell>

import tempfile
from sklearn import  datasets

# <codecell>

help(datasets.fetch_mldata)

# <codecell>

import pickle

# <codecell>

d = { "abc" : [1, 2, 3], "qwerty" : [4,5,6] }

# <codecell>

afile = open(r'd.pkl','wb')

# <codecell>

help(pickle.dump)

# <markdowncell>

# 数据串行化保存

# <codecell>

pickle.dump(d,afile)

# <codecell>

afile.close()

# <codecell>

fid= open('d.pkl','rb')

# <codecell>

new_file = pickle.load(fid)
new_file

# <headingcell level=2>

# .npy 文件操作

# <markdowncell>

# 对于不大的文件使用.npy 保存

# <markdowncell>

# 相关参考：http://www.astrobetter.com/blog/2013/07/29/python-tip-storing-data/

# <codecell>

from sklearn.datasets.mldata import fetch_mldata

# <codecell>

dataset = fetch_mldata('MNIST Original')

# <headingcell level=2>

# [读取.mat 文件](http://stackoverflow.com/questions/874461/read-mat-files-in-python)

# <codecell>

from scipy.io import loadmat

# <codecell>

mat =loadmat('New/mauna-loa-atmospheric-co2.mat')

# <headingcell level=2>

# [bz2 压缩文件读取](http://pymotw.com/2/bz2/)

# <codecell>

ls mnist/

# <codecell>

import bz2

# <codecell>

lorem = open('mnist/mnist.scale.bz2','rt').read()

# <codecell>

data = bz2.decompress(lorem)

# <codecell>

data.count(44)

# <codecell>

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

# <codecell>


