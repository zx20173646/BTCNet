import os
import random
from scipy.io import loadmat

prefix = 'HSI/California'
results = os.walk('.')
names = []
for root, dirs, files in os.walk(prefix, topdown=False):
    for file in files:
        if '.mat' in file:
            names.append(file)

num = len(names)
testname = random.sample(names, round(0.1 * num))
print(testname)
print(names)
for name in testname:
    names.remove(name)

print(names)

for trainname in names:
    f = open('trainpath/train.txt', 'a')
    f.write(os.path.join(prefix, trainname).replace('\\', '/'))
    f.write('\n')

for test in testname:
    f = open('testpath/test.txt', 'a')
    f.write(os.path.join(prefix, test).replace('\\', '/'))
    f.write('\n')

