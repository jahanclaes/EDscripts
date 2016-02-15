from __future__ import print_function
import sys
from copy import copy
import random
import math
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import stats

# Import all the parameters from input.dat and neighbors.txt
file = open("input.dat")
inputDict={}
for line in file:
    tokens=line.split("=")
    try:
        inputDict[tokens[0]]=float(tokens[1])
    except:
        inputDict[tokens[0]]=tokens[1]
file.close()
J = inputDict["Heisenberg1_J"]
markovSteps = int(inputDict["markovSteps"])
TimeStepsTaken = int(inputDict["TimeStepsTaken"])
StepSize=inputDict["StepSize"]
burnIn = int(.2*markovSteps)
NumWalkers=int( inputDict["NumWalkers"])
print( "J =", J)
print("beta =",2*TimeStepsTaken*StepSize)
N=0
neighbors=[]
file = open("neighbors.txt")
for line in file:
    chars = line.split() 
    site = int(chars[0])
    N=max(N,site+1)
    for i in chars[2:]:
        if int(i)>site:
            neighbors.append((site, int(i)))
file.close()
print("N =",N)

# Define H
rows, columns, entries = [],[],[]
for i in range(2**N):
    bi = '0'*(N-len(bin(i))+2)+bin(i)[2:]
    if 1==1:#bi.count('1')==N/2: # restrict to half-filling
        diagonalEntry = 0
        for a,b in neighbors:
            diagonalEntry += J*(-1)**(int(bi[a])+int(bi[b]))
            if bi[a]!=bi[b]:
                bj = bi[0:a]+bi[b]+bi[a+1:b]+bi[a]+bi[b+1:]
                j = int(bj, base=2)
                rows.append(i)
                columns.append(j)
                entries.append(2*J)
        rows.append(i)
        columns.append(i)
        entries.append(diagonalEntry)
H=csr_matrix((entries, (rows, columns)), shape=(2**N,2**N))
print("H Done")

# Construct the Staggered Magnetization matrix SM
s = [0 for i in range(N)]
s[0]=1
for i,j in neighbors:
    s[j] = s[i]*(-1)
entries = []
for i in range(2**N):
    stagMag=0
    bi = '0'*(N-len(bin(i))+2)+bin(i)[2:]
    for i in range(N):
        if bi[i]=="1":
            stagMag+=s[i]
        if bi[i]=="0":
            stagMag-=s[i]
    entries.append(stagMag**2)
SM=csr_matrix((entries,(range(2**N),range(2**N))))
print("SM Done")

def MarkovRun(TimeStepsTaken=TimeStepsTaken):
    observableList = []
    bi = ''
    for i in range(N):
        if i%2==0:
            bi+="1"
        else:
            bi+="0"
    currentVec = csr_matrix(([1.],([int(bi, base=2)],[0])), shape=(2**N,1))
    for i in range(burnIn):
        print(i,end=" ")
        sys.stdout.flush()
        for j in range(TimeStepsTaken):
            currentVec = currentVec-(StepSize*H)*currentVec
        currentVec = currentVec*(currentVec.H*currentVec)[0,0]**(-.5)
        sample = np.random.choice(currentVec.nonzero()[0],p=[abs(currentVec[k,0])**2 for k in currentVec.nonzero()[0]])
        currentVec = csr_matrix(([1.],([sample],[0])), shape=(2**N,1))
    print("\n")
    for i in range(markovSteps-burnIn):
        sys.stdout.flush()
        for j in range(TimeStepsTaken):
            currentVec = currentVec-(StepSize*H)*currentVec
        currentVec = currentVec*(currentVec.H*currentVec)[0,0]**(-.5)
        observableList.append((currentVec.H*SM*currentVec)[0,0])
        sample = np.random.choice(currentVec.nonzero()[0],p=[abs(currentVec[k,0])**2 for k in currentVec.nonzero()[0]])
        print((N+2-len(bin(sample)))*'0'+bin(sample)[2:],end=" ")
        currentVec = csr_matrix(([1.],([sample],[0])), shape=(2**N,1))
    observableResult=stats.Stats(np.array(observableList))
    return (observableResult[0], observableResult[2])

print(MarkovRun())
