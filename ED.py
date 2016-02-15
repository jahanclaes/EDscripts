from copy import copy 
import random
import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

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
beta = 2*TimeStepsTaken*StepSize
NumWalkers=int( inputDict["NumWalkers"])
print "J =", J
print "beta =", beta
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
print "N =",N

# Define H
rows, columns, entries = [],[],[]
for i in range(2**N):
    bi = '0'*(N-len(bin(i))+2)+bin(i)[2:]
    if bi.count('1')==N/2: # restrict to half-filling
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
print "H Done"

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
print "SM Done"

# Find the eigenvalues relevant to calculating the density matrix
test = 1
cutoff = min(260,2**N-1)
while test>.002 and cutoff < 270: # .01 ratio between largest entry and largest cutoff entry in rho
    eigValues, eigVectors =  eigsh(H, k=cutoff, which='SA')
    print "Current Cutoff =", cutoff
    cutoff +=10
    test=math.e**(beta*eigValues[0]-beta*eigValues[-1])
    print "Current ratio =", test
rho = np.diag([math.e**(-beta*(eig-eigValues[0])) for eig in eigValues])
SMReduced=np.matrix(eigVectors).H*SM*np.matrix(eigVectors)
staggeredMagnetization = np.trace(SMReduced*rho)/np.trace(rho)
print "Observable =", staggeredMagnetization
