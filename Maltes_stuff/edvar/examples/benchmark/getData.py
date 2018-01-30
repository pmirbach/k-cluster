
import numpy as np
import matplotlib.pylab as plt
import pickle

fileName = 'Hello_WorldThu07May2015_18-46-36.pkl'
fileName = 'Hello_WorldWed15Apr2015_15-42-59.pkl'
#fileName= 'Hello_WorldThu07May2015_18-58-40.pkl'



f = open(fileName,'rb')
pickDic = pickle.load(f)
f.close()


#
print('\navailable keys of dictionary:')
for ii in pickDic.keys():
    print(ii)

print(pickDic['parameters']['aim'])
    
print('\navailable keys in paramters:')
for ii in pickDic['parameters'].keys():
    print(ii)

print('\navailable keys in results:')
for ii in pickDic['results'].keys():
    print(ii)
# example 1: print some observables
print('\nobservables')
print('phiTilde:', pickDic['results']['fopt'])
print( '<nd>:', np.sum(pickDic['results']['densMat'][0,0]))
print( '<nn>:', np.sum(pickDic['results']['twoPart']['updn'][0,0,0,0]))
print( 'S:', -0.5 + np.sqrt(0.25+pickDic['results']['S^2']))
print( 'opt Point', pickDic['results']['xopt'])


# example 2: plot optimal coefficients
plt.figure(1)
plt.plot(pickDic['parameters']['aim']['epsBath1'],pickDic['results']['coeffsC1'][0,1:],'ob')
plt.plot(pickDic['parameters']['aim']['epsBath1'],pickDic['results']['coeffsD'][0,1:],'or')
plt.plot(0,pickDic['results']['coeffsC1'][0,0],'xb')
plt.plot(0,pickDic['results']['coeffsD'][0,0],'xr')
plt.show()
