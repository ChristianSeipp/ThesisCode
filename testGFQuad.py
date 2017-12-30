from GF_QUAD import *
N = 101
x = numpy.linspace(-2,2,N)
alpha =1.5625

f = lambda x: x
print(genExpIntegrate(f,x,alpha))