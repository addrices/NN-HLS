import torch 
import numpy as np


oweight1 = np.load("./fpga/oweight1.npy")
oweight2 = np.load("./fpga/oweight2.npy")
oweight3 = np.load("./fpga/oweight3.npy")
oweight4 = np.load("./fpga/oweight4.npy")
obias1 = np.load("./fpga/obias1.npy")
obias2 = np.load("./fpga/obias2.npy")
obias3 = np.load("./fpga/obias3.npy")
obias4 = np.load("./fpga/obias4.npy")
ofcweight1 = np.load("./fpga/ofcweight1.npy")
ofcweight2 = np.load("./fpga/ofcweight2.npy")
ofcbias1 = np.load("./fpga/ofcbias1.npy")
ofcbias2 = np.load("./fpga/ofcbias2.npy")

def hex256(num):
    if(num < 0 ):
        return "{0:#0{1}X}".format(num+256,4)[2:]
    else: 
        return "{0:#0{1}X}".format(num,4)[2:]

#生成Gemm版本conv权重的函数
#const ap_int<WBIT*C2_KSIZE*C2_KSIZE> C2_W[C2_OUTCHANNEL][C2_INCHANNEL]
def weight_transform_gemm(FileName,P_tensor,InChannel,OutChannel,Ksize):
    file = open(FileName,'w',encoding = 'utf-8')
    assert(OutChannel == P_tensor.shape[0])
    assert(InChannel == P_tensor.shape[1])
    assert(Ksize == P_tensor.shape[2])
    file.write('{')
    for i in range(OutChannel):
        for j in range(InChannel):
            if(j == 0):
                file.write('{')
            for k in range(Ksize):
                for l in range(Ksize):
                    k_ = Ksize-k-1
                    l_ = Ksize-l-1
                    off = k_*Ksize + l_
                    if(k == 0 and l == 0):
                        file.write('"0x')
                    file.write(hex256(P_tensor[i][j][k_][l_]))
                    if(k == Ksize-1 and l == Ksize-1):
                        file.write('"')
            if(j != InChannel-1):
                file.write(',')
            else:
                file.write('}')
        if(i != OutChannel-1):
            file.write(',\n')
    file.write('}\n')
    file.close()

#生成Dot版本conv权重的函数
#ap_int<WBIT*C1_MIDP_I*C1_MIDP_O> C1_W[C1_KSIZE*C1_KSIZE][C1_OUTCHANNEL/C1_MIDP_O][C1_INCHANNEL/C1_MIDP_I]
def weight_transform_dot(FileName,P_tensor,InChannel,OutChannel,Ksize,Inp,Outp):
    file = open(FileName,'w',encoding = 'utf-8')
    assert(OutChannel == P_tensor.shape[0])
    assert(InChannel == P_tensor.shape[1])
    assert(Ksize == P_tensor.shape[2])
    assert(InChannel % Inp == 0)
    assert(OutChannel % Outp == 0)
    outpack = int(OutChannel/Outp)
    inpack = int(InChannel/Inp)
    file.write('{')
    for i in range(Ksize):
        for j in range(Ksize):
            file.write('{')
            for k in range(outpack):
                file.write('{')
                for l in range(inpack):
                    file.write('"0x')
                    for m in range(Inp):
                        for n in range(Outp):
                            m_ = Inp-1-m
                            n_ = Outp-1-n
                            file.write(hex256(P_tensor[k*Outp+n_][l*Inp+m_][i][j]))
                    file.write('"')
                    if( l != inpack-1):
                        file.write(',')
                file.write('}')
                if(k != outpack-1):
                    file.write(',')
            file.write('}')
            if(i != Ksize -1 or j != Ksize - 1):
                file.write(',')
            file.write('\n')
    file.write('}\n')
    file.close()
    

weight_transform_gemm('gw1',oweight1,1,16,3)
weight_transform_gemm('gw2',oweight2,16,32,3)
weight_transform_gemm('gw3',oweight3,32,64,3)
weight_transform_gemm('gw4',oweight4,64,64,3)

weight_transform_dot('dw1',oweight1,1,16,3,1,4)
weight_transform_dot('dw2',oweight2,16,32,3,8,8)
weight_transform_dot('dw3',oweight3,32,64,3,4,8)
weight_transform_dot('dw4',oweight4,64,64,3,4,4)
# weight_transform_dot('ow2',oweight2,16,32,3,8,8)