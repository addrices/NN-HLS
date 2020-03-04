#pragma once
#include <ap_int.h>
#include <hls_stream.h>
#include <assert.h>
#include <iostream>
#include "util.h"
#include "Padding.h"

template<int P,int ABit,int WBit,int MBit>
ap_int<MBit> Dot(ap_int<P*WBit> weights,ap_int<P*ABit> in){
	ap_int<MBit> acc = 0;

	for (unsigned p = 0; p < P; p++) {
#pragma HLS UNROLL
		ap_int<MBit> result;
		ap_int<WBit> temp_w = weights( (p+1)*WBit-1, p*WBit );
		ap_uint<ABit> temp_in = in( (p+1)*ABit-1, p*ABit );
		result = temp_w*temp_in;
		acc += result;
	}

	return acc;
}

template<int MBit,
int OBit,
int M2Bit>
ap_int<OBit> ACTIVATE(ap_int<MBit> in,const ap_int<M2Bit> factorA,const ap_int<M2Bit> factorB){
	const ap_int<OBit> limit = (1 << (OBit-1))-1;
	ap_int<M2Bit+MBit> temp_result = in*factorA+factorB;
	ap_int<OBit> result;
	if(temp_result < 0)
		result = 0;
	else if(temp_result > limit)
		result = limit;
	else
		result = temp_result;
	//std::cout << "act  result" << result << std::endl;
	return result;
}

template<int ABit,int WBit,int MBit>
ap_uint<ABit> SCALE_RELU_ScaleBit(ap_int<MBit> in,const unsigned ScaleBit){
	const ap_uint<ABit> limit = (1 << ABit)-1;
	const unsigned HALF = (1 << (ScaleBit-1));
	ap_int<MBit> temp_result;
	if(in < 0)
		temp_result = (in - HALF) >> ScaleBit;
	else
		temp_result = (in + HALF - 1) >> ScaleBit;
	ap_uint<ABit> result;
	if(temp_result < 0)
		result = 0;
	else if(temp_result > limit)
		result = limit;
	else
		result = temp_result(ABit-1,0);
	return result;
}

template<int Winsize,
int InChannel,
int Stride,
int IOBit,
int MatrixW,int MatrixH>			//Conv Stream from image
void ConvStreamGenerator_New(hls::stream<ap_uint<IOBit*InChannel> >& in,hls::stream<ap_uint<IOBit*InChannel> >& out,unsigned reps = 1){
	assert(MatrixW%Stride == 0);
	int InP = 0;
	int Ox = 0;int Oy = 0;
	ap_uint<IOBit*InChannel> Local1[Winsize][MatrixW];
	ap_uint<IOBit*InChannel> temp;
	for(unsigned rep = 0;rep < reps;rep++){
	unsigned line = 0;
	for(int i = 0;i < Winsize-1;i++){
		for(int j = 0;j < MatrixW;j++){
			Local1[i][j] = in.read();
		}
		line++;
	}
	for(int i = 0;i < MatrixH-Winsize+1;i++){
		for(int j = 0;j < MatrixW;j++){
			Local1[line][j] = in.read();
		}
		line = (line+1)%Winsize;
		for(int p = 0;p < MatrixW-Winsize+1;p+=Stride){
			for(int m = 0;m < Winsize;m++){
				for(int n = 0;n < Winsize;n++){
#pragma HLS PIPELINE II=1
					int offset = m*Winsize+n;
					out.write(Local1[(line+m)%Winsize][p+n]);
				}
			}
		}
	}
	}
}

template<int KSize,
int InChannel,int OutChannel,
int Stride,
int WBit,int ABit, int MBit,
int Size,int InP,int OutP>
void Conv_MulAct_ScaleBit_NEW(hls::stream<ap_uint<ABit*InP> >& in,hls::stream<ap_uint<ABit*OutP> >&out,const ap_int<WBit*InP> Weight[(InChannel/InP)*KSize*KSize*(OutChannel/OutP)][OutP],const ap_int<WBit> Bias[OutChannel],const unsigned Scale,unsigned reps = 1){
	const unsigned InPack = InChannel/InP;
	const unsigned OutPack = OutChannel/OutP;
	const unsigned Nums = ((Size-KSize)/Stride+1)*((Size-KSize)/Stride+1);
	ap_int<MBit> OutSinglePix[OutChannel];
#pragma HLS array_partition variable=OutSinglePix complete
	ap_uint<ABit*OutP> OutTemp;
	for(int i = 0;i < Nums*reps;i++){
		for(int j = 0;j < KSize*KSize;j++){
			if(j == 0){
				for(int q = 0;q < OutChannel;q++){
					OutSinglePix[q] = Bias[q];
				}
			}
			for(int m = 0;m < InPack;m++){
				ap_int<ABit*InP> InTemp = in.read();
				for(int n = 0;n < OutPack;n++){
#pragma HLS PIPELINE II=1
					unsigned WeightOffset = j*InPack*OutPack+m*OutPack+n;
					for(int a = 0;a < OutP;a++){
#pragma HLS UNROLL
						unsigned NChannel = n*OutP+a;
						int OP = n*OutP+a;	//4 3 0
						OutSinglePix[NChannel] += Dot<InP,ABit,WBit,MBit>(Weight[WeightOffset][a],InTemp);
					}
					if(m == InPack-1 && j == KSize*KSize-1){
						for(int a = 0;a < OutP;a++){
							unsigned NChannel = n*OutP+a;
							OutTemp((a+1)*ABit-1,a*ABit) = SCALE_RELU_ScaleBit<ABit,WBit,MBit>(OutSinglePix[NChannel],Scale);
						}
					}
					if(m == InPack-1 && j == KSize*KSize-1){
						out.write(OutTemp);
					}
				}
			}
		}
	}
}

template<int KSize, int WBit, int ABit, int MBit,
int InChannel, int OutChannel,
int Stride,
int Size,int InP,int OutP>
void ConvLayer_NOPAD_ScaleBit(hls::stream<ap_uint<ABit*InChannel> >& in,hls::stream<ap_uint<ABit*OutChannel> >& out,const ap_int<WBit*InP> Weight[(InChannel/InP)*KSize*KSize*(OutChannel/OutP)][OutP],const ap_int<WBit> Bias[OutChannel],const unsigned Scale,unsigned reps = 1){
#pragma HLS DATAFLOW
	const int Pixs = ((Size-KSize)/Stride+1)*((Size-KSize)/Stride+1);
	hls::stream<ap_uint<ABit*InChannel> > ConvMatrixStream;
	ConvStreamGenerator_New<KSize,InChannel,Stride,ABit,Size,Size>(in,ConvMatrixStream,reps);
	hls::stream<ap_uint<ABit*InP> > InPStream;
	hls::stream<ap_uint<ABit*OutP> > OutPStream;
	splitStream_Length<ABit*InChannel,ABit*InP,Pixs*KSize*KSize>(ConvMatrixStream,InPStream,reps);
	Conv_MulAct_ScaleBit_NEW<KSize,InChannel,OutChannel,Stride,WBit,ABit,MBit,Size,InP,OutP>(InPStream,OutPStream,Weight,Bias,Scale,reps);
	mergeStream_Length<ABit*OutP,ABit*OutChannel,Pixs>(OutPStream,out,reps);
}
