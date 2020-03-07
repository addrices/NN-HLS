#pragma once
#include <ap_int.h>
#include <hls_stream.h>
#include <assert.h>
#include <iostream>
#include "util.h"
#include "Padding.h"

template<unsigned P,unsigned ABit,unsigned WBit,unsigned MBit>
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

template<unsigned MBit,
unsigned OBit,
unsigned M2Bit>
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

template<unsigned ABit,unsigned MBit>
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

template<unsigned WinSize,
unsigned InChannel,
unsigned Stride,
unsigned IOBit,
unsigned MatrixW,unsigned MatrixH>			//Conv Stream from image
void ConvStreamGenerator(hls::stream<ap_uint<IOBit*InChannel> >& in,hls::stream<ap_uint<IOBit*InChannel> >& out,unsigned reps = 1){
	ap_uint<IOBit*InChannel> Local1[WinSize][MatrixW];
	ap_uint<IOBit*InChannel> temp;
	for(unsigned rep = 0;rep < reps;rep++){
		unsigned line = 0;
		for(unsigned i = 0;i < WinSize-1;i++){
			for(unsigned j = 0;j < MatrixW;j++){
				Local1[i][j] = in.read();
			}
			line++;
		}
		for(unsigned i = 0;i < MatrixH-WinSize+1;i++){
			for(unsigned j = 0;j < MatrixW;j++){
				Local1[line][j] = in.read();
			}
			line = (line+1)%WinSize;
			for(unsigned p = 0;p < MatrixW-WinSize+1;p+=Stride){
				for(unsigned m = 0;m < WinSize;m++){
					for(unsigned n = 0;n < WinSize;n++){
	#pragma HLS PIPELINE II=1
						out.write(Local1[(line+m)%WinSize][p+n]);
					}
				}
			}
		}
	}
}

template<unsigned KSize,
unsigned InChannel,unsigned OutChannel,
unsigned Stride,
unsigned WBit,unsigned ABit, unsigned MBit,
unsigned Size,unsigned InP,unsigned OutP>
void Conv_MulAct_ScaleBit(hls::stream<ap_uint<ABit*InP> >& in,hls::stream<ap_uint<ABit*OutP> >&out,const ap_int<WBit*InP> Weight[(InChannel/InP)*KSize*KSize*(OutChannel/OutP)][OutP],const ap_int<WBit> Bias[OutChannel],const unsigned Scale,unsigned reps = 1){
	const unsigned InPack = InChannel/InP;
	const unsigned OutPack = OutChannel/OutP;
	const unsigned Nums = ((Size-KSize)/Stride+1)*((Size-KSize)/Stride+1);
	ap_int<MBit> OutSinglePix[OutChannel];
#pragma HLS array_partition variable=OutSinglePix complete
	ap_uint<ABit*OutP> OutTemp;
	for(unsigned i = 0;i < Nums*reps;i++){
		for(unsigned j = 0;j < KSize*KSize;j++){
			if(j == 0){
				for(unsigned q = 0;q < OutChannel;q++){
					OutSinglePix[q] = Bias[q];
				}
			}
			for(unsigned m = 0;m < InPack;m++){
				ap_int<ABit*InP> InTemp = in.read();
				for(unsigned n = 0;n < OutPack;n++){
#pragma HLS PIPELINE II=1
					unsigned WeightOffset = j*InPack*OutPack+m*OutPack+n;
					for(unsigned a = 0;a < OutP;a++){
#pragma HLS UNROLL
						unsigned NChannel = n*OutP+a;
						unsigned OP = n*OutP+a;	//4 3 0
						OutSinglePix[NChannel] += Dot<InP,ABit,WBit,MBit>(Weight[WeightOffset][a],InTemp);
					}
					if(m == InPack-1 && j == KSize*KSize-1){
						for(unsigned a = 0;a < OutP;a++){
							unsigned NChannel = n*OutP+a;
							OutTemp((a+1)*ABit-1,a*ABit) = SCALE_RELU_ScaleBit<ABit,MBit>(OutSinglePix[NChannel],Scale);
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

template<unsigned KSize, unsigned WBit, unsigned ABit, unsigned MBit,
unsigned InChannel, unsigned OutChannel,
unsigned Stride,
unsigned Size,unsigned InP,unsigned OutP>
void ConvLayer_NOPAD_ScaleBit(hls::stream<ap_uint<ABit*InChannel> >& in,hls::stream<ap_uint<ABit*OutChannel> >& out,const ap_int<WBit*InP> Weight[(InChannel/InP)*KSize*KSize*(OutChannel/OutP)][OutP],const ap_int<WBit> Bias[OutChannel],const unsigned Scale,unsigned reps = 1){
#pragma HLS DATAFLOW
	const unsigned Pixs = ((Size-KSize)/Stride+1)*((Size-KSize)/Stride+1);
	hls::stream<ap_uint<ABit*InChannel> > ConvMatrixStream;
	ConvStreamGenerator<KSize,InChannel,Stride,ABit,Size,Size>(in,ConvMatrixStream,reps);
	hls::stream<ap_uint<ABit*InP> > InPStream;
	hls::stream<ap_uint<ABit*OutP> > OutPStream;
	splitStream_Length<ABit*InChannel,ABit*InP,Pixs*KSize*KSize>(ConvMatrixStream,InPStream,reps);
	Conv_MulAct_ScaleBit<KSize,InChannel,OutChannel,Stride,WBit,ABit,MBit,Size,InP,OutP>(InPStream,OutPStream,Weight,Bias,Scale,reps);
	mergeStream_Length<ABit*OutP,ABit*OutChannel,Pixs>(OutPStream,out,reps);
}
