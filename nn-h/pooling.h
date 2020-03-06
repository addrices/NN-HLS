#pragma once
#include <ap_int.h>
#include <hls_stream.h>
#include <iostream>

#include "util.h"
//#define	DEBUG_

template<unsigned Size,
unsigned IBit>
ap_uint<IBit> Max(ap_uint<IBit> a[Size]){
	ap_uint<IBit> max = a[0];
	for(unsigned i = 1;i < Size;i++){
		if(max < a[i])
			max = a[i];
	}
	return max;
}

template<unsigned Size,
unsigned IBit>
ap_uint<IBit>  Min(ap_uint<IBit> a[Size]){
	ap_uint<IBit> min = a[0];
	for(unsigned i = 1;i < Size;i++){
		if(min < a[i])
			min = a[i];
	}
	return min;
}

template<unsigned Winsize,
unsigned InChannel,
unsigned IOBit,
unsigned MatrixW,unsigned MatrixH>		//pool Stream from image
void PoolStreamGenerator_IOP(hls::stream<ap_uint<IOBit*InChannel> >& in,hls::stream<ap_uint<IOBit*InChannel> >& out,unsigned reps = 1){
	ap_uint<IOBit*InChannel> Local1[Winsize][MatrixW];
#pragma HLS ARRAY_PARTITION variable=Local1 complete
	ap_uint<IOBit*InChannel> temp;
	ap_uint<IOBit*InChannel> outTemp;
	for(unsigned rep = 0;rep < reps;rep++){
	unsigned line = 0;
	bool flag = false;
	for(unsigned i = 0;i < MatrixH;i++){
		for(unsigned j = 0;j < MatrixW;j++){
#pragma HLS PIPELINE II = 1
			Local1[line][j] = in.read();
		}
		line++;
		if(line == Winsize){
			flag = true;
			line = 0;
		}
		if(flag == true){
			for(unsigned p = 0;p < MatrixW-Winsize+1;p+=Winsize){
				for(unsigned m = 0;m < Winsize;m++){
					for(unsigned n = 0;n < Winsize;n++){
#pragma HLS PIPELINE II=1
						out.write(Local1[m][p+n]);
					}
				}
			}
			flag = false;
		}
	}
	}
}

template<unsigned WinSize,
unsigned IObit,
unsigned Channel,
unsigned Packs,
unsigned InP,
unsigned OutP>
void MaxPooling_IOP(hls::stream<ap_uint<IObit*InP> >& in, hls::stream<ap_uint<IObit*OutP> >& out,unsigned reps = 1){
	const unsigned InPack = Channel/InP;
	const unsigned OutPack = Channel/OutP;
	ap_uint<IObit*InP> InTemp;
	ap_uint<IObit*OutP> OutTemp;
	ap_uint<IObit> PoolVec[Channel][WinSize*WinSize];
	for(unsigned rep = 0; rep < reps;rep++){
	for(unsigned i = 0;i < Packs;i++){
		for(unsigned j = 0;j < WinSize*WinSize;j++){
			for(unsigned k = 0;k < InPack;k++){
#pragma HLS PIPELINE II = 1
				InTemp = in.read();
				for(unsigned l = 0; l < InP;l++){
					unsigned NChannel = k*InP+l;
					PoolVec[NChannel][j] = InTemp((l+1)*IObit-1,l*IObit);
				}
			}
		}
		for(unsigned w = 0;w < OutPack;w++){
			for(unsigned e = 0;e < OutP;e++){
				unsigned NChannel = w * OutP+e;
				OutTemp((e+1)*IObit-1,e*IObit) = Max<WinSize*WinSize,IObit>(PoolVec[NChannel]);
			}
			out.write(OutTemp);
		}
	}
	}
}

template<unsigned WinSize,
unsigned IObit,
unsigned MatrixH,
unsigned MatrixW,
unsigned Channel,
unsigned InP,
unsigned OutP>
void MaxPool_IOP(hls::stream<ap_uint<IObit*Channel> >& in, hls::stream<ap_uint<IObit*Channel> >& out,unsigned reps = 1){
#pragma HLS DATAFLOW
//	assert(MatrixH%WinSize == 0);
//	assert(MatrixW%WinSize == 0);
//	assert(Channel%InP == 0);
//	assert(Channel%OutP == 0);
	const unsigned Packs = MatrixH*MatrixW;
	const unsigned OutPacks = (MatrixH/WinSize)*(MatrixW/WinSize);
	hls::stream<ap_uint<IObit*Channel> > PoolPacks;
	PoolStreamGenerator_IOP<WinSize,Channel,IObit,MatrixW,MatrixH>(in,PoolPacks,reps);
	hls::stream<ap_uint<IObit*InP> > PoolInP;
	hls::stream<ap_uint<IObit*OutP> > PoolOutP;
	splitStream_Length<IObit*Channel,IObit*InP,Packs>(PoolPacks,PoolInP,reps);
	MaxPooling_IOP<WinSize,IObit,Channel,OutPacks,InP,OutP>(PoolInP,PoolOutP,reps);
	mergeStream_Length<IObit*OutP,IObit*Channel,OutPacks>(PoolOutP,out,reps);
}


template<unsigned WinSize,
unsigned IObit,
unsigned MatrixH,
unsigned MatrixW,
unsigned Channel,
unsigned InP,
unsigned OutP>
void MaxPool_IOP_T(hls::stream<ap_uint<IObit*Channel> >& in, hls::stream<ap_uint<IObit*OutP> >& out,unsigned reps = 1){
#pragma HLS DATAFLOW
//	assert(MatrixH%WinSize == 0);
//	assert(MatrixH%WinSize == 0);
//	assert(Channel%InP == 0);
//	assert(Channel%OutP == 0);
	const unsigned Packs = MatrixH*MatrixW;
	const unsigned OutPacks = (MatrixH/WinSize)*(MatrixW/WinSize);
	hls::stream<ap_uint<IObit*Channel> > PoolPacks;
	PoolStreamGenerator_IOP<WinSize,Channel,IObit,MatrixW,MatrixH>(in,PoolPacks,reps);
	hls::stream<ap_uint<IObit*InP> > PoolInP;
	splitStream_Length<IObit*Channel,IObit*InP,Packs>(PoolPacks,PoolInP,reps);
	MaxPooling_IOP<WinSize,IObit,Channel,OutPacks,InP,OutP>(PoolInP,out,reps);
	//mergeStream_Length<IObit*OutP,IObit*Channel,OutPacks>(PoolOutP,out);
}
