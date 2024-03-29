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

template<unsigned WinSize,
unsigned Channel,
unsigned IOBit,
unsigned Size,
unsigned IOP>
void PoolStreamGenerator_IOP(hls::stream<ap_uint<IOP*IOBit> >& in,hls::stream<ap_uint<IOP*IOBit> >& out,unsigned reps = 1){
	assert(Channel%IOP == 0);
	const unsigned IOPack = Channel / IOP;
	ap_uint<IOBit*IOP> Local1[WinSize][Size][IOPack];
#pragma HLS ARRAY_PARTITION variable=Local1 complete
	ap_uint<IOBit*IOP> temp;
	for(unsigned rep = 0;rep < reps;rep++){
		unsigned line = 0;
		for(unsigned i = 0;i < Size;i++){
			for(unsigned j = 0;j < Size;j++){
				for(unsigned m = 0;m < IOPack;m++){
#pragma HLS PIPELINE II = 1
					Local1[line][j][m] = in.read();
				}
			}
			line++;
			if(line == WinSize){
				line = 0;
				for(unsigned p = 0;p < Size-WinSize+1;p+=WinSize){
						for(unsigned m = 0;m < WinSize;m++){
							for(unsigned n = 0;n < WinSize;n++){
								for(unsigned pack = 0;pack < IOPack;pack++){
#pragma HLS PIPELINE II=1
								out.write(Local1[m][p+n][pack]);
							}
						}
					}
				}
			}
		}
	}
}

template<unsigned WinSize,
unsigned InChannel,
unsigned IOBit,
unsigned MatrixW,unsigned MatrixH>		//pool Stream from image
void PoolStreamGenerator(hls::stream<ap_uint<IOBit*InChannel> >& in,hls::stream<ap_uint<IOBit*InChannel> >& out,unsigned reps = 1){
	ap_uint<IOBit*InChannel> Local1[WinSize][MatrixW];
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
			if(line == WinSize){
				flag = true;
				line = 0;
			}
			if(flag == true){
				for(unsigned p = 0;p < MatrixW-WinSize+1;p+=WinSize){
					for(unsigned m = 0;m < WinSize;m++){
						for(unsigned n = 0;n < WinSize;n++){
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
unsigned IOBit,
unsigned Channel,
unsigned Packs,
unsigned InP,
unsigned OutP>
void MaxPooling_Run(hls::stream<ap_uint<IOBit*InP> >& in, hls::stream<ap_uint<IOBit*OutP> >& out,unsigned reps = 1){
	const unsigned InPack = Channel/InP;
	const unsigned OutPack = Channel/OutP;
	ap_uint<IOBit*InP> InTemp;
	ap_uint<IOBit*OutP> OutTemp;
	ap_uint<IOBit> PoolVec[Channel][WinSize*WinSize];
	for(unsigned rep = 0; rep < reps;rep++){
		for(unsigned i = 0;i < Packs;i++){
			for(unsigned j = 0;j < WinSize*WinSize;j++){
				for(unsigned k = 0;k < InPack;k++){
					InTemp = in.read();
					for(unsigned l = 0; l < InP;l++){
#pragma HLS UNROLL
						unsigned NChannel = k*InP+l;
						PoolVec[NChannel][j] = InTemp((l+1)*IOBit-1,l*IOBit);
					}
				}
			}
			for(unsigned w = 0;w < OutPack;w++){
				for(unsigned e = 0;e < OutP;e++){
#pragma HLS UNROLL
					unsigned NChannel = w * OutP+e;
					OutTemp((e+1)*IOBit-1,e*IOBit) = Max<WinSize*WinSize,IOBit>(PoolVec[NChannel]);
				}
				out.write(OutTemp);
			}
		}
	}
}

template<unsigned WinSize,
unsigned IOBit,
unsigned Size,
unsigned Channel,
unsigned InP,
unsigned OutP>
void MaxPool_IOP(hls::stream<ap_uint<IOBit*InP> >& in, hls::stream<ap_uint<IOBit*OutP> >& out,unsigned reps = 1){
#pragma HLS DATAFLOW
//	assert(Size%WinSize == 0);
//	assert(Size%WinSize == 0);
//	assert(Channel%InP == 0);
//	assert(Channel%OutP == 0);
	const unsigned Packs = Size*Size;
	const unsigned OutPacks = (Size/WinSize)*(Size/WinSize);
	hls::stream<ap_uint<IOBit*InP> > PoolPacks;
	PoolStreamGenerator_IOP<WinSize,Channel,IOBit,Size,InP>(in,PoolPacks,reps);
	MaxPooling_Run<WinSize,IOBit,Channel,OutPacks,InP,OutP>(PoolPacks,out,reps);
}

template<unsigned WinSize,
unsigned IOBit,
unsigned MatrixH,
unsigned MatrixW,
unsigned Channel,
unsigned InP,
unsigned OutP>
void MaxPool_Channel(hls::stream<ap_uint<IOBit*Channel> >& in, hls::stream<ap_uint<IOBit*Channel> >& out,unsigned reps = 1){
#pragma HLS DATAFLOW
//	assert(MatrixH%WinSize == 0);
//	assert(MatrixW%WinSize == 0);
//	assert(Channel%InP == 0);
//	assert(Channel%OutP == 0);
	const unsigned Packs = MatrixH*MatrixW;
	const unsigned OutPacks = (MatrixH/WinSize)*(MatrixW/WinSize);
	hls::stream<ap_uint<IOBit*Channel> > PoolPacks;
	PoolStreamGenerator<WinSize,Channel,IOBit,MatrixW,MatrixH>(in,PoolPacks,reps);
	hls::stream<ap_uint<IOBit*InP> > PoolInP;
	hls::stream<ap_uint<IOBit*OutP> > PoolOutP;
	splitStream_Length<IOBit*Channel,IOBit*InP,Packs>(PoolPacks,PoolInP,reps);
	MaxPooling_Run<WinSize,IOBit,Channel,OutPacks,InP,OutP>(PoolInP,PoolOutP,reps);
	mergeStream_Length<IOBit*OutP,IOBit*Channel,OutPacks>(PoolOutP,out,reps);
}


template<unsigned WinSize,
unsigned IOBit,
unsigned MatrixH,
unsigned MatrixW,
unsigned Channel,
unsigned InP,
unsigned OutP>
void MaxPool_Channel_T(hls::stream<ap_uint<IOBit*Channel> >& in, hls::stream<ap_uint<IOBit*OutP> >& out,unsigned reps = 1){
#pragma HLS DATAFLOW
//	assert(MatrixH%WinSize == 0);
//	assert(MatrixH%WinSize == 0);
//	assert(Channel%InP == 0);
//	assert(Channel%OutP == 0);
	const unsigned Packs = MatrixH*MatrixW;
	const unsigned OutPacks = (MatrixH/WinSize)*(MatrixW/WinSize);
	hls::stream<ap_uint<IOBit*Channel> > PoolPacks;
	PoolStreamGenerator<WinSize,Channel,IOBit,MatrixW,MatrixH>(in,PoolPacks,reps);
	hls::stream<ap_uint<IOBit*InP> > PoolInP;
	splitStream_Length<IOBit*Channel,IOBit*InP,Packs>(PoolPacks,PoolInP,reps);
	MaxPooling_Run<WinSize,IOBit,Channel,OutPacks,InP,OutP>(PoolInP,out,reps);
	//mergeStream_Length<IOBit*OutP,IOBit*Channel,OutPacks>(PoolOutP,out);
}
