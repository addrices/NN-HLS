#pragma once
#include <ap_int.h>
#include <hls_stream.h>
#include <iostream>

template<int InPix,
		int OutPix,
		int InSize,
		int OutSize,
		int WBit,
		int ABit>	//has InPix and OutPix
void FcnnLayer_ScaleBit(hls::stream<ap_uint<ABit*InPix> >& in,hls::stream<ap_uint<ABit*OutPix> >& out,const ap_int<WBit> Weight[OutSize][InSize],const ap_int<WBit> Bias[OutSize],const unsigned ScaleBit,unsigned reps = 1){
	const ap_uint<ABit+1> limit = (1 << ABit);
	ap_int<ABit+WBit> result[OutSize];
	const int Pack = OutSize/OutPix;
	for(unsigned rep = 0; rep < reps;rep++){
	for(int q = 0; q < OutSize;q++){
		result[q] = Bias[q];
	}
	for(int i = 0;i < InSize;i+=InPix){
		ap_uint<ABit*InPix> Rin = in.read();
		for(int m = 0;m < InPix;m++){
#pragma HLS PIPELINE II = 1
			for(int n = 0; n < OutSize;n++){
				ap_uint<ABit> Rin_P = Rin((m+1)*ABit-1,m*ABit);
				result[n] += Weight[n][i+m] * Rin_P;
			}
		}
	}
	//clamp(0,1<<ABit)
	for(int a = 0; a < OutSize;a++){
		//std::cout << a << "  " << result[a] << std::endl;
		const unsigned HALF = (1 << (ScaleBit-1));
		if(result[a] < 0)
			result[a] = 0;
		result[a] = (result[a]+HALF) >> ScaleBit;
		if(result[a] >= limit)
			result[a] = limit - 1;
	}
	ap_uint<ABit*OutPix> OutTemp;
	for(int w = 0;w < Pack;w++){
		for(int e = 0;e < OutPix;e++){
#pragma HLS UNROLL
			ap_uint<ABit> OP = result[w*OutPix+e];
			OutTemp((e+1)*ABit-1,e*ABit) = OP;
		}
		out.write(OutTemp);
	}
	}
	return;
}
