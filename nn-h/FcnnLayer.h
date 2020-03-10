#pragma once
#include <ap_int.h>
#include <hls_stream.h>
#include <iostream>

template<unsigned InPix,
		unsigned OutPix,
		unsigned InSize,
		unsigned OutSize,
		unsigned WBit,
		unsigned ABit,
		unsigned MBit>	//has InPix and OutPix
void FcnnLayer_ScaleBit(hls::stream<ap_uint<ABit*InPix> >& in,hls::stream<ap_uint<ABit*OutPix> >& out,const ap_int<WBit> Weight[OutSize][InSize],const ap_int<WBit> Bias[OutSize],const unsigned ScaleBit,unsigned reps = 1){
	const ap_uint<ABit+1> limit = (1 << ABit);
	ap_int<MBit> result[OutSize];
	const unsigned Pack = OutSize/OutPix;
	for(unsigned rep = 0; rep < reps;rep++){
		for(unsigned q = 0; q < OutSize;q++){
			result[q] = Bias[q];
		}
		for(unsigned i = 0;i < InSize;i+=InPix){
			ap_uint<ABit*InPix> Rin = in.read();
			for(unsigned m = 0;m < InPix;m++){
	#pragma HLS PIPELINE II = 1
				for(unsigned n = 0; n < OutSize;n++){
					ap_uint<ABit> Rin_P = Rin((m+1)*ABit-1,m*ABit);
					result[n] += Weight[n][i+m] * Rin_P;
				}
			}
		}
		//clamp(0,1<<ABit)
		for(unsigned a = 0; a < OutSize;a++){
			std::cout << a << "  " << result[a];
			const unsigned HALF = (1 << (ScaleBit-1));
			if(result[a] < 0)
				result[a] = 0;
			result[a] = (result[a]+HALF) >> ScaleBit;
			if(result[a] >= limit)
				result[a] = limit - 1;
			std::cout << " out " << result[a] << std::endl;
		}
		ap_uint<ABit*OutPix> OutTemp;
		for(unsigned w = 0;w < Pack;w++){
			for(unsigned e = 0;e < OutPix;e++){
	#pragma HLS UNROLL
				ap_uint<ABit> OP = result[w*OutPix+e];
				OutTemp((e+1)*ABit-1,e*ABit) = OP;
			}
			out.write(OutTemp);
		}
	}
	return;
}
