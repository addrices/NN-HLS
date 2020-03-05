#pragma once
// #include <ap_int.h>
// #include <hls_stream.h>
// #include <assert.h>
#include <iostream>
using namespace std;
// #include "util.h"
// #include "Padding.h"
/*                             WSize = 2
 *                               w w
 *                               w w
 *                               w w
 *                               w w
 *                               w w  DEPTH = 9
 *                               w w
 *                               w w
 *                               w w
 *                               w w
 * 
 *           a a a a a a a a a   r r
 * ASize = 3 a a a a a a a a a   r r
 *           a a a a a a a a a   r r
 * 			     Depth = 9
 * 
 *  the module has ASIZE*WSize MACs  Latency=Depth
 *  HLS don't use it
 */
template<int Depth,int ASize,int WSize,int ABit,int WBit,int MBit>
void Orbital_Gemm(ap_uint<ABit> activation[ASize][Depth],ap_int<WBit> weight[WSize][Depth],ap_int<MBit> o[ASize][WSize]){
//#pragma HLS interface ap_none port=activation
//#pragma HLS interface ap_none port=weight
//#pragma HLS interface ap_none port=o
	ap_int<WBit> W[WSize][Depth];
	ap_uint<ABit> A[ASize][Depth];
	ap_int<MBit> O[ASize][WSize];
#pragma HLS array_partition variable=W complete
#pragma HLS array_partition variable=A complete
#pragma HLS array_partition variable=O complete

	for(int i = 0;i < WSize;i++)
#pragma HLS UNROLL
		for(int j = 0;j < Depth;j++)
#pragma HLS UNROLL
			W[i][j] = weight[i][(i+j)%Depth];

	for(int i = 0;i < ASize;i++)
#pragma HLS UNROLL
		for(int j = 0;j < Depth;j++)
#pragma HLS UNROLL
			A[i][j] = activation[i][(i+j)%Depth];

	for(int i = 0;i < ASize;i++)
#pragma HLS UNROLL
		for(int j = 0;j < WSize;j++)
#pragma HLS UNROLL
			O[i][j] = 0;

	for(int i = 0;i < Depth;i++){
#pragma HLS PIPELINE II = 1
		for(int j = 0; j < ASize; j++){
#pragma HLS UNROLL
			for(int k = 0; k < WSize;k++){
#pragma HLS UNROLL
				O[j][k] = O[j][k] + W[k][j] * A[j][k];
			}
		}
		for(int j = 0;j < ASize;j++){
#pragma HLS UNROLL
			ap_uint<ABit> ATemp = A[j][0];
			for(int d = 0;d < Depth-1;d++){
#pragma HLS UNROLL
				A[j][d] = A[j][d+1];
			}
			A[j][Depth-1] = ATemp;
		}
		for(int k = 0;k < WSize;k++){
			ap_int<WBit> WTemp = W[k][0];
			for(int d = 0;d < Depth-1;d++){
#pragma HLS UNROLL
				W[k][d] = W[k][d+1];
			}
			W[k][Depth-1] = WTemp;
		}
	}

	for(int c = 0; c < ASize; c++){
#pragma HLS UNROLL
		for(int d = 0; d < WSize; d++){
#pragma HLS UNROLL
			o[c][d] = O[c][d];
		}
	}
	return;
}



 //using 9*9 systolic Ksize = 3 一个channel一个channel输入，计算输出一个一个的卷积块
 template<int IOBit,int Size,int InChannel,int OutChannel>
 void ConvStreamGenerator_systolic(hls::stream<ap_uint<IOBit*9> >& in,hls::stream<ap_uint<IOBit*9> >& out){
 	ap_uint<IOBit> Local1[Size*Size]; //maybe 奇偶Buffer
 #pragma HLS ARRAY_PARTITION variable=Local1 complete
 	ap_uint<IOBit*3*3> InTemp;
 	ap_uint<IOBit*3*3> OutTemp;
 	int outPix;
 	const unsigned Packs = Size*Size/9;
 	const unsigned Remainder = Size*Size-Packs*9;
 	for(int r = 0; r < InChannel;r++){
 #pragma HLS DATAFLOW
 		for(int i = 0;i < Size*Size;i+=9){
 #pragma HLS PIPELINE II = 1
 			InTemp = in.read();
 			for(int j = 0;j < 9;j++){
 #pragma HLS UNROLL
 				if(i+j < Size*Size)
 					Local1[i+j] = InTemp((j+1)*IOBit-1,j*IOBit);
 			}
 		}
 		for(int i = 0;i < Packs;i++){
 #pragma HLS PIPELINE II = 1
 			for(int j = 0;j < 9;j++){
 #pragma HLS UNROLL
 				OutTemp((j+1)*IOBit-1,j*IOBit) = Local1[i*9+j];
 			}
 			out.write(OutTemp);
 		}
 		for(int j = 0;j < 9;j++){
 #pragma HLS UNROLL
 			if(j < Remainder)
 				OutTemp((j+1)*IOBit-1,j*IOBit) = Local1[Packs*9+j];
 			else
 				OutTemp((j+1)*IOBit-1,j*IOBit) = 0;
 		}
 		out.write(OutTemp);
 	}
 }

// template<int KSize, int WBit, int ABit,
// int InChannel, int OutChannel,
// int Stride,
// int Size,int reps>
// void ConvLayer_systolic(hls::stream<ap_uint<ABit*InChannel> >& in,hls::stream<ap_uint<ABit*OutChannel> >& out,const ap_int<WBit> Weight[reps][OutChannel][InChannel][KSize][KSize],const ap_int<WBit> Bias[reps][OutChannel],const ap_uint<WBit> Scale){
// #pragma HLS DATAFLOW
// #pragma HLS INTERFACE ap_hs port=in
// #pragma HLS INTERFACE ap_hs port=out
// 	hls::stream<ap_uint<ABit*KSize*KSize*InChannel> > ConvMatrixStream;
// 	ConvStreamGenerator_N<KSize,InChannel,Stride,ABit,Size,Size,1,1,reps>(in,ConvMatrixStream);
// 	Conv_MulAct_N<KSize,InChannel,OutChannel,Stride,WBit,ABit,Size,reps>(ConvMatrixStream,out,Weight,Bias,Scale);
// }

// template<int InPix,		//8
// 		int OutPix,		//8
// 		int InSize,		//32
// 		int OutSize,	//32
// 		int WBit,
// 		int ABit>	//has InPix and OutPix
// void FcnnLayer_systolic(hls::stream<ap_uint<ABit*InPix> >& in,hls::stream<ap_uint<ABit*OutPix> >& out,const ap_int<WBit> Weight[OutSize][InSize],const ap_int<WBit> Bias[OutSize],const ap_uint<WBit> Scale){
// 	const ap_uint<ABit+1> limit = (1 << ABit);
// 	ap_int<ABit+WBit> result[OutSize];
// 	const int Pack = OutSize/OutPix;
// 	for(int q = 0; q < OutSize;q++){
// 		result[q] = Bias[q];
// 	}
// 	for(int i = 0;i < InSize;i+=InPix){
// 		ap_uint<ABit*InPix> Rin = in.read();
// #pragma HLS PIPELINE II = InPix
// 		for(int m = 0;m < InPix;m++){
// 			for(int n = 0; n < OutSize;n++){
// 				ap_uint<ABit> Rin_P = Rin((m+1)*ABit-1,m*ABit);
// 				result[n] += Weight[n][i+m] * Rin_P;
// 			}
// 		}
// 	}
// 	//clamp(0,1<<ABit)
// 	for(int a = 0; a < OutSize;a++){
// 		std::cout << a << "  " << result[a] << std::endl;
// 		result[a] = result[a] / Scale;
// 		if(result[a] < 0)
// 			result[a] = 0;
// 		else if(result[a] >= limit)
// 			result[a] = limit - 1;
// 	}
// 	ap_uint<ABit*OutPix> OutTemp;
// 	for(int w = 0;w < Pack;w++){
// 		for(int e = 0;e < OutPix;e++){
// #pragma HLS UNROLL
// 			ap_uint<ABit> OP = result[w*OutPix+e];
// 			OutTemp((e+1)*ABit-1,e*ABit) = OP;
// 		}
// 		out.write(OutTemp);
// 	}
// 	return;
// }

