#pragma once
#include <ap_int.h>
#include <hls_stream.h>
#include <assert.h>
#include <iostream>
#include "conv.h"
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
 *  the module has ASize*WSize MACs  Latency=Depth
 *  HLS don't use it
 */
template<unsigned Depth,unsigned ASize,unsigned WSize,unsigned ABit,unsigned WBit,unsigned MBit>
void Orbital_Gemm_demo(ap_uint<ABit> activation[ASize][Depth],ap_int<WBit> weight[WSize][Depth],ap_int<MBit> o[ASize][WSize]){
//#pragma HLS interface ap_none port=activation
//#pragma HLS interface ap_none port=weight
//#pragma HLS interface ap_none port=o
	ap_int<WBit> W[WSize][Depth];
	ap_uint<ABit> A[ASize][Depth];
	ap_int<MBit> O[ASize][WSize];
#pragma HLS array_partition variable=W complete
#pragma HLS array_partition variable=A complete
#pragma HLS array_partition variable=O complete

	for(unsigned i = 0;i < WSize;i++)
#pragma HLS UNROLL
		for(unsigned j = 0;j < Depth;j++)
#pragma HLS UNROLL
			W[i][j] = weight[i][(i+j)%Depth];

	for(unsigned i = 0;i < ASize;i++)
#pragma HLS UNROLL
		for(unsigned j = 0;j < Depth;j++)
#pragma HLS UNROLL
			A[i][j] = activation[i][(i+j)%Depth];

	for(unsigned i = 0;i < ASize;i++)
#pragma HLS UNROLL
		for(unsigned j = 0;j < WSize;j++)
#pragma HLS UNROLL
			O[i][j] = 0;

	for(unsigned i = 0;i < Depth;i++){
#pragma HLS PIPELINE II = 1
		for(unsigned j = 0; j < ASize; j++){
#pragma HLS UNROLL
			for(unsigned k = 0; k < WSize;k++){
#pragma HLS UNROLL
				O[j][k] = O[j][k] + W[k][j] * A[j][k];
			}
		}
		for(unsigned j = 0;j < ASize;j++){
#pragma HLS UNROLL
			ap_uint<ABit> ATemp = A[j][0];
			for(unsigned d = 0;d < Depth-1;d++){
#pragma HLS UNROLL
				A[j][d] = A[j][d+1];
			}
			A[j][Depth-1] = ATemp;
		}
		for(unsigned k = 0;k < WSize;k++){
			ap_int<WBit> WTemp = W[k][0];
			for(unsigned d = 0;d < Depth-1;d++){
#pragma HLS UNROLL
				W[k][d] = W[k][d+1];
			}
			W[k][Depth-1] = WTemp;
		}
	}

	for(unsigned c = 0; c < ASize; c++){
#pragma HLS UNROLL
		for(unsigned d = 0; d < WSize; d++){
#pragma HLS UNROLL
			o[c][d] = O[c][d];
		}
	}
	return;
}

template<unsigned Depth,unsigned ASize,unsigned WSize,unsigned ABit,unsigned WBit,unsigned MBit>
ap_int<MBit*ASize*WSize> Orbital_Gemm(ap_uint<ABit*ASize*Depth> activation,ap_int<WBit*WSize*Depth> weight){
//#pragma HLS interface ap_none port=activation
//#pragma HLS interface ap_none port=weight
//#pragma HLS interface ap_none port=o
	ap_int<WBit> W[WSize][Depth];
	ap_uint<ABit> A[ASize][Depth];
	ap_int<MBit> O[ASize][WSize];
#pragma HLS array_partition variable=W complete
#pragma HLS array_partition variable=A complete
#pragma HLS array_partition variable=O complete

	for(unsigned i = 0;i < WSize;i++){
#pragma HLS UNROLL
		for(unsigned j = 0;j < Depth;j++){
#pragma HLS UNROLL
			W[i][j] = weight((i*Depth+(i+j)%Depth+1)*WBit-1,(i*Depth+(i+j)%Depth)*WBit);
//			cout << W[i][j] << ' ';
		}
//		cout << endl;
	}

	for(unsigned i = 0;i < ASize;i++){
#pragma HLS UNROLL
		for(unsigned j = 0;j < Depth;j++){
#pragma HLS UNROLL
			A[i][j] = activation((i*Depth+(i+j)%Depth+1)*ABit-1,(i*Depth+(i+j)%Depth)*ABit);
//			cout << A[i][j] << ' ';
		}
//		cout << endl;
	}

	for(unsigned i = 0;i < ASize;i++)
#pragma HLS UNROLL
		for(unsigned j = 0;j < WSize;j++)
#pragma HLS UNROLL
			O[i][j] = 0;

	for(unsigned i = 0;i < Depth;i++){
#pragma HLS PIPELINE II = 1
		for(unsigned j = 0; j < ASize; j++){
#pragma HLS UNROLL
			for(unsigned k = 0; k < WSize;k++){
#pragma HLS UNROLL
				O[j][k] = O[j][k] + W[k][j] * A[j][k];
			}
		}
		for(unsigned j = 0;j < ASize;j++){
#pragma HLS UNROLL
			ap_uint<ABit> ATemp = A[j][0];
			for(unsigned d = 0;d < Depth-1;d++){
#pragma HLS UNROLL
				A[j][d] = A[j][d+1];
			}
			A[j][Depth-1] = ATemp;
		}
		for(unsigned k = 0;k < WSize;k++){
			ap_int<WBit> WTemp = W[k][0];
			for(unsigned d = 0;d < Depth-1;d++){
#pragma HLS UNROLL
				W[k][d] = W[k][d+1];
			}
			W[k][Depth-1] = WTemp;
		}
	}
	ap_int<MBit*ASize*WSize> o;
	for(unsigned c = 0; c < ASize; c++){
#pragma HLS UNROLL
		for(unsigned d = 0; d < WSize; d++){
#pragma HLS UNROLL
			o((c*WSize+d+1)*MBit-1,(c*WSize+d)*MBit) = O[c][d];
//			cout << O[c][d] << " ";
		}
//		cout << endl;
	}
//	cout << o << endl;
	return o;
}

template<unsigned Batch,unsigned WinSize,unsigned Size,unsigned Channel,unsigned IOP,unsigned IOBit,unsigned Stride>
void ConvStreamGenerator_Batch(hls::stream<ap_uint<Batch*IOBit*IOP> >& in,hls::stream<ap_uint<Batch*IOBit*IOP> >& out,unsigned reps = 1){
	assert(Channel%IOP == 0);
	const unsigned IOPack = Channel / IOP;
	ap_uint<IOBit*Batch*IOP> Local1[WinSize][Size][IOPack];
	ap_uint<IOBit*Batch*IOP> temp;
	for(unsigned rep = 0;rep < reps;rep++){
		unsigned line = 0;
		for(unsigned i = 0;i < WinSize-1;i++){
			for(unsigned j = 0;j < Size;j++){
				for(unsigned pack = 0;pack < IOPack;pack++){
					Local1[i][j][pack] = in.read();
					// cout << "read" << hex << Local1[i][j][pack]<< endl;
				}
			}
			line++;
		}
		for(unsigned i = 0;i < Size-WinSize+1;i++){
			for(unsigned j = 0;j < Size;j++){
				for(unsigned pack = 0;pack < IOPack;pack++){
					Local1[line][j][pack] = in.read();
					// cout << "read" << hex << Local1[line][j][pack]<< endl;
				}
			}
			line = (line+1)%WinSize;
			for(unsigned p = 0;p < Size-WinSize+1;p+=Stride){
				for(unsigned pack = 0;pack < IOPack;pack++){	
					for(unsigned m = 0;m < WinSize;m++){
						for(unsigned n = 0;n < WinSize;n++){
#pragma HLS PIPELINE II=1
							out.write(Local1[(line+m)%WinSize][p+n][pack]);
							// cout << "write" << Local1[(line+m)%WinSize][p+n][pack] << endl;
						}
					}
				}
			}
		}
	}
}

template<unsigned Batch,unsigned KSize,unsigned WBit,unsigned ABit,unsigned MBit,unsigned InChannel,unsigned OutChannel,unsigned Stride,unsigned Size,unsigned InP,unsigned MidP,unsigned OutP>
void Conv_MulAct_Oribital(hls::stream<ap_uint<Batch*ABit*InP> >& in,hls::stream<ap_uint<Batch*ABit*OutP> >& out,const ap_int<WBit*KSize*KSize> Weight[OutChannel][InChannel],const ap_int<WBit> Bias[OutChannel],const unsigned Scale,unsigned reps = 1){
	assert(InChannel%InP == 0);
	assert(OutChannel%OutP == 0);
	assert(OutChannel%MidP == 0);
	const unsigned Depth = KSize*KSize;
	const unsigned InPack = InChannel / InP;
	const unsigned MidPack = OutChannel / MidP;
	const unsigned OutPack = OutChannel / OutP;
	ap_uint<ABit> InArray[InP][Batch][Depth];
	ap_int<MBit> MidArray[Batch][OutChannel];
	ap_uint<ABit> OutArray[Batch][OutChannel];
	const unsigned Nums = ((Size-KSize)/Stride+1)*((Size-KSize)/Stride+1);
//reps
	LoopRep:for(unsigned rep = 0;rep < reps;rep++){
		LoopPix:for(unsigned i = 0;i < Nums;i++){
			for(unsigned batch = 0;batch < Batch;batch++){
#pragma HLS UNROLL
				for(unsigned ochan = 0;ochan < OutChannel;ochan++){
#pragma HLS UNROLL
					MidArray[batch][ochan] = Bias[ochan];
				}
			}
			LoopInPack:for(unsigned ipack = 0;ipack < InPack;ipack++){
				Orbital_pro:for(unsigned depth = 0;depth < Depth;depth++){
#pragma HLS PIPELINE II = 1
					ap_uint<Batch*ABit*InP> InTemp = in.read();
					for(unsigned inp = 0;inp < InP;inp++){
#pragma HLS UNROLL
						for(unsigned batch = 0;batch < Batch;batch++){
#pragma HLS UNROLL
							InArray[inp][batch][depth] = InTemp((inp*Batch+batch+1)*ABit-1,(inp*Batch+batch)*ABit);
						}
					}
				}
				for(int a = 0; a < 3;a++){
					for(int b = 0;b < 3;b++){
						for(int c = 0;c < Batch;c++){
							// cout << InArray[0][c][a*3+b] << " ";
						}
					}
					// cout << endl;
				}
				for(unsigned mpack = 0;mpack < MidPack;mpack++){
					Orbital:for(unsigned inp = 0;inp < InP;inp++){
						//one orbital
						unsigned NInChannel = ipack * InP + inp;
						ap_uint<ABit> act[Batch][Depth];
						ap_int<WBit> wei[MidP][Depth];
						ap_int<MBit> res[Batch][MidP];
#pragma HLS array_partition variable=act complete
#pragma HLS array_partition variable=wei complete
#pragma HLS array_partition variable=res complete
						//read
						for(unsigned batch = 0;batch < Batch;batch++){
#pragma HLS UNROLL
							for(unsigned depth = 0;depth < Depth;depth++){
#pragma HLS UNROLL
								act[batch][depth] = InArray[inp][batch][(batch+depth)%Depth];
								// cout << act[batch][depth] << " ";
							}
							// cout << endl;
						}
						
						for(unsigned midp = 0;midp < MidP;midp++){
#pragma HLS UNROLL
							unsigned NOutChannel = mpack * MidP + midp;
							for(unsigned depth = 0;depth < Depth;depth++){
#pragma HLS UNROLL
								wei[midp][depth] = Weight[NOutChannel][NInChannel](((midp+depth)%Depth+1)*WBit-1,((midp+depth)%Depth)*WBit);
								// cout << wei[midp][depth] << " ";
							}
							// cout << endl;
						}

						for(unsigned i = 0;i < Batch;i++)
#pragma HLS UNROLL
							for(unsigned j = 0;j < MidP;j++)
#pragma HLS UNROLL
								res[i][j] = 0;

						//calculate
						for(unsigned i = 0;i < Depth;i++){
#pragma HLS PIPELINE II = 1
							for(unsigned j = 0; j < Batch; j++){
#pragma HLS UNROLL
								for(unsigned k = 0; k < MidP;k++){
#pragma HLS UNROLL
									res[j][k] = res[j][k] + wei[k][j] * act[j][k];
								}
							}
							for(unsigned j = 0;j < Batch;j++){
#pragma HLS UNROLL
								ap_uint<ABit> ATemp = act[j][0];
								for(unsigned d = 0;d < Depth-1;d++){
#pragma HLS UNROLL
									act[j][d] = act[j][d+1];
								}
								act[j][Depth-1] = ATemp;
							}
							for(unsigned k = 0;k < MidP;k++){
#pragma HLS UNROLL
								ap_int<WBit> WTemp = wei[k][0];
								for(unsigned d = 0;d < Depth-1;d++){
#pragma HLS UNROLL
									wei[k][d] = wei[k][d+1];
								}
								wei[k][Depth-1] = WTemp;
							}
						}
						//write
						for(unsigned j = 0; j < Batch; j++){
#pragma HLS UNROLL
							for(unsigned k = 0; k < MidP;k++){
#pragma HLS UNROLL
								unsigned OChannel = mpack*MidP+k;
								MidArray[j][OChannel] += res[j][k];
								// cout << " res " << res[j][k] << " Mid " << MidArray[j][OChannel];
							}
							// cout << endl;
						}
					}
				}
				if(ipack == InPack -1){
					for(unsigned opack = 0;opack < OutPack;opack++){
						ap_uint<Batch*ABit*OutP> OutTemp;
						for(unsigned outp = 0;outp < OutP;outp++){
							for(unsigned batch = 0;batch < Batch;batch++){
								unsigned NChannel = opack*OutP + outp;
								unsigned OToffset = outp*Batch + batch;
								OutArray[batch][NChannel] = SCALE_RELU_ScaleBit<ABit,MBit>(MidArray[batch][NChannel],Scale);
								// cout << OutArray[batch][NChannel] << endl;
								OutTemp((OToffset+1)*ABit-1,OToffset*ABit) = OutArray[batch][NChannel];
							}
						}
						out.write(OutTemp);
						// cout << OutTemp << endl << endl;
					}
				}
			}
		}
	}
}

template<unsigned Batch,unsigned KSize,unsigned Size,unsigned InChannel,unsigned OutChannel,unsigned InP,unsigned MidP,unsigned OutP,unsigned Stride,unsigned WBit,unsigned ABit,unsigned MBit>
void ConvLayer_NOPAD_Orbital(hls::stream<ap_uint<Batch*InP*ABit> >& in,hls::stream<ap_uint<Batch*OutP*ABit> >& out,const ap_int<WBit*KSize*KSize> Weight[OutChannel][InChannel],const ap_int<WBit> Bias[OutChannel],const unsigned Scale,unsigned reps = 1){
#pragma HLS DATAFLOW
	assert(InChannel%InP == 0);
	assert(OutChannel%OutP == 0);
	assert(OutChannel%MidP == 0);
	hls::stream<ap_uint<Batch*InP*ABit> > Conv_Str;
	ConvStreamGenerator_Batch<Batch,KSize,Size,InChannel,InP,ABit,Stride>(in,Conv_Str,reps);
	Conv_MulAct_Oribital<Batch,KSize,WBit,ABit,MBit,InChannel,OutChannel,Stride,Size,InP,MidP,OutP>(Conv_Str,out,Weight,Bias,Scale,reps);
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

