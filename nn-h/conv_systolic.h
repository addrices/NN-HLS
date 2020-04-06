#pragma once
#include <ap_int.h>
#include <hls_stream.h>
#include <assert.h>
#include <iostream>
#include "conv.h"
#include "util.h"
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
ap_int<MBit*ASize*WSize> Orbital_Gemm_Debug(ap_uint<ABit*ASize*Depth> activation,ap_int<WBit*WSize*Depth> weight){
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
			cout << W[i][j] << ' ';
		}
		cout << endl;
	}

	for(unsigned i = 0;i < ASize;i++){
#pragma HLS UNROLL
		for(unsigned j = 0;j < Depth;j++){
#pragma HLS UNROLL
			A[i][j] = activation((i*Depth+(i+j)%Depth+1)*ABit-1,(i*Depth+(i+j)%Depth)*ABit);
			cout << A[i][j] << ' ';
		}
		cout << endl;
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
	for(unsigned c = 0; c < WSize; c++){
#pragma HLS UNROLL
		for(unsigned d = 0; d < ASize; d++){
#pragma HLS UNROLL
			o((c*ASize+d+1)*MBit-1,(c*ASize+d)*MBit) = O[d][c];
			cout << O[d][c] << " ";
		}
		cout << endl;
	}
	return o;
}
template<unsigned Depth,unsigned ASize,unsigned WSize,unsigned ABit,unsigned WBit,unsigned MBit>
ap_int<MBit*ASize*WSize> Orbital_Gemm(ap_uint<ABit*ASize*Depth> activation,ap_int<WBit*WSize*Depth> weight){
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
			// cout << W[i][j] << ' ';
		}
		// cout << endl;
	}

	for(unsigned i = 0;i < ASize;i++){
#pragma HLS UNROLL
		for(unsigned j = 0;j < Depth;j++){
#pragma HLS UNROLL
			A[i][j] = activation((i*Depth+(i+j)%Depth+1)*ABit-1,(i*Depth+(i+j)%Depth)*ABit);
			// cout << A[i][j] << ' ';
		}
		// cout << endl;
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
				ap_int<MBit> tem = W[k][j] * A[j][k];
#pragma HLS resource variable=tem core=Mul_LUT
				O[j][k] = O[j][k] +tem;
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
	for(unsigned c = 0; c < WSize; c++){
#pragma HLS UNROLL
		for(unsigned d = 0; d < ASize; d++){
#pragma HLS UNROLL
			o((c*ASize+d+1)*MBit-1,(c*ASize+d)*MBit) = O[d][c];
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

template<unsigned Batch,unsigned WinSize,unsigned Size,unsigned Channel,unsigned IOP,unsigned IOBit,unsigned Stride>
void ConvStreamGenerator_Batch_CF(hls::stream<ap_uint<Batch*IOBit*IOP> >& in,hls::stream<ap_uint<Batch*IOBit*IOP> >& out,unsigned reps = 1){
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
				for(unsigned m = 0;m < WinSize;m++){
					for(unsigned n = 0;n < WinSize;n++){
						for(unsigned pack = 0;pack < IOPack;pack++){	

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

template<unsigned OutChannel,unsigned InChannel,unsigned Pixs,unsigned Depth,unsigned Batch,unsigned ABit,unsigned InP,unsigned MidP>
void ActGenerator(hls::stream<ap_uint<Batch*ABit*InP> >& in,hls::stream<ap_uint<InP*ABit*Batch*Depth> >& out,unsigned reps = 1){
	const unsigned InPack = InChannel / InP;
	const unsigned MidPack = OutChannel / MidP;
	// unsigned num = 0;
	for(unsigned rep = 0;rep < reps;rep++){
		for(unsigned i = 0;i < Pixs;i++){
			for(unsigned inp = 0;inp < InPack;inp++){
				ap_uint<ABit*Batch*Depth> act[InP];
				for(unsigned depth = 0;depth < Depth;depth++){
					ap_uint<Batch*ABit*InP> InTemp = in.read();
					for(unsigned inp = 0;inp < InP;inp++){
#pragma HLS UNROLL
						for(unsigned batch = 0;batch < Batch;batch++){
#pragma HLS UNROLL
							act[inp]((batch*Depth+depth+1)*ABit-1,(batch*Depth+depth)*ABit) = InTemp((inp*Batch+batch+1)*ABit-1,(inp*Batch+batch)*ABit);
						}
					}
				}
				for(unsigned m = 0;m < MidPack;m++){
					ap_uint<InP*ABit*Batch*Depth> OutTemp;
					for(unsigned inp = 0;inp < InP;inp++){
#pragma HLS UNROLL
						OutTemp((inp+1)*ABit*Batch*Depth-1,inp*ABit*Batch*Depth) = act[inp];
					}
					out.write(OutTemp);
					// num++;
				}
			}
		}
	}
	// cout << "act generator " << num << endl;
}

template<unsigned OutChannel,unsigned InChannel,unsigned Pixs,unsigned Depth,unsigned Batch,unsigned WBit,unsigned InP,unsigned MidP>
void WeiGenerator(const ap_int<WBit*Depth> Weight[OutChannel][InChannel],hls::stream<ap_int<InP*WBit*MidP*Depth> >& out,unsigned reps = 1){
	const unsigned InPack = InChannel / InP;
	const unsigned MidPack = OutChannel / MidP;
	// unsigned num = 0;
	for(unsigned rep = 0;rep < reps;rep++){
		for(unsigned i = 0;i < Pixs;i++){
			ap_int<WBit*MidP*Depth> wei[InP];
			for(unsigned ipack = 0;ipack < InPack;ipack++){
				for(unsigned mpack = 0;mpack < MidPack;mpack++){
#pragma HLS PIPELINE II = Depth
					Orbital:for(unsigned inp = 0;inp < InP;inp++){
#pragma HLS UNROLL
						//one orbital
						unsigned NInChannel = ipack * InP + inp;
						for(unsigned midp = 0;midp < MidP;midp++){
#pragma HLS UNROLL
							unsigned NOutChannel = mpack * MidP + midp;
							wei[inp]((midp+1)*Depth*WBit-1,midp*Depth*WBit) = Weight[NOutChannel][NInChannel];
						}
					}
					ap_int<InP*WBit*MidP*Depth> OutTemp;
					for(unsigned inp = 0;inp < InP;inp++){
#pragma HLS UNROLL
						OutTemp((inp+1)*WBit*MidP*Depth-1,inp*WBit*MidP*Depth) = wei[inp];
					}
					out.write(OutTemp);
					// num++;
				}
			}
		}
	}
	// cout << "wei generator " << num << endl;
}

template<unsigned OutChannel,unsigned InChannel,unsigned Pixs,unsigned Depth,unsigned Batch,unsigned MidP,unsigned ABit,unsigned WBit,unsigned MBit,unsigned InP>
void Orbital_Gemm_Str(hls::stream<ap_uint<InP*ABit*Batch*Depth> >& activation,hls::stream<ap_int<InP*WBit*MidP*Depth> >& weight,hls::stream<ap_int<MBit*Batch*MidP> >& out,unsigned reps = 1){
	const unsigned InPack = InChannel / InP;
	const unsigned MidPack = OutChannel / MidP;
	// unsigned num = 0;
	for(int rep = 0;rep < reps;rep++){
		for(int i = 0;i < Pixs;i++){
			ap_int<MBit*Batch*MidP> res[InP];
			ap_int<MBit*Batch*MidP> res_sum;
			ap_uint<ABit*Batch*Depth> act[InP];
			ap_int<WBit*MidP*Depth> wei[InP];
			for(unsigned ipack = 0;ipack < InPack;ipack++){
				for(unsigned mpack = 0;mpack < MidPack;mpack++){
					ap_uint<InP*ABit*Batch*Depth> Atemp = activation.read();
					ap_int<InP*WBit*MidP*Depth> Wtemp = weight.read();
					for(unsigned j = 0;j < InP;j++){
#pragma HLS UNROLL
						act[j] = Atemp((j+1)*ABit*Batch*Depth-1,j*ABit*Batch*Depth);
						wei[j] = Wtemp((j+1)*WBit*MidP*Depth-1,j*WBit*MidP*Depth);
						// cout << i << " act " << act[j] << endl;
						// cout << i << " wei " << wei[j] << endl;
						res[j] = Orbital_Gemm<Depth,Batch,MidP,ABit,WBit,MBit>(act[j],wei[j]);
						// cout << i << " res " << res[j] << endl;
					}
					res_sum = 0;
					for(unsigned j = 0; j < MidP; j++){
#pragma HLS UNROLL
						for(unsigned k = 0; k < Batch;k++){
#pragma HLS UNROLL		
							for(unsigned inp = 0;inp < InP;inp++){ 
								res_sum((j*Batch+k+1)*MBit-1,(j*Batch+k)*MBit) = (ap_int<MBit>)res[inp]((j*Batch+k+1)*MBit-1,(j*Batch+k)*MBit) + (ap_int<MBit>)res_sum((j*Batch+k+1)*MBit-1,(j*Batch+k)*MBit) ;						
							}
						}
					}
					out.write(res_sum);
					// num++;
				}
			}
		}
	}
	// cout << "Orbital_Gemm " << num << endl;
}

template<unsigned OutChannel,unsigned InChannel,unsigned Pixs,unsigned Depth,unsigned Batch,unsigned ABit,unsigned WBit,unsigned MBit,unsigned InP,unsigned MidP,unsigned OutP>
void RELU_Str(hls::stream<ap_int<MBit*Batch*MidP> >& res,hls::stream<ap_uint<Batch*ABit*OutP> >& out,const ap_int<WBit> Bias[OutChannel],const unsigned Scale,unsigned reps = 1){
	const unsigned InPack = InChannel / InP;
	const unsigned MidPack = OutChannel / MidP;
	const unsigned OutPack = OutChannel / OutP;
	ap_int<MBit> MidArray[Batch*OutChannel];
	ap_uint<ABit> OutArray[Batch*OutChannel];
#pragma HLS array_partition variable=MidArray complete
#pragma HLS array_partition variable=OutArray complete
	// unsigned num = 0;
	for(unsigned rep = 0;rep < reps;rep++){
		for(unsigned i = 0;i < Pixs;i++){
			for(unsigned w = 0;w < OutChannel;w++){
#pragma HLS UNROLL
				for(unsigned q = 0;q < Batch;q++){
#pragma HLS UNROLL
					MidArray[w*Batch+q] = Bias[w];
				}
			}
			for(unsigned ipack = 0;ipack < InPack;ipack++){
				for(unsigned mpack = 0;mpack < MidPack;mpack++){
					ap_int<MBit*Batch*MidP> ResTemp = res.read();
					for(unsigned m = 0;m < MidP;m++){
#pragma HLS UNROLL
						for(unsigned b = 0;b < Batch;b++){
#pragma HLS UNROLL
							for(unsigned l = 0;l < MidPack;l++){
#pragma HLS UNROLL						
								if(mpack == l)
									MidArray[l*MidP*Batch+m*Batch+b] += ResTemp((m*Batch+b+1)*MBit-1,(m*Batch+b)*MBit);						
								// MidArray[m*Batch+b] += ResTemp((m*Batch+b+1)*MBit-1,(m*Batch+b)*MBit);
								// if(i == 0 && b == 0){
								// 	cout << ResTemp((m*Batch+b+1)*MBit-1,(m*Batch+b)*MBit) << endl;
								// }
							}
						}
					}
				}
			}
			for(unsigned opack = 0;opack < OutPack;opack++){
				ap_uint<Batch*ABit*OutP> OutTemp;
				for(unsigned outp = 0;outp < OutP;outp++){
					for(unsigned batch = 0;batch < Batch;batch++){
						unsigned NChannel = opack*OutP + outp;
						unsigned OToffset = outp*Batch + batch;
						// if(batch == 0)
						// 	cout << i << " " << MidArray[NChannel*Batch+batch] << endl;
						OutArray[NChannel*Batch+batch] = SCALE_RELU_ScaleBit<ABit,MBit>(MidArray[NChannel*Batch+batch],Scale);
						// cout << OutArray[batch][NChannel] << endl;
						OutTemp((OToffset+1)*ABit-1,OToffset*ABit) = OutArray[NChannel*Batch+batch];
					}
				}
				out.write(OutTemp);
				// num++;
				// cout << OutTemp << endl << endl;
			}
		}
	}
	// cout << "relu num " << num << endl;
}

template<unsigned Batch,unsigned KSize,unsigned WBit,unsigned ABit,unsigned MBit,unsigned InChannel,unsigned OutChannel,unsigned Stride,unsigned Size,unsigned InP,unsigned MidP,unsigned OutP>
void Conv_MulAct_Oribital_Str(hls::stream<ap_uint<Batch*ABit*InP> >& in,hls::stream<ap_uint<Batch*ABit*OutP> >& out,const ap_int<WBit*KSize*KSize> Weight[OutChannel][InChannel],const ap_int<WBit> Bias[OutChannel],const unsigned Scale,unsigned reps = 1){
	assert(InChannel%InP == 0);
	assert(OutChannel%OutP == 0);
	assert(OutChannel%MidP == 0);
	const unsigned Depth = KSize*KSize;
	const unsigned InPack = InChannel / InP;
	const unsigned MidPack = OutChannel / MidP;
	const unsigned OutPack = OutChannel / OutP;
	const unsigned Pixs = ((Size-KSize)/Stride+1)*((Size-KSize)/Stride+1);
#pragma HLS DATAFLOW
	hls::stream<ap_uint<InP*ABit*Batch*Depth> > acts;
	hls::stream<ap_int<InP*WBit*MidP*Depth> > weis;
	hls::stream<ap_int<MBit*Batch*MidP> > gemmout;
#pragma HLS stream variable=acts depth=1 off 
#pragma HLS stream variable=weis depth=1 off 
#pragma HLS stream variable=gemmout depth=1 off 
	ActGenerator<OutChannel,InChannel,Pixs,Depth,Batch,ABit,InP,MidP>(in,acts,reps);
	WeiGenerator<OutChannel,InChannel,Pixs,Depth,Batch,WBit,InP,MidP>(Weight,weis,reps);
	Orbital_Gemm_Str<OutChannel,InChannel,Pixs,Depth,Batch,MidP,ABit,WBit,MBit,InP>(acts,weis,gemmout,reps);
	RELU_Str<OutChannel,InChannel,Pixs,Depth,Batch,ABit,WBit,MBit,InP,MidP,OutP>(gemmout,out,Bias,Scale,reps);
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
	ap_uint<ABit> InArray[InP*Batch*Depth];
	ap_int<MBit> MidArray[Batch*OutChannel];
	ap_uint<ABit> OutArray[Batch*OutChannel];
#pragma HLS array_partition variable=InArray complete
#pragma HLS array_partition variable=MidArray complete
#pragma HLS array_partition variable=OutArray complete

	const unsigned Nums = ((Size-KSize)/Stride+1)*((Size-KSize)/Stride+1);
//reps
	LoopRep:for(unsigned rep = 0;rep < reps;rep++){
		LoopPix:for(unsigned i = 0;i < Nums;i++){
			for(unsigned ochan = 0;ochan < OutChannel;ochan++){
#pragma HLS UNROLL
				for(unsigned batch = 0;batch < Batch;batch++){
#pragma HLS UNROLL
					MidArray[ochan*Batch+batch] = Bias[ochan];
				}
			}
			LoopInPack:for(unsigned ipack = 0;ipack < InPack;ipack++){
				ap_uint<ABit*Batch*Depth> act[InP];
				ap_int<WBit*MidP*Depth> wei[InP];
				Orbital_pro:for(unsigned depth = 0;depth < Depth;depth++){
#pragma HLS PIPELINE II = 1
					ap_uint<Batch*ABit*InP> InTemp = in.read();
					for(unsigned inp = 0;inp < InP;inp++){
#pragma HLS UNROLL
						for(unsigned batch = 0;batch < Batch;batch++){
#pragma HLS UNROLL
							act[inp]((batch*Depth+depth+1)*ABit-1,(batch*Depth+depth)*ABit) = InTemp((inp*Batch+batch+1)*ABit-1,(inp*Batch+batch)*ABit);
						}

					}
				}
				for(unsigned mpack = 0;mpack < MidPack;mpack++){
#pragma HLS PIPELINE II = Depth+2
					Orbital:for(unsigned inp = 0;inp < InP;inp++){
#pragma HLS UNROLL
						//one orbital
						unsigned NInChannel = ipack * InP + inp;
						ap_int<MBit*Batch*MidP> res;
						for(unsigned midp = 0;midp < MidP;midp++){
#pragma HLS UNROLL
							unsigned NOutChannel = mpack * MidP + midp;
							wei[inp]((midp+1)*Depth*WBit-1,midp*Depth*WBit) = Weight[NOutChannel][NInChannel];
						}

						//calculate
						// cout << i << " act " << act[inp] << endl;
						// cout << i << " wei " << wei[inp] << endl;
						res = Orbital_Gemm<Depth,Batch,MidP,ABit,WBit,MBit>(act[inp],wei[inp]);
						// cout << i << " res " << res << endl;
						//write
						for(unsigned j = 0; j < MidP; j++){
#pragma HLS UNROLL
							for(unsigned k = 0; k < Batch;k++){
#pragma HLS UNROLL
								for(unsigned l = 0;l < MidPack;l++){
#pragma HLS UNROLL						
									if(mpack == l)
										MidArray[l*MidP*Batch+j*Batch+k] += res((j*Batch+k+1)*MBit-1,(j*Batch+k)*MBit);						
									// if(i == 0 && k == 0){
									// 	cout << res((j*Batch+k+1)*MBit-1,(j*Batch+k)*MBit) << endl;
									// }
								}
							}
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
								// if(batch == 0)
								// 	cout << i << " " << MidArray[NChannel*Batch+batch] << endl;
								OutArray[NChannel*Batch+batch] = SCALE_RELU_ScaleBit<ABit,MBit>(MidArray[NChannel*Batch+batch],Scale);
								// cout << OutArray[batch][NChannel] << endl;
								OutTemp((OToffset+1)*ABit-1,OToffset*ABit) = OutArray[NChannel*Batch+batch];
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

template<unsigned Batch,unsigned KSize,unsigned Size,unsigned InChannel,unsigned OutChannel,unsigned InP,unsigned MidP_i,unsigned MidP_o,unsigned OutP,unsigned Stride,unsigned WBit,unsigned ABit,unsigned MBit>
void ConvLayer_NOPAD_Orbital(hls::stream<ap_uint<Batch*InP*ABit> >& in,hls::stream<ap_uint<Batch*OutP*ABit> >& out,const ap_int<WBit*KSize*KSize> Weight[OutChannel][InChannel],const ap_int<WBit> Bias[OutChannel],const unsigned Scale,unsigned reps = 1){
#pragma HLS DATAFLOW
	assert(InChannel%InP == 0);
	assert(OutChannel%OutP == 0);
	assert(OutChannel%MidP_o == 0);
	assert(InP%MidP_i == 0);
	const unsigned StrLen = (InChannel/InP)*Size*Size;
	hls::stream<ap_uint<Batch*MidP_i*ABit> > Conv_Str;
	hls::stream<ap_uint<Batch*MidP_i*ABit> > in_m;
	splitStream_Length<Batch*InP*ABit,Batch*MidP_i*ABit,StrLen>(in,in_m,reps);
	ConvStreamGenerator_Batch<Batch,KSize,Size,InChannel,MidP_i,ABit,Stride>(in_m,Conv_Str,reps);
	Conv_MulAct_Oribital_Str<Batch,KSize,WBit,ABit,MBit,InChannel,OutChannel,Stride,Size,MidP_i,MidP_o,OutP>(Conv_Str,out,Weight,Bias,Scale,reps);
}

template<unsigned KSize,unsigned Size,unsigned InChannel,unsigned OutChannel,unsigned InP,unsigned MidP_i,unsigned MidP_o,unsigned OutP,unsigned Stride,unsigned WBit,unsigned ABit,unsigned MBit>
void ConvLayer_NOPAD_IOP(hls::stream<ap_uint<InP*ABit> >& in,hls::stream<ap_uint<OutP*ABit> >& out,const ap_int<WBit*MidP_i> Weight[(InChannel/MidP_i)*KSize*KSize*(OutChannel/MidP_o)][MidP_o],const ap_int<WBit> Bias[OutChannel],const unsigned Scale,unsigned reps = 1){
#pragma HLS DATAFLOW
	assert(InChannel%InP == 0);
	assert(OutChannel%OutP == 0);
	assert(InChannel%InP == 0);
	const unsigned StrLen = (InChannel/InP)*Size*Size;
	const unsigned Pixs = ((Size-KSize)/Stride+1)*((Size-KSize)/Stride+1);
	hls::stream<ap_uint<MidP_i*ABit> > Conv_Str;
	hls::stream<ap_uint<MidP_i*ABit> > in_m;
	hls::stream<ap_uint<MidP_o*ABit> > OutPStream;
	splitStream_Length<InP*ABit,MidP_i*ABit,StrLen>(in,in_m,reps);
	ConvStreamGenerator_Batch_CF<1,KSize,Size,InChannel,MidP_i,ABit,Stride>(in_m,Conv_Str,reps);
	Conv_MulAct_ScaleBit<KSize,InChannel,OutChannel,Stride,WBit,ABit,MBit,Size,MidP_i,MidP_o>(Conv_Str,OutPStream,Weight,Bias,Scale,reps);
	mergeStream_Length<MidP_o*ABit,OutP*ABit,Pixs*(OutChannel/OutP)>(OutPStream,out,reps);
}
