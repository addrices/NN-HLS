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

template<unsigned Depth,unsigned ASize,unsigned WSize,unsigned ABit,unsigned WBit,unsigned MBit,unsigned Pixs>
void Gemm_str(hls::stream<ap_uint<ABit*ASize> >& activation,hls::stream<ap_int<WBit*WSize> >& weight,hls::stream<ap_int<MBit*ASize*WSize> >& out,unsigned reps = 1){
//#pragma HLS interface ap_none port=activation
//#pragma HLS interface ap_none port=weight
//#pragma HLS interface ap_none port=o
	ap_int<WBit> W[WSize][Depth];
	ap_uint<ABit> A[ASize][Depth];
	ap_int<MBit> O[ASize][WSize];
	ap_int<WBit> W_[WSize][Depth];
	ap_uint<ABit> A_[ASize][Depth];
	ap_int<MBit> O_[ASize][WSize];
#pragma HLS array_partition variable=W complete
#pragma HLS array_partition variable=A complete
#pragma HLS array_partition variable=O complete
#pragma HLS array_partition variable=W_ complete
#pragma HLS array_partition variable=A_ complete
#pragma HLS array_partition variable=O_ complete
	for(unsigned rep = 0;rep < reps;rep++){
		for(unsigned num = 0;num < Pixs;num++){
#pragma HLS DATAFLOW
			Wread:for(unsigned j = 0;j < Depth;j++){
#pragma HLS PIPELINE II = 1
				ap_uint<ABit*ASize> wei = weight.read();
				for(unsigned i = 0;i < WSize;i++){
#pragma HLS UNROLL
					W_[i][j] = wei(((i+j)%Depth+1)*WBit-1,((i+j)%Depth)*WBit);
				}
			}

			Aread:for(unsigned j = 0;j < Depth;j++){
#pragma HLS PIPELINE II = 1
				ap_uint<ABit*ASize> act = activation.read();
				for(unsigned i = 0;i < ASize;i++){
#pragma HLS UNROLL
					A_[i][j] = act(((i+j)%Depth+1)*WBit-1,((i+j)%Depth)*WBit);
				}
			}
			for(unsigned j = 0;j < Depth;j++){
#pragma HLS UNROLL
				for(unsigned i = 0;i < WSize;i++){
#pragma HLS UNROLL
					W[i][j] = W_[i][j];
				}
			}

			for(unsigned j = 0;j < Depth;j++){
#pragma HLS UNROLL
				for(unsigned i = 0;i < ASize;i++){
#pragma HLS UNROLL
					A[i][j] = A_[i][j];
				}
			}

			for(unsigned i = 0;i < ASize;i++)
#pragma HLS UNROLL
				for(unsigned j = 0;j < WSize;j++)
#pragma HLS UNROLL
				O[i][j] = 0;

			exec:for(unsigned i = 0;i < Depth;i++){
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
				}
			}
			out.write(o);
		}
	}
}

template<unsigned Depth,unsigned ASize,unsigned WSize,unsigned ABit,unsigned WBit,unsigned MBit>
inline ap_int<MBit*ASize*WSize> Orbital_Gemm(ap_uint<ABit*ASize*Depth> activation,ap_int<WBit*WSize*Depth> weight){
	ap_int<WBit> W[WSize][Depth];
	ap_uint<ABit> A[ASize][Depth];
	ap_int<MBit> O[ASize][WSize];
#pragma HLS array_partition variable=W complete
#pragma HLS array_partition variable=A complete
#pragma HLS array_partition variable=O complete
	ap_int<MBit*ASize*WSize> o;
	for(unsigned i = 0;i < Depth;i++){
#pragma HLS PIPELINE II = 1
		if(i == 0){
			for(unsigned j = 0; j < ASize; j++){
#pragma HLS UNROLL
				for(unsigned k = 0; k < WSize;k++){
#pragma HLS UNROLL
					ap_int<WBit> w = weight((k*Depth+(j+k)%Depth+1)*WBit-1,(k*Depth+(j+k)%Depth)*WBit);
					ap_int<ABit> a = activation((j*Depth+(j+k)%Depth+1)*ABit-1,(j*Depth+(j+k)%Depth)*ABit);
					ap_int<MBit> tem = w*a;
#pragma HLS resource variable=tem core=Mul_LUT
					O[j][k] = tem;
				}
			}

			for(unsigned a = 0;a < WSize;a++){
#pragma HLS UNROLL
				for(unsigned b = 0;b < Depth;b++){
#pragma HLS UNROLL
					if(b == 0)
						W[a][Depth-1] = weight((a*Depth+a+1)*WBit-1,(a*Depth+a)*WBit);
					else 
						W[a][b-1] = weight((a*Depth+(a+b)%Depth+1)*WBit-1,(a*Depth+(a+b)%Depth)*WBit);
					// cout << W[i][j] << ' ';
				}
				// cout << endl;
			}
			for(unsigned a = 0;a < ASize;a++){
#pragma HLS UNROLL
				for(unsigned b = 0;b < Depth;b++){
#pragma HLS UNROLL
					if(b == 0)
						A[a][Depth-1] = activation((a*Depth+a+1)*ABit-1,(a*Depth+a)*ABit);
					else
						A[a][b-1] = activation((a*Depth+(a+b)%Depth+1)*ABit-1,(a*Depth+(a+b)%Depth)*ABit);
					// cout << A[i][j] << ' ';
				}
				// cout << endl;
			}
		}
		else if(i == Depth -1){
			for(unsigned j = 0; j < ASize; j++){
#pragma HLS UNROLL
				for(unsigned k = 0; k < WSize;k++){
#pragma HLS UNROLL
					ap_int<MBit> tem = W[k][j] * A[j][k];
#pragma HLS resource variable=tem core=Mul_LUT
					o((k*ASize+j+1)*MBit-1,(k*ASize+j)*MBit) = O[j][k]+tem;
				}
			}
		}
		else{
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
#pragma HLS UNROLL
				ap_int<WBit> WTemp = W[k][0];
				for(unsigned d = 0;d < Depth-1;d++){
#pragma HLS UNROLL
					W[k][d] = W[k][d+1];
				}
				W[k][Depth-1] = WTemp;
			}
		}
	}
	return o;
}


template<unsigned Depth,unsigned ASize,unsigned WSize,unsigned ABit,unsigned WBit,unsigned MBit>
ap_int<MBit*ASize*WSize> Normal_Gemm(ap_uint<ABit*ASize*Depth> activation,ap_int<WBit*WSize*Depth> weight){
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
			W[i][j] = weight((i*Depth+j+1)*WBit-1,(i*Depth+j)*WBit);
			// cout << W[i][j] << ' ';
		}
		// cout << endl;
	}

	for(unsigned i = 0;i < ASize;i++){
#pragma HLS UNROLL
		for(unsigned j = 0;j < Depth;j++){
#pragma HLS UNROLL
			A[i][j] = activation((i*Depth+j+1)*ABit-1,(i*Depth+j)*ABit);
			// cout << A[i][j] << ' ';
		}
		// cout << endl;
	}

	for(unsigned i = 0;i < ASize;i++)
#pragma HLS UNROLL
		for(unsigned j = 0;j < WSize;j++)
#pragma HLS UNROLL
			O[i][j] = 0;

	for(unsigned i = 0;i < ASize;i++){
#pragma HLS UNROLL
		for(unsigned j = 0;j < WSize;j++){
#pragma HLS UNROLL
			for(unsigned k = 0;k < Depth;k++){
#pragma HLS UNROLL
				ap_int<MBit> tmp = A[i][k]*W[j][k];
#pragma HLS resource variable=tmp core=Mul_LUT
				O[i][j] += tmp;
			}
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
void ConvStreamGenerator_Batch_Dot(hls::stream<ap_uint<Batch*IOBit*IOP> >& in,hls::stream<ap_uint<Batch*IOBit*IOP> >& out,unsigned reps = 1){
	assert(Channel%IOP == 0);
	const unsigned IOPack = Channel / IOP;
	ap_uint<IOBit*Batch*IOP> Local1[WinSize][Size][IOPack];
	ap_uint<IOBit*Batch*IOP> temp;
	// unsigned nums = 0;
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
			if(line == WinSize-1)
				line = 0;
			else
				line = line+1;
			for(unsigned p = 0;p < Size-WinSize+1;p+=Stride){
				for(unsigned pack = 0;pack < IOPack;pack++){	
					for(unsigned m = 0;m < WinSize;m++){
						unsigned lline = line+m;
						if(lline >= WinSize)
							lline = lline - WinSize;
						for(unsigned n = 0;n < WinSize;n++){
#pragma HLS PIPELINE II=1
							out.write(Local1[lline][p+n][pack]);
							// nums++;
							// cout << "write" << Local1[(line+m)%WinSize][p+n][pack] << endl;
						}
					}
				}
			}
		}
	}
	// cout << "in generator " << nums << endl;
}

template<unsigned Batch,unsigned WinSize,unsigned Size,unsigned Channel,unsigned IOP,unsigned IOBit,unsigned Stride>
void ConvStreamGenerator_Batch_Gemm(hls::stream<ap_uint<Batch*IOBit*IOP> >& in,hls::stream<ap_uint<Batch*IOBit*WinSize*WinSize> >& out,unsigned reps = 1){
	assert(Channel%IOP == 0);
	const unsigned IOPack = Channel / IOP;
	ap_uint<IOBit*Batch> Local1[WinSize][Size][IOPack][IOP];
#pragma HLS array_partition variable=Local1 complete
	ap_uint<Batch*IOBit*IOP> temp;
	unsigned num = 0;
	// for(unsigned i = 0; i < Size*Size*IOPack;i++){
	// 	ap_uint<Batch*IOBit*IOP> temp = in.read();
	// }
	// 	ap_uint<Batch*IOBit*IOP> temp = in.read();
	// unsigned nums = 0;
	for(unsigned rep = 0;rep < reps;rep++){
		unsigned line = 0;
		for(unsigned i = 0;i < WinSize-1;i++){
			for(unsigned j = 0;j < Size;j++){
				for(unsigned pack = 0;pack < IOPack;pack++){
					temp = in.read();
					for(unsigned w = 0;w < IOP;w++){
#pragma HLS UNROLL
						Local1[i][j][pack][w] = temp((w+1)*Batch*IOBit-1,w*Batch*IOBit);
					}
					// cout << "read" << hex << Local1[i][j][pack]<< endl;
				}
			}
			line++;
		}
		for(unsigned i = 0;i < Size-WinSize+1;i++){
			for(unsigned j = 0;j < Size;j++){
				for(unsigned pack = 0;pack < IOPack;pack++){
					temp = in.read();
					for(unsigned w = 0;w < IOP;w++){
#pragma HLS UNROLL
						Local1[line][j][pack][w] = temp((w+1)*Batch*IOBit-1,w*Batch*IOBit);
					}
					// cout << "read" << hex << Local1[i][j][pack]<< endl;
				}
			}
			if(line == WinSize-1)
				line = 0;
			else
				line = line+1;
			for(unsigned p = 0;p < Size-WinSize+1;p+=Stride){
				for(unsigned pack = 0;pack < IOPack;pack++){	
					for(unsigned w = 0;w < IOP;w++){
#pragma HLS PIPELINE II=1
						ap_uint<Batch*IOBit*WinSize*WinSize> Temp;
						for(unsigned m = 0;m < WinSize;m++){
							unsigned lline = line+m;
							if(lline >= WinSize)
								lline = lline - WinSize;
							for(unsigned n = 0;n < WinSize;n++){
								unsigned Wshift = m*WinSize+n;
								Temp((Wshift+1)*Batch*IOBit-1,Wshift*Batch*IOBit) = Local1[lline][p+n][pack][w];
							}
						}
						out.write(Temp);
					}
					// nums++;
					// cout << "write" << Local1[(line+m)%WinSize][p+n][pack] << endl;
				}
			}
		}
	}
	// cout << "in generator " << nums << endl;
}

template<unsigned Batch,unsigned KSize,unsigned WBit,unsigned ABit,unsigned MBit,unsigned InChannel,unsigned OutChannel,unsigned Stride,unsigned Size,unsigned InP,unsigned MidP,unsigned OutP>
void Conv_MulAct_Orbital(hls::stream<ap_uint<Batch*ABit*InP> >& in,hls::stream<ap_uint<Batch*ABit*OutP> >& out,const ap_int<WBit*KSize*KSize> Weight[OutChannel][InChannel],const ap_int<WBit> Bias[OutChannel],const unsigned Scale,unsigned reps = 1){
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
						// if(i == 0)
						// 	res = Orbital_Gemm_Debug<Depth,Batch,MidP,ABit,WBit,MBit>(act[inp],wei[inp]);
						// else
						res = Orbital_Gemm<Depth,Batch,MidP,ABit,WBit,MBit>(act[inp],wei[inp]);
						
						
						// cout << i << " res " << res << endl;
						//write
						for(unsigned j = 0; j < MidP; j++){
// #pragma HLS UNROLL
							for(unsigned k = 0; k < Batch;k++){
// #pragma HLS UNROLL
								MidArray[mpack*MidP*Batch+j*Batch+k] += res((j*Batch+k+1)*MBit-1,(j*Batch+k)*MBit);						
								// if(i == 0 && k == 0){
								// 	cout << res((j*Batch+k+1)*MBit-1,(j*Batch+k)*MBit) << endl;
								// }
								// if(i == 0){
								// 	cout << "mid" << mpack*MidP*Batch+j*Batch+k << " " << res((j*Batch+k+1)*MBit-1,(j*Batch+k)*MBit) << " " << MidArray[mpack*MidP*Batch+j*Batch+k] <<endl;
								// }
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
								// cout << OutArray[NChannel*Batch+batch]  << endl;
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


template<unsigned Batch,unsigned KSize,unsigned WBit,unsigned ABit,unsigned MBit,unsigned InChannel,unsigned OutChannel,unsigned Stride,unsigned Size,unsigned InP,unsigned MidP,unsigned OutP>
void Conv_MulAct_Orbital_New(hls::stream<ap_uint<Batch*ABit*KSize*KSize> >& in,hls::stream<ap_uint<Batch*ABit*OutP> >& out,const ap_int<WBit*KSize*KSize> Weight[OutChannel][InChannel],const ap_int<WBit> Bias[OutChannel],const unsigned Scale,unsigned reps = 1){
	assert(InChannel%InP == 0);
	assert(OutChannel%OutP == 0);
	assert(OutChannel%MidP == 0);
	const unsigned Depth = KSize*KSize;
	// const unsigned InPack = InChannel / InP;
	const unsigned MidPack = OutChannel / MidP;
	const unsigned OutPack = OutChannel / OutP;
	ap_int<MBit> MidArray[Batch*OutChannel];
	ap_uint<ABit> OutArray[Batch*OutChannel];
#pragma HLS array_partition variable=MidArray complete
#pragma HLS array_partition variable=OutArray complete

	const unsigned Nums = ((Size-KSize)/Stride+1)*((Size-KSize)/Stride+1);
//reps
	LoopRep:for(unsigned rep = 0;rep < reps;rep++){
		LoopPix:for(unsigned num = 0;num < Nums;num++){
			for(unsigned ochan = 0;ochan < OutChannel;ochan++){
#pragma HLS UNROLL
				for(unsigned batch = 0;batch < Batch;batch++){
#pragma HLS UNROLL
					MidArray[ochan*Batch+batch] = Bias[ochan];
				}
			}
			LoopInPack:for(unsigned ichan = 0;ichan < InChannel;ichan++){
				ap_uint<ABit*Batch*Depth> act;
				ap_int<WBit*MidP*Depth> wei;
				ap_uint<Batch*ABit*InP*Depth> InTemp = in.read();
				Orbital_pro:for(unsigned depth = 0;depth < Depth;depth++){
#pragma HLS UNROLL
					for(unsigned batch = 0;batch < Batch;batch++){
#pragma HLS UNROLL
						act((batch*Depth+depth+1)*ABit-1,(batch*Depth+depth)*ABit) = InTemp((depth*Batch+batch+1)*ABit-1,(depth*Batch+batch)*ABit);
					}
				}
				Mpack:for(unsigned mpack = 0;mpack < MidPack;mpack++){
					//one orbital
					ap_int<MBit*Batch*MidP> res;
					for(unsigned midp = 0;midp < MidP;midp++){
#pragma HLS UNROLL
						unsigned NOutChannel = mpack * MidP + midp;
						wei((midp+1)*Depth*WBit-1,midp*Depth*WBit) = Weight[NOutChannel][ichan];
					}

	const unsigned WSize = MidP;
	const unsigned ASize = Batch;
	ap_int<WBit> W[WSize][Depth];
	ap_uint<ABit> A[ASize][Depth];
	ap_int<MBit> O[ASize][WSize];
#pragma HLS array_partition variable=W complete
#pragma HLS array_partition variable=A complete
#pragma HLS array_partition variable=O complete
	// ap_int<MBit*ASize*WSize> res;
	for(unsigned i = 0;i < Depth;i++){
#pragma HLS PIPELINE II = 1
		if(i == 0){
			for(unsigned j = 0; j < ASize; j++){
#pragma HLS UNROLL
				for(unsigned k = 0; k < WSize;k++){
#pragma HLS UNROLL
					ap_int<WBit> w = wei((k*Depth+(j+k)%Depth+1)*WBit-1,(k*Depth+(j+k)%Depth)*WBit);
					ap_int<ABit> a = act((j*Depth+(j+k)%Depth+1)*ABit-1,(j*Depth+(j+k)%Depth)*ABit);
					ap_int<MBit> tem = w*a;
#pragma HLS resource variable=tem core=Mul_LUT
					O[j][k] = tem;
				}
			}

			for(unsigned a = 0;a < WSize;a++){
#pragma HLS UNROLL
				for(unsigned b = 0;b < Depth;b++){
#pragma HLS UNROLL
					if(b == 0)
						W[a][Depth-1] = wei((a*Depth+a+1)*WBit-1,(a*Depth+a)*WBit);
					else 
						W[a][b-1] = wei((a*Depth+(a+b)%Depth+1)*WBit-1,(a*Depth+(a+b)%Depth)*WBit);
					// cout << W[i][j] << ' ';
				}
				// cout << endl;
			}
			for(unsigned a = 0;a < ASize;a++){
#pragma HLS UNROLL
				for(unsigned b = 0;b < Depth;b++){
#pragma HLS UNROLL
					if(b == 0)
						A[a][Depth-1] = act((a*Depth+a+1)*ABit-1,(a*Depth+a)*ABit);
					else
						A[a][b-1] = act((a*Depth+(a+b)%Depth+1)*ABit-1,(a*Depth+(a+b)%Depth)*ABit);
					// cout << A[i][j] << ' ';
				}
				// cout << endl;
			}
		}
		else if(i == Depth -1){
			for(unsigned j = 0; j < ASize; j++){
#pragma HLS UNROLL
				for(unsigned k = 0; k < WSize;k++){
#pragma HLS UNROLL
					ap_int<MBit> tem = W[k][j] * A[j][k];
#pragma HLS resource variable=tem core=Mul_LUT
					res((k*ASize+j+1)*MBit-1,(k*ASize+j)*MBit) = O[j][k]+tem;
				}
			}
		}
		else{
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
#pragma HLS UNROLL
				ap_int<WBit> WTemp = W[k][0];
				for(unsigned d = 0;d < Depth-1;d++){
#pragma HLS UNROLL
					W[k][d] = W[k][d+1];
				}
				W[k][Depth-1] = WTemp;
			}
		}
	}
					// res = Orbital_Gemm<Depth,Batch,MidP,ABit,WBit,MBit>(act,wei);

					//write
					for(unsigned j = 0; j < MidP; j++){
// #pragma HLS UNROLL
						for(unsigned k = 0; k < Batch;k++){
// #pragma HLS UNROLL
							MidArray[mpack*MidP*Batch+j*Batch+k] += res((j*Batch+k+1)*MBit-1,(j*Batch+k)*MBit);						
							// if(i == 0 && k == 0){
							// 	cout << res((j*Batch+k+1)*MBit-1,(j*Batch+k)*MBit) << endl;
							// }
							// if(i == 0){
							// 	cout << "mid" << mpack*MidP*Batch+j*Batch+k << " " << res((j*Batch+k+1)*MBit-1,(j*Batch+k)*MBit) << " " << MidArray[mpack*MidP*Batch+j*Batch+k] <<endl;
							// }
						}
					}
				}
				if(ichan == InChannel -1){
					for(unsigned opack = 0;opack < OutPack;opack++){
						ap_uint<Batch*ABit*OutP> OutTemp;
						for(unsigned outp = 0;outp < OutP;outp++){
							for(unsigned batch = 0;batch < Batch;batch++){
								unsigned NChannel = opack*OutP + outp;
								unsigned OToffset = outp*Batch + batch;
								// if(batch == 0)
								// 	cout << i << " " << MidArray[NChannel*Batch+batch] << endl;
								OutArray[NChannel*Batch+batch] = SCALE_RELU_ScaleBit<ABit,MBit>(MidArray[NChannel*Batch+batch],Scale);
								// cout << OutArray[NChannel*Batch+batch]  << endl;
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


template<unsigned Batch,unsigned KSize,unsigned WBit,unsigned ABit,unsigned MBit,unsigned InChannel,unsigned OutChannel,unsigned Stride,unsigned Size,unsigned InP,unsigned MidP,unsigned OutP>
void Conv_MulAct_Normal_Gemm(hls::stream<ap_uint<Batch*ABit*KSize*KSize> >& in,hls::stream<ap_uint<Batch*ABit*OutP> >& out,const ap_int<WBit*KSize*KSize> Weight[OutChannel][InChannel],const ap_int<WBit> Bias[OutChannel],const unsigned Scale,unsigned reps = 1){
	assert(InChannel%InP == 0);
	assert(OutChannel%OutP == 0);
	assert(OutChannel%MidP == 0);
	const unsigned Depth = KSize*KSize;
	// const unsigned InPack = InChannel / InP;
	const unsigned MidPack = OutChannel / MidP;
	const unsigned OutPack = OutChannel / OutP;
	ap_int<MBit> MidArray[Batch*OutChannel];
	ap_uint<ABit> OutArray[Batch*OutChannel];
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
			LoopInPack:for(unsigned ichan = 0;ichan < InChannel;ichan++){
				ap_uint<ABit*Batch*Depth> act;
				ap_int<WBit*MidP*Depth> wei;
				ap_uint<Batch*ABit*InP*Depth> InTemp = in.read();
				Orbital_pro:for(unsigned depth = 0;depth < Depth;depth++){
#pragma HLS UNROLL
					for(unsigned batch = 0;batch < Batch;batch++){
#pragma HLS UNROLL
						act((batch*Depth+depth+1)*ABit-1,(batch*Depth+depth)*ABit) = InTemp((depth*Batch+batch+1)*ABit-1,(depth*Batch+batch)*ABit);
					}
				}
				for(unsigned mpack = 0;mpack < MidPack;mpack++){
#pragma HLS PIPELINE II = 1
					//one orbital
					ap_int<MBit*Batch*MidP> res;
					for(unsigned midp = 0;midp < MidP;midp++){
#pragma HLS UNROLL
						unsigned NOutChannel = mpack * MidP + midp;
						wei((midp+1)*Depth*WBit-1,midp*Depth*WBit) = Weight[NOutChannel][ichan];
					}

					//calculate
					// cout << i << " act " << act[inp] << endl;
					// cout << i << " wei " << wei[inp] << endl;
					// if(i == 0)
					// 	res = Orbital_Gemm_Debug<Depth,Batch,MidP,ABit,WBit,MBit>(act[inp],wei[inp]);
					// else
					res = Normal_Gemm<Depth,Batch,MidP,ABit,WBit,MBit>(act,wei);
					
					
					// cout << i << " res " << res << endl;
					//write
					for(unsigned j = 0; j < MidP; j++){
// #pragma HLS UNROLL
						for(unsigned k = 0; k < Batch;k++){
// #pragma HLS UNROLL
							MidArray[mpack*MidP*Batch+j*Batch+k] += res((j*Batch+k+1)*MBit-1,(j*Batch+k)*MBit);						
							// if(i == 0 && k == 0){
							// 	cout << res((j*Batch+k+1)*MBit-1,(j*Batch+k)*MBit) << endl;
							// }
							// if(i == 0){
							// 	cout << "mid" << mpack*MidP*Batch+j*Batch+k << " " << res((j*Batch+k+1)*MBit-1,(j*Batch+k)*MBit) << " " << MidArray[mpack*MidP*Batch+j*Batch+k] <<endl;
							// }
						}
					}
				}
				if(ichan == InChannel -1){
					for(unsigned opack = 0;opack < OutPack;opack++){
						ap_uint<Batch*ABit*OutP> OutTemp;
						for(unsigned outp = 0;outp < OutP;outp++){
							for(unsigned batch = 0;batch < Batch;batch++){
								unsigned NChannel = opack*OutP + outp;
								unsigned OToffset = outp*Batch + batch;
								// if(batch == 0)
								// 	cout << i << " " << MidArray[NChannel*Batch+batch] << endl;
								OutArray[NChannel*Batch+batch] = SCALE_RELU_ScaleBit<ABit,MBit>(MidArray[NChannel*Batch+batch],Scale);
								// cout << OutArray[NChannel*Batch+batch]  << endl;
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


template<unsigned Batch,unsigned KSize,unsigned WBit,unsigned ABit,unsigned MBit,unsigned InChannel,unsigned OutChannel,unsigned Stride,unsigned Size,unsigned InP,unsigned MidP,unsigned OutP>
void Conv_MulAct_Normal(hls::stream<ap_uint<Batch*ABit*InP> >& in,hls::stream<ap_uint<Batch*ABit*OutP> >& out,const ap_int<WBit*MidP*InP> Weight[KSize*KSize][OutChannel/MidP][InChannel/InP],const ap_int<WBit> Bias[OutChannel],const unsigned Scale,unsigned reps = 1){
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
				ap_uint<ABit> act[InP][Batch][Depth];
				ap_int<WBit> wei[InP][MidP];
#pragma HLS array_partition variable=act complete
#pragma HLS array_partition variable=wei complete
				for(unsigned mpack = 0;mpack < MidPack;mpack++){
					for(unsigned depth = 0;depth < Depth;depth++){
#pragma HLS PIPELINE II = 1
						for(unsigned inp = 0;inp < InP;inp++){
#pragma HLS UNROLL
							for(unsigned midp = 0;midp < MidP;midp++){
#pragma HLS UNROLL
								wei[inp][midp] = Weight[depth][mpack][ipack]((inp*MidP+midp+1)*WBit-1,(inp*MidP+midp)*WBit);
							}
						}
						if(mpack == 0){
							ap_uint<Batch*ABit*InP> InTemp = in.read();
							for(unsigned inp = 0;inp < InP;inp++){
#pragma HLS UNROLL
								for(unsigned batch = 0;batch < Batch;batch++){
#pragma HLS UNROLL					
									act[inp][batch][depth] = InTemp((inp*Batch+batch+1)*ABit-1,(inp*Batch+batch)*ABit);
								}
							}
						}
						for(unsigned inp = 0;inp < InP;inp++){
#pragma HLS UNROLL
							for(unsigned batch = 0;batch < Batch;batch++){
#pragma HLS UNROLL
								for(unsigned midp = 0;midp < MidP;midp++){
#pragma HLS UNROLL
									unsigned OutChan = mpack*MidP + midp;
									ap_int<WBit> W_now = wei[inp][midp];
									// if(i == 0 && batch == 0)
										// cout << OutChan << " " << depth << " " << inp << " " << batch << " " << act[inp][batch][depth] << " " << W_now << endl;
									ap_int<MBit> ResTemp = act[inp][batch][depth]*W_now;
#pragma HLS resource variable=ResTemp core=Mul_LUT
									MidArray[OutChan*Batch+batch] += ResTemp;
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
						// cout << "write out" << endl;
						out.write(OutTemp);
						// cout << OutTemp << endl << endl;
					}
				}
			}
		}
	}
}

template<unsigned Batch,unsigned KSize,unsigned Size,unsigned InChannel,unsigned OutChannel,unsigned InP,unsigned MidP_i,unsigned MidP_o,unsigned OutP,unsigned Stride,unsigned WBit,unsigned ABit,unsigned MBit>
void ConvLayer_NOPAD_Normal(hls::stream<ap_uint<Batch*InP*ABit> >& in,hls::stream<ap_uint<Batch*OutP*ABit> >& out,const ap_int<WBit*MidP_o*MidP_i> Weight[KSize*KSize][OutChannel/MidP_o][InChannel/MidP_i],const ap_int<WBit> Bias[OutChannel],const unsigned Scale,unsigned reps = 1){
#pragma HLS DATAFLOW
	assert(InChannel%InP == 0);
	assert(OutChannel%OutP == 0);
	assert(OutChannel%MidP_o == 0);
	assert(InP%MidP_i == 0);
	const unsigned StrLen = (InChannel/InP)*Size*Size;
	hls::stream<ap_uint<Batch*MidP_i*ABit> > Conv_Str;
	hls::stream<ap_uint<Batch*MidP_i*ABit> > in_m;
	hls::stream<ap_uint<Batch*OutP*ABit> > out1;
	hls::stream<ap_uint<Batch*OutP*ABit> > out2;
	splitStream_Length<Batch*InP*ABit,Batch*MidP_i*ABit,StrLen>(in,in_m,reps);
	ConvStreamGenerator_Batch_Dot<Batch,KSize,Size,InChannel,MidP_i,ABit,Stride>(in_m,Conv_Str,reps);
	Conv_MulAct_Normal<Batch,KSize,WBit,ABit,MBit,InChannel,OutChannel,Stride,Size,MidP_i,MidP_o,OutP>(Conv_Str,out,Weight,Bias,Scale,reps);
}

template<unsigned Batch,unsigned KSize,unsigned Size,unsigned InChannel,unsigned OutChannel,unsigned InP,unsigned MidP_i,unsigned MidP_o,unsigned OutP,unsigned Stride,unsigned WBit,unsigned ABit,unsigned MBit>
void ConvLayer_NOPAD_Normal_Gemm(hls::stream<ap_uint<Batch*InP*ABit> >& in,hls::stream<ap_uint<Batch*OutP*ABit> >& out,const ap_int<WBit*KSize*KSize> Weight[OutChannel][InChannel],const ap_int<WBit> Bias[OutChannel],const unsigned Scale,unsigned reps = 1){
#pragma HLS DATAFLOW
	assert(InChannel%InP == 0);
	assert(OutChannel%OutP == 0);
	assert(OutChannel%MidP_o == 0);
	assert(InP%MidP_i == 0);
	const unsigned StrLen = (InChannel/InP)*Size*Size;
	hls::stream<ap_uint<Batch*KSize*KSize*ABit> > Conv_Str;
	hls::stream<ap_uint<Batch*MidP_i*ABit> > in_m;
	hls::stream<ap_uint<Batch*OutP*ABit> > out1;
	hls::stream<ap_uint<Batch*OutP*ABit> > out2;
	splitStream_Length<Batch*InP*ABit,Batch*MidP_i*ABit,StrLen>(in,in_m,reps);
	ConvStreamGenerator_Batch_Gemm<Batch,KSize,Size,InChannel,MidP_i,ABit,Stride>(in_m,Conv_Str,reps);
	Conv_MulAct_Normal_Gemm<Batch,KSize,WBit,ABit,MBit,InChannel,OutChannel,Stride,Size,MidP_i,MidP_o,OutP>(Conv_Str,out,Weight,Bias,Scale,reps);
}
