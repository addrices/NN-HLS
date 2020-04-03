//#define AP_INT_MAX_W 2048
#include "config.h"
#include <hls_stream.h>
#include <assert.h>
//#include "../../nn-h/FcnnLayer.h"
#include "../../nn-h/util.h"
//#include "../../nn-h/conv.h"
#include "../../nn-h/pooling.h"
#include "../../nn-h/conv_systolic.h"
#include "../../nn-h/FcnnLayer_systolic.h"
using namespace std;


void ConvStreamGenerator_Batch_Test(){
	const unsigned Batch = 5;
	const int ABit = 4;
	const int InP = 4;

	ap_uint<ABit> IMG1[Batch][16][28][28];
	for(int a = 0;a < Batch;a++){
		for(int b = 0;b < 16;b++){
			for(int c = 0;c < 28;c++){
				for(int d =0;d < 28;d++){
					IMG1[a][b][c][d] = (a+b+c+d)%10;
				}
				cout << endl;
			}
		}
	}


	hls::stream<ap_uint<Batch*ABit*InP> > in;
	hls::stream<ap_uint<Batch*ABit*InP> > out;
	ap_uint<Batch*ABit*InP> Temp;
	for(unsigned ipack = 0;ipack < 4;ipack++){
		for(unsigned a = 0;a < 28;a++){
			for(unsigned b = 0;b < 28;b++){
				for(unsigned ip = 0;ip < 4;ip++){
					for(unsigned bat = 0;bat < Batch;bat++){
						int offset = ip*Batch +bat;
						Temp((offset+1)*ABit-1,offset*ABit) = IMG1[bat][ipack*4+ip][a][b];
					}
				}
				in.write(Temp);
			}
		}
	}

	ConvStreamGenerator_Batch<Batch,3,28,16,4,ABit,1>(in,out,1);
	for(unsigned ipack = 0;ipack < 4;ipack++){
		for(int a = 0;a < 26;a++){
			for(int b = 0;b < 26;b++){
				for(int offa = 0;offa < 3;offa++){
					for(int offb = 0;offb < 3;offb++){
						ap_uint<Batch*ABit*4> Temp = out.read();
						for(unsigned ip = 0;ip < 4;ip++){
							for(int bat = 0;bat < Batch;bat++){
								unsigned offset = ip*Batch+bat;
								ap_uint<ABit> B = Temp((offset+1)*ABit-1,offset*ABit);
								if(B != IMG1[bat][ipack*4+ip][a+offa][b+offb]){
									cout << "bat " << bat << " a " << a << " b " << b << " offa " << offa << " offb " << offb << " str " << B << " img " << IMG1[bat][0][a+offa][b+offb] << endl;
									assert(0);
								}
							}
						}
					}
				}
			}
		}
	}
	cout << "good" << endl;
}

void ConvLayer_NoPad_Orbital_Test(){
	const unsigned Batch = 1;
	const unsigned InChannel = 1;
	const unsigned OutChannel = 4;
	const unsigned ABit = 4;
	const unsigned WBit = 8;
	const unsigned MBit = 12;
	const unsigned KSize =3;
	const unsigned Size = 28;
	const unsigned InP = 1;
	const unsigned OutP = 4;
	const unsigned MidP = 2;
	const unsigned Stride = 1;
	const unsigned Scale = 1;
	ap_uint<ABit> IMG1[Batch][InChannel][28][28] =
			{{{{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}}}};


	hls::stream<ap_uint<Batch*ABit*InP> > in_o;
	hls::stream<ap_uint<Batch*OutP*ABit> > out_o;
	hls::stream<ap_uint<ABit*InChannel> > in;
	hls::stream<ap_uint<ABit*OutChannel> > out;
	for(int a = 0;a < Size;a++){
		for(int b = 0;b < Size;b++){
			in.write(a+b);
			in_o.write(a+b);
		}
	}

	ap_int<WBit> OW[OutChannel][InChannel][KSize][KSize] = {{{{1,2,3},{4,5,-4},{3,2,-1}}},{{{1,2,2},{3,5,-4},{3,1,-1}}},{{{1,2,3},{4,0,-4},{3,2,-1}}},{{{-1,2,3},{4,5,-4},{4,3,1}}}};
    ap_int<WBit> Bias[OutChannel] = {1,1,1,1};
	ap_int<WBit*InP> Weight_N[(InChannel/InP)*KSize*KSize*(OutChannel/OutP)][OutP];
	ap_int<WBit*KSize*KSize> Weight_O[OutChannel][InChannel];

	unsigned InPack = InChannel/InP;
	unsigned OutPack = OutChannel/OutP;
	trans_normal<InChannel,OutChannel,KSize,InP,OutP,WBit>(OW,Weight_N);
	trans_orbital<InChannel,OutChannel,KSize,WBit>(OW,Weight_O);
	ConvLayer_NOPAD_ScaleBit<KSize,WBit,ABit,MBit,InChannel,OutChannel,Stride,Size,InP,OutP>(in,out,Weight_N,Bias,Scale,1);
	ConvLayer_NOPAD_Orbital<Batch,KSize,Size,InChannel,OutChannel,InP,InP,MidP,OutP,Stride,WBit,ABit,MBit>(in_o,out_o,Weight_O,Bias,Scale,1);

	for(int i = 0;i < 26;i++){
		for(int j = 0;j < 26;j++){
			ap_uint<Batch*OutP*ABit> O1 = out_o.read();
			ap_uint<ABit*OutChannel> O2 = out.read();
			if(O1 != O2){
				cout << "!!!!!!!!!!!!!!!!!" << endl;
				cout << " i " << i << " j " << j << " O1 " << O1 << " O2 " << O2 << endl;
				assert(0);
			}
		}
	}
	//ConvLayer_NOPAD_Orbital<Batch,KSize,WBit,ABit,MBit,InChannel,OutChannel,Stride,Size,InP,MidP,OutP>(in_o,out_o,1);
	//ConvLayer_NOPAD_ScaleBit<KSize,WBit,ABit,MBit,InChannel,OutChannel,Stride,Size,InP,OutP>(in,out,1);
}

//void Orbital_Test(){
//	const unsigned Batch = 2;
//	ap_uint<ABIT> IMG1[Batch][C1_INCHANNEL][C1_SIZE][C1_SIZE] =
//			{{{{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}}},
//			{{{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0},
//			{0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0},
//			{0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,1,0,0,0,0,0},
//			{0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,0,0,0,0,0},
//			{0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0},
//			{0,0,0,0,0,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0},
//			{0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0},
//			{0,0,0,0,0,1,1,0,0,1,1,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}}}};
//
//	hls::stream<ap_uint<Batch*1*ABIT> > O1_in;
//	hls::stream<ap_uint<Batch*8*ABIT> > O1_out;		//1*1*4
//	hls::stream<ap_uint<Batch*1*ABIT> > O2_out;		//1*1*4
//
//	hls::stream<ap_uint<ABIT*C1_INCHANNEL> > C1_in1;
//	hls::stream<ap_uint<ABIT*C1_OUTCHANNEL> > C1_out1;
//	hls::stream<ap_uint<ABIT*C1_INCHANNEL> > C1_in2;
//	hls::stream<ap_uint<ABIT*C1_OUTCHANNEL> > C1_out2;
//
//	hls::stream<ap_uint<ABIT*C2_OUTCHANNEL> > C2_out1;
//	hls::stream<ap_uint<ABIT*C2_OUTCHANNEL> > C2_out2;
//
//	for(int i = 0;i < 28;i++){
//		for(int j = 0;j < 28;j++){
//			C1_in1.write(IMG1[0][0][i][j]);
//			C1_in2.write(IMG1[1][0][i][j]);
//			ap_uint<Batch*C1_INP*ABIT> OTemp;
//			OTemp((2*C1_INP*ABIT)-1,C1_INP*ABIT) = IMG1[1][0][i][j];
//			OTemp(C1_INP*ABIT-1,0) = IMG1[0][0][i][j];
//			O1_in.write(OTemp);
//		}
//	}
//	ConvLayer_NOPAD_ScaleBit<C1_KSIZE,WBIT,ABIT,C1_MBIT,C1_INCHANNEL,C1_OUTCHANNEL,C1_STRIDE,C1_SIZE,1,2>(C1_in1,C1_out1,C1_W_,C1_B,C1_SCALEBIT,1);
//	ConvLayer_NOPAD_ScaleBit<C2_KSIZE,WBIT,ABIT,C2_MBIT,C2_INCHANNEL,C2_OUTCHANNEL,C2_STRIDE,C2_SIZE,8,8>(C1_out1,C2_out1,C2_W_,C2_B,C2_SCALEBIT,1);
//
//	ConvLayer_NOPAD_ScaleBit<C1_KSIZE,WBIT,ABIT,C1_MBIT,C1_INCHANNEL,C1_OUTCHANNEL,C1_STRIDE,C1_SIZE,1,2>(C1_in2,C1_out2,C1_W_,C1_B,C1_SCALEBIT,1);
//	ConvLayer_NOPAD_ScaleBit<C2_KSIZE,WBIT,ABIT,C2_MBIT,C2_INCHANNEL,C2_OUTCHANNEL,C2_STRIDE,C2_SIZE,8,8>(C1_out2,C2_out2,C2_W_,C2_B,C2_SCALEBIT,1);
//
//	ConvLayer_NOPAD_Orbital<Batch,C1_KSIZE,C1_SIZE,C1_INCHANNEL,C1_OUTCHANNEL,1,1,C1_MIDP,8,C1_STRIDE,WBIT,ABIT,C1_MBIT>(O1_in,O1_out,C1_W,C1_B,C1_SCALEBIT,1);
//	ConvLayer_NOPAD_Orbital<Batch,C2_KSIZE,C2_SIZE,C2_INCHANNEL,C2_OUTCHANNEL,8,4,C2_MIDP,1,C2_STRIDE,WBIT,ABIT,C2_MBIT>(O1_out,O2_out,C2_W,C2_B,C2_SCALEBIT,1);
//
//	for(int i = 0;i < 24;i++){
//		for(int j = 0;j < 24;j++){
//			ap_uint<128> N1 = C2_out1.read();
//			ap_uint<128> N2 = C2_out2.read();
//			for(int m = 0;m < 32;m++){
//				ap_uint<8> O = O2_out.read();
//				ap_uint<4> O1 = O(3,0);
//				ap_uint<4> O2 = O(7,4);
//				ap_uint<4> n1 = N1((m+1)*4-1,m*4);
//				ap_uint<4> n2 = N2((m+1)*4-1,m*4);
//				if(O1!=n1 || O2!=n2){
//					cout << "i:" << i << " j:" << j << " m:" << m << endl;
//					assert(0);
//				}
//			}
//		}
//	}
//	cout << " good " << endl;
//}

void test_temp(){
	const unsigned ABit = 8;
	ap_uint<ABit> IMG[1][2][3][3] =
		{{{{1,0,3},{2,1,3},{3,2,3}},
		  {{1,0,4},{3,5,1},{5,5,8}}}};
	ap_int<8> W1[2][2][3][3] =
		{{{{1,2,0},{2,0,3},{3,0,3}},
	      {{1,0,0},{0,3,1},{1,3,4}}},
		  {{{1,2,3},{2,1,3},{3,2,3}},
	      {{1,0,4},{3,3,1},{2,3,4}}}};
	ap_int<8> B1[2] = {1,-1};

	ap_int<8> Wn[2*9*2][1];
	ap_int<8*9> Wo[2][2];
	trans_normal<2,2,3,1,1,8>(W1,Wn);
	trans_orbital<2,2,3,8>(W1,Wo);

	hls::stream<ap_uint<ABit*2> > C1_in1;
	hls::stream<ap_uint<ABit*2> > C1_out1;
	hls::stream<ap_uint<ABit*1> > O1_in1;
	hls::stream<ap_uint<ABit*1> > O1_out1;
	for(int i = 0;i < 3;i++){
		for(int j = 0;j < 3;j++){
			ap_uint<ABit*2> C1;
			C1(2*ABit-1,ABit) = IMG[0][1][i][j];
			C1(ABit-1,0) = IMG[0][0][i][j];
			C1_in1.write(C1);
			O1_in1.write(IMG[0][0][i][j]);
			O1_in1.write(IMG[0][1][i][j]);
		}
	}

	ConvLayer_NOPAD_ScaleBit<3,8,ABit,12,2,2,1,3,1,1>(C1_in1,C1_out1,Wn,B1,1,1);
	ConvLayer_NOPAD_Orbital<1,3,3,2,2,1,1,2,1,1,8,ABit,12>(O1_in1,O1_out1,Wo,B1,1,1);

	ap_uint<ABit*2> Cout = C1_out1.read();
	ap_uint<ABit*1> Oout1 = O1_out1.read();
	ap_uint<ABit*1> Oout2 = O1_out1.read();
	cout << hex << Cout << endl;
	cout << hex << Oout1 << endl;
	cout << hex << Oout2 << endl;

}

void test_gemm(){
	ap_uint<4> act[3][9] = {{1,1,1,1,1,1,1,1,1},{1,1,1,1,1,1,1,1,1},{1,1,1,1,1,1,1,1,1}};
	ap_int<8> wei[3][9] = {{1,1,1,1,1,1,1,1,1},{1,1,1,1,1,1,1,1,1},{1,1,1,1,1,1,1,1,1}};
	ap_int<12> res[3][3];

	ap_uint<4*3*9> A;
	for(int i = 0;i < 3;i++){
		for(int j = 0;j < 9;j++){
			A((i*9+j+1)*4-1,(i*9+j)*4) = act[i][j];
		}
	}
	ap_uint<8*3*9> W;
	for(int i = 0;i < 3;i++){
		for(int j = 0;j < 9;j++){
			W((i*9+j+1)*8-1,(i*9+j)*8) = wei[i][j];
		}
	}
	ap_int<12*3*3> R;
	R = Orbital_Gemm<9,3,3,4,8,12>(A,W);
	cout << R << endl;
	for(int i = 0;i < 3;i++){
		for(int j = 0;j < 3;j++){
			res[i][j] = R((i*3+j+1)*12-1,(i*3+j)*12);
			cout << res[i][j] << ' ';
		}
		cout << endl;
	}
}

void TEST(){
	const unsigned Batch = 2;
	ap_uint<ABIT> IMG1[Batch][C1_INCHANNEL][C1_SIZE][C1_SIZE] =
			{{{{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}}},
			{{{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0},
			{0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0},
			{0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,1,0,0,0,0,0},
			{0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,0,0,0,0,0},
			{0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0},
			{0,0,0,0,0,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0},
			{0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0},
			{0,0,0,0,0,1,1,0,0,1,1,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}}}};

	hls::stream<ap_uint<Batch*C1_INP*ABIT> > C1_in;
	hls::stream<ap_uint<Batch*C1_OUTP*ABIT> > C1_out;
	hls::stream<ap_uint<Batch*C2_OUTP*ABIT> > C2_out;
	hls::stream<ap_uint<Batch*P2_OUTP*ABIT> > P2_out;
	hls::stream<ap_uint<Batch*C3_OUTP*ABIT> > C3_out;
	hls::stream<ap_uint<Batch*P3_OUTP*ABIT> > P3_out;
	hls::stream<ap_uint<Batch*C4_OUTP*ABIT> > C4_out;
	hls::stream<ap_uint<Batch*P4_OUTP*ABIT> > P4_out;

	hls::stream<ap_uint<Batch*F5_OUTPIX*ABIT> > F5_out;
	hls::stream<ap_uint<Batch*F6_OUTPIX*ABIT> > F6_out;

	for(int i = 0;i < C1_SIZE;i++){
		for(int j = 0;j < C1_SIZE;j++){
			ap_uint<Batch*ABIT> temp;
			for(int b = 0;b < Batch;b++){
				temp((b+1)*ABIT-1,b*ABIT) = IMG1[b][0][i][j];
			}
			C1_in.write(temp);
		}
	}
	ConvLayer_NOPAD_Orbital<Batch,C1_KSIZE,C1_SIZE,C1_INCHANNEL,C1_OUTCHANNEL,C1_INP,C1_MIDP_I,C1_MIDP_O,C1_OUTP,C1_STRIDE,WBIT,ABIT,C1_MBIT>(C1_in,C1_out,C1_W,C1_B,C1_SCALEBIT,1);
	ConvLayer_NOPAD_Orbital<Batch,C2_KSIZE,C2_SIZE,C2_INCHANNEL,C2_OUTCHANNEL,C2_INP,C2_MIDP_I,C2_MIDP_O,C2_OUTP,C2_STRIDE,WBIT,ABIT,C2_MBIT>(C1_out,C2_out,C2_W,C2_B,C2_SCALEBIT,1);

//	unsigned OutPack = C2_OUTCHANNEL/C2_OUTP;
//	for(int i = 0;i < 24;i++){
//		for(int j = 0;j < 24;j++){
//			for(int m = 0;m < OutPack;m++){
//				ap_uint<Batch*C2_OUTP*ABIT> otemp = C2_out.read();
//				for(int n = 0;n < C2_OUTP;n++){
//					for(int q = 0;q < Batch;q++){
//						unsigned offset = n*Batch+q;
//						ap_uint<ABIT> cc = otemp((offset+1)*ABIT-1,offset*ABIT);
//						cout << cc << " " ;
//					}
//				}
//			}
//			cout << endl;
//		}
//	}


	MaxPool_IOP<P2_PSIZE,ABIT,P2_SIZE,P2_CHANNEL*Batch,P2_INP*Batch,P2_OUTP*Batch>(C2_out,P2_out);

//	unsigned OutPack = P2_CHANNEL/P2_OUTP;
//	for(int i = 0;i < 12;i++){
//		for(int j = 0;j < 12;j++){
//			for(int m = 0;m < OutPack;m++){
//				ap_uint<Batch*P2_OUTP*ABIT> otemp = P2_out.read();
//				for(int n = 0;n < P2_OUTP;n++){
//					for(int q = 0;q < Batch;q++){
//						unsigned offset = n*Batch+q;
//						ap_uint<ABIT> cc = otemp((offset+1)*ABIT-1,offset*ABIT);
//						cout << cc << " " ;
//					}
//				}
//			}
//			cout << endl;
//		}
//	}

	ConvLayer_NOPAD_Orbital<Batch,C3_KSIZE,C3_SIZE,C3_INCHANNEL,C3_OUTCHANNEL,C3_INP,C3_MIDP_I,C3_MIDP_O,C3_OUTP,C3_STRIDE,WBIT,ABIT,C3_MBIT>(P2_out,C3_out,C3_W,C3_B,C3_SCALEBIT,1);
	MaxPool_IOP<P3_PSIZE,ABIT,P3_SIZE,P3_CHANNEL*Batch,P3_INP*Batch,P3_OUTP*Batch>(C3_out,P3_out);
//	unsigned OutPack = P3_CHANNEL/P3_OUTP;
//	for(int i = 0;i < 5;i++){
//		for(int j = 0;j < 5;j++){
//			for(int m = 0;m < OutPack;m++){
//				ap_uint<Batch*P3_OUTP*ABIT> otemp = P3_out.read();
//				for(int n = 0;n < P3_OUTP;n++){
//					for(int q = 0;q < Batch;q++){
//						unsigned offset = n*Batch+q;
//						ap_uint<ABIT> cc = otemp((offset+1)*ABIT-1,offset*ABIT);
//						cout << cc << " " ;
//					}
//				}
//			}
//			cout << endl;
//		}
//	}
//
	ConvLayer_NOPAD_Orbital<Batch,C4_KSIZE,C4_SIZE,C4_INCHANNEL,C4_OUTCHANNEL,C4_INP,C4_MIDP_I,C4_MIDP_O,C4_OUTP,C4_STRIDE,WBIT,ABIT,C4_MBIT>(P3_out,C4_out,C4_W,C4_B,C4_SCALEBIT,1);
	MaxPool_IOP<P4_PSIZE,ABIT,P4_SIZE,P4_CHANNEL*Batch,P4_INP*Batch,P4_OUTP*Batch>(C4_out,P4_out);
//	unsigned OutPack = P4_CHANNEL/P4_OUTP;
//	for(int i = 0;i < 1;i++){
//		for(int j = 0;j < 1;j++){
//			for(int m = 0;m < OutPack;m++){
//				ap_uint<Batch*P4_OUTP*ABIT> otemp = P4_out.read();
//				for(int n = 0;n < P4_OUTP;n++){
//					for(int q = 0;q < Batch;q++){
//						unsigned offset = n*Batch+q;
//						ap_uint<ABIT> cc = otemp((offset+1)*ABIT-1,offset*ABIT);
//						cout << cc << " " ;
//					}
//				}
//			}
//			cout << endl;
//		}
//	}

	FcnnLayer_Batch<Batch,F5_INPIX,F5_OUTPIX,F5_INSIZE,F5_OUTSIZE,WBIT,ABIT,F5_MBIT,F5_DEPTH,F5_WSIZE>(P4_out,F5_out,F5_W,F5_B,F5_SCALEBIT,1);
//	for(int i = 0;i < 32;i++){
//		ap_uint<Batch*F5_OUTPIX*ABIT> OutTemp = F5_out.read();
//		for(int j = 0;j < Batch;j++){
//			cout << OutTemp((j+1)*ABIT-1,j*ABIT) << " ";
//		}
//		cout << endl;
//	}
	FcnnLayer_Batch<Batch,F6_INPIX,F6_OUTPIX,F6_INSIZE,F6_OUTSIZE,WBIT,ABIT,F6_MBIT,F6_DEPTH,F6_WSIZE>(F5_out,F6_out,F6_W,F6_B,F6_SCALEBIT,1);
	for(int i = 0;i < 10;i++){
		ap_uint<Batch*F6_OUTPIX*ABIT> OutTemp = F6_out.read();
		for(int j = 0;j < Batch;j++){
			cout << OutTemp((j+1)*ABIT-1,j*ABIT) << " ";
		}
		cout << endl;
	}
}

struct ap_axis{
	ap_uint<128> data;
	ap_int<1> last;
	ap_int<16> keep;
};
void top(hls::stream<ap_axis >& in,hls::stream<ap_axis >& out,unsigned reps = 1);
void topTest(){
	unsigned Batch = 8;
	ap_uint<8> IMG[2][C1_SIZE][C1_SIZE] =
	{{{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}},
	{{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0},
	{0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0},
	{0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,1,0,0,0,0,0},
	{0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,0,0,0,0,0},
	{0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0},
	{0,0,0,0,0,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0},
	{0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0},
	{0,0,0,0,0,1,1,0,0,1,1,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}}};

	hls::stream<ap_axis> in;
	hls::stream<ap_axis> out;

	ap_axis temp;

	for(int j = 0; j < 56;j++){
		for(int bat = 0;bat < Batch;bat++){
			for(int k = 0; k < 14;k++){
				int a = j/2;
				int b = (j%2)*14+k;
				temp.data((k+1)*8-1,k*8) = IMG[bat%2][a][b];
				temp.keep(k,k) = 1;
			}
			in.write(temp);
		}
	}



	top(in,out,1);

	for(int i = 0;i < 10;i++){
		ap_axis OTemp = out.read();
		for(int j = 0; j < 16;j++){
			ap_uint<ABIT> q1 = OTemp.data((j+1)*8-5,j*8);
			cout << q1 << " " ;
		}
		cout << endl;
	}
}

int main(){
	// ConvStreamGenerator_Batch_Test();
	// ConvLayer_NoPad_Orbital_Test();
	topTest();
}



//void Total_Transform(){
//	const unsigned WBit = 8;
//	const unsigned KSize = 3;
//	const unsigned OutChannel1 = 16;
//	const unsigned InChannel1 = 1;
//	const unsigned OutChannel2 = 32;
//	const unsigned InChannel2 = 16;
//	const unsigned OutChannel3 = 64;
//	const unsigned InChannel3 = 32;
//	const unsigned OutChannel4 = 64;
//	const unsigned InChannel4 = 64;
//	ap_int<WBit*KSize*KSize> OutWeight1[OutChannel1][InChannel1];
//	trans_orbital<InChannel1,OutChannel1,KSize,WBit>(L1_Wei,OutWeight1);
//	cout << "L1" << endl;
//	cout_weight<WBit*KSize*KSize,OutChannel1,InChannel1>(OutWeight1);
//
//	ap_int<WBit*KSize*KSize> OutWeight2[OutChannel2][InChannel2];
//	trans_orbital<InChannel2,OutChannel2,KSize,WBit>(L2_Wei,OutWeight2);
//	cout << "L2" << endl;
//	cout_weight<WBit*KSize*KSize,OutChannel2,InChannel2>(OutWeight2);
//
//	ap_int<WBit*KSize*KSize> OutWeight3[OutChannel3][InChannel3];
//	trans_orbital<InChannel3,OutChannel3,KSize,WBit>(L3_Wei,OutWeight3);
//	cout << "L3" << endl;
//	cout_weight<WBit*KSize*KSize,OutChannel3,InChannel3>(OutWeight3);
//
//	ap_int<WBit*KSize*KSize> OutWeight4[OutChannel4][InChannel4];
//	trans_orbital<InChannel4,OutChannel4,KSize,WBit>(L4_Wei,OutWeight4);
//	cout << "L4" << endl;
//	cout_weight<WBit*KSize*KSize,OutChannel4,InChannel4>(OutWeight4);
//}
