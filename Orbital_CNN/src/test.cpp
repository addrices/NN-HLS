#define AP_INT_MAX_W 8192
#include "config.h"
#include <hls_stream.h>
#include <assert.h>
//#include "../../nn-h/FcnnLayer.h"
#include "../../nn-h/util.h"
//#include "../../nn-h/conv.h"
#include "../../nn-h/pooling.h"
#include "../../nn-h/conv_systolic.h"
#include "../../nn-h/FcnnLayer_systolic.h"
#include "../../nn-h/io.h"

using namespace std;


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

//void top(hls::stream<ap_axis >& in,hls::stream<ap_axis >& out,unsigned reps = 1);
void top(hls::stream<ap_axis_512 >& in,hls::stream<ap_axis_512 >& out,unsigned reps = 1);
//void topTest(){
//	unsigned Batch = 4;
//	ap_uint<8> IMG[2][C1_SIZE][C1_SIZE] =
//	{{{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}},
//	{{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0},
//	{0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0},
//	{0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,1,0,0,0,0,0},
//	{0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,0,0,0,0,0},
//	{0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0},
//	{0,0,0,0,0,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0},
//	{0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0},
//	{0,0,0,0,0,1,1,0,0,1,1,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
//	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}}};
//
//	hls::stream<ap_axis> in;
//	hls::stream<ap_axis> out;
//
//	ap_axis temp;
//
//	for(int j = 0; j < 56;j++){
//		for(int bat = 0;bat < Batch;bat++){
//			for(int k = 0; k < 14;k++){
//				int a = j/2;
//				int b = (j%2)*14+k;
//				temp.data((k+1)*8-1,k*8) = IMG[bat%2][a][b];
////				temp.data((k+1)*8-1,k*8) = k%3;
//				temp.keep(k,k) = 1;
//			}
//			in.write(temp);
//		}
//	}
//
//
//
//	top(in,out,1);
//
//	for(int i = 0;i < 10;i++){
//		ap_axis OTemp = out.read();
//		for(int j = 0; j < 16;j++){
//			ap_uint<ABIT> q1 = OTemp.data((j+1)*8-5,j*8);
//			cout << q1 << " " ;
//		}
//		cout << endl;
//	}
//}

void gemmtest(){
	hls::stream<ap_axis_512> in;
	hls::stream<ap_axis_512> out;

	ap_axis_512 temp;
	for(int j = 0; j < 200;j++){
		for(int k = 0; k < 64;k++){
			int a = j/2;
			int b = (j%2)*14+k;
			temp.data((k+1)*8-1,k*8) = 1;
//				temp.data((k+1)*8-1,k*8) = k%3;
			temp.keep(k,k) = 1;
		}
		in.write(temp);
	}
	top(in,out,1);
	for(int i = 0;i < 200;i++){
		ap_axis_512 OTemp = out.read();
		for(int j = 0; j < 32;j++){
			ap_uint<ABIT> q1 = OTemp.data((j+1)*16-1,j*16);
			cout << q1 << " " ;
		}
		cout << endl;
	}
}

int main(){
	// ConvStreamGenerator_Batch_Test();
	// ConvLayer_NoPad_Orbital_Test();
	gemmtest();
//	topTest();
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
