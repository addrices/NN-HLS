#define AP_INT_MAX_W 8192
#include "../../nn-h/FcnnLayer.h"
#include "../../nn-h/FcnnLayer_systolic.h"
#include "../../nn-h/util.h"
#include "../../nn-h/conv.h"
#include "../../nn-h/pooling.h"
#include "../../nn-h/io.h"
#include "../../nn-h/conv_systolic.h"
#include "config.h"
#include <ap_int.h>
#include <hls_stream.h>




// unsigned depth = 9;
// unsigned wbit = 8;
// unsigned abit = 8;
// unsigned mbit = 16;
// unsigned asize = 8;
// unsigned wsize = 8;
// void top(hls::stream<ap_int<depth*abit*asize> >& a,hls::stream<ap_uint<depth*abit*asize> >& b,hls::stream<ap_int<20*8*8> >& c){
// 	template<unsigned Depth,unsigned ASize,unsigned WSize,unsigned ABit,unsigned WBit,unsigned MBit>
// 	ap_int<8*8*9> act = a.read;

// }

// const unsigned asize = 8;
// const unsigned wsize = 8;
// const unsigned depth = 9;
// const unsigned abit = 8;
// const unsigned wbit = 8;
// const unsigned mbit = 16;
// const unsigned pixs = 100;
// void top(hls::stream<ap_uint<abit*asize> >& act,hls::stream<ap_int<wbit*wsize> >& wei,hls::stream<ap_int<mbit*asize*wsize> >& out){
// 	Gemm_str<depth,asize,wsize,abit,wbit,mbit,pixs>(act,wei,out,1);
// }


//void top(hls::stream<ap_axis >& in,hls::stream<ap_axis >& out,unsigned reps = 1){
//#pragma HLS INTERFACE axis register both port=in
//#pragma HLS INTERFACE axis register both port=out
//#pragma HLS INTERFACE s_axilite port=reps bundle=control
//#pragma HLS INTERFACE s_axilite port=return bundle=control
//#pragma HLS DATAFLOW
//	const unsigned Batch = 4;
//	hls::stream<ap_uint<8> > In8;
//
//	hls::stream<ap_uint<Batch*C1_INP*ABIT> > C1_in;
//	hls::stream<ap_uint<Batch*C1_OUTP*ABIT> > C1_out;
//	hls::stream<ap_uint<Batch*C2_OUTP*ABIT> > C2_out;
//	hls::stream<ap_uint<Batch*P2_OUTP*ABIT> > P2_out;
//	hls::stream<ap_uint<Batch*C3_OUTP*ABIT> > C3_out;
//	hls::stream<ap_uint<Batch*P3_OUTP*ABIT> > P3_out;
//	hls::stream<ap_uint<Batch*C4_OUTP*ABIT> > C4_out;
//	hls::stream<ap_uint<Batch*P4_OUTP*ABIT> > P4_out;
//
//	hls::stream<ap_uint<Batch*F5_OUTPIX*ABIT> > F5_out;
//	hls::stream<ap_uint<Batch*F6_OUTPIX*ABIT> > F6_out;
//
//
//	hls::stream<ap_uint<128> > tin;
//	hls::stream<ap_uint<14*8> > in112;
//	hls::stream<ap_uint<128> > tout;
//
//	DelHead<56*Batch>(in,tin,reps);
//	ReduceStreamWidth_Length<128,14*8,56*Batch>(tin,in112,reps);
//	Trans_BatchStr<Batch,8,ABIT,14,C1_INP,56>(in112,C1_in,reps);
//
//	C1:ConvLayer_NOPAD_Orbital<Batch,C1_KSIZE,C1_SIZE,C1_INCHANNEL,C1_OUTCHANNEL,C1_INP,C1_MIDP_I,C1_MIDP_O,C1_OUTP,C1_STRIDE,WBIT,ABIT,C1_MBIT>(C1_in,C1_out,C1_W,C1_B,C1_SCALEBIT,reps);
////	unsigned OutPack = C1_OUTCHANNEL/C1_OUTP;
////	for(int i = 0;i < 26;i++){
////		for(int j = 0;j < 26;j++){
////			for(int m = 0;m < OutPack;m++){
////				ap_uint<Batch*C1_OUTP*ABIT> otemp = C1_out.read();
////				for(int n = 0;n < C1_OUTP;n++){
////					for(int q = 0;q < Batch;q++){
////						unsigned offset = n*Batch+q;
////						ap_uint<ABIT> cc = otemp((offset+1)*ABIT-1,offset*ABIT);
////						if(q == 0)
////							cout << cc << " " ;
////					}
////				}
////			}
////			cout << endl;
////		}
////	}
////	return;
//	C2:ConvLayer_NOPAD_Orbital<Batch,C2_KSIZE,C2_SIZE,C2_INCHANNEL,C2_OUTCHANNEL,C2_INP,C2_MIDP_I,C2_MIDP_O,C2_OUTP,C2_STRIDE,WBIT,ABIT,C2_MBIT>(C1_out,C2_out,C2_W,C2_B,C2_SCALEBIT,reps);
////		unsigned OutPack = C2_OUTCHANNEL/C2_OUTP;
////		for(int i = 0;i < 24;i++){
////			for(int j = 0;j < 24;j++){
////				for(int m = 0;m < OutPack;m++){
////					ap_uint<Batch*C2_OUTP*ABIT> otemp = C2_out.read();
////					for(int n = 0;n < C2_OUTP;n++){
////						for(int q = 0;q < Batch;q++){
////							unsigned offset = n*Batch+q;
////							ap_uint<ABIT> cc = otemp((offset+1)*ABIT-1,offset*ABIT);
////							cout << cc << " " ;
////						}
////					}
////				}
////				cout << endl;
////			}
////		}
////		return;
//
//	P2:MaxPool_IOP<P2_PSIZE,ABIT,P2_SIZE,P2_CHANNEL*Batch,P2_INP*Batch,P2_OUTP*Batch>(C2_out,P2_out,reps);
//	C3:ConvLayer_NOPAD_Orbital<Batch,C3_KSIZE,C3_SIZE,C3_INCHANNEL,C3_OUTCHANNEL,C3_INP,C3_MIDP_I,C3_MIDP_O,C3_OUTP,C3_STRIDE,WBIT,ABIT,C3_MBIT>(P2_out,C3_out,C3_W,C3_B,C3_SCALEBIT,reps);
//	P3:MaxPool_IOP<P3_PSIZE,ABIT,P3_SIZE,P3_CHANNEL*Batch,P3_INP*Batch,P3_OUTP*Batch>(C3_out,P3_out,reps);
//	C4:ConvLayer_NOPAD_Orbital<Batch,C4_KSIZE,C4_SIZE,C4_INCHANNEL,C4_OUTCHANNEL,C4_INP,C4_MIDP_I,C4_MIDP_O,C4_OUTP,C4_STRIDE,WBIT,ABIT,C4_MBIT>(P3_out,C4_out,C4_W,C4_B,C4_SCALEBIT,reps);
//	P4:MaxPool_IOP<P4_PSIZE,ABIT,P4_SIZE,P4_CHANNEL*Batch,P4_INP*Batch,P4_OUTP*Batch>(C4_out,P4_out,reps);
//
//
//	F5:FcnnLayer_Batch<Batch,F5_INPIX,F5_OUTPIX,F5_INSIZE,F5_OUTSIZE,WBIT,ABIT,F5_MBIT,F5_DEPTH,F5_WSIZE>(P4_out,F5_out,F5_W,F5_B,F5_SCALEBIT,1);
//	F6:FcnnLayer_Batch<Batch,F6_INPIX,F6_OUTPIX,F6_INSIZE,F6_OUTSIZE,WBIT,ABIT,F6_MBIT,F6_DEPTH,F6_WSIZE>(F5_out,F6_out,F6_W,F6_B,F6_SCALEBIT,1);
//
//	hls::stream<ap_uint<Batch*F6_OUTPIX*8> > res8_str;
//	EleExtend<F6_OUTPIX*Batch,ABIT,8,10>(F6_out,res8_str,reps);
//				//1    * 4
//	ExtendStreamWidth_Length<Batch*F6_OUTPIX*8,128,10>(res8_str,tout,reps);
//	AddLast<10>(tout,out,reps);
//	return;
//}

//ap_int<1024> Normal_Gemm_8(ap_uint<512> a,ap_int<512> b){
//	ap_int<1024> otemp;
//	ap_int<16> res[4][4][4];
//	ap_uint<8> A[4][4][4];
//	ap_int<8> B[4][4][4];
//#pragma HLS resource variable=res core=RAM_1P_LUTRAM
//#pragma HLS array_partition variable=res complete
//#pragma HLS array_partition variable=A complete
//#pragma HLS array_partition variable=B complete
//
//	for(unsigned i = 0;i < 4;i++){
//#pragma HLS UNROLL
//		for(unsigned m = 0;m < 4;m++){
//#pragma HLS UNROLL
//			for(unsigned n = 0;n < 4;n++){
//#pragma HLS UNROLL
//				unsigned off = i*16+m*4+n;
//				A[i][m][n] = a((off+1)*8-1,off*8);
//				B[i][m][n] = b((off+1)*8-1,off*8);
//				res[i][m][n] = 0;
//			}
//		}
//	}
//	ap_int<256> temp;
//
//	for(ap_uint<4> num = 0;num < 8;num++){
//#pragma HLS PIPELINE II = 1
//		ap_uint<2> a_n,b_n;
//		b_n = num(1,0);
//		a_n = num(2,2)*2+num(0,0);
//		ap_uint<2> i_off = num(2,1);
////		cout << i_off << endl;
//		for(unsigned i = 0;i < 4;i++){
//#pragma HLS UNROLL
//			for(unsigned j = 0;j < 4;j++){
//#pragma HLS UNROLL
//				for(unsigned k = 0;k < 4;k++){
//#pragma HLS UNROLL
//					ap_int<16> tmp = A[a_n][i][k]*B[b_n][j][k];
//#pragma HLS resource variable=tmp core=Mul_LUT
////					cout << res[i_off][i][j] << " ";
//					res[i_off][i][j] += tmp;
////					cout << tmp << " " << res[i_off][i][j] << " ";
//				}
//			}
//		}
////		cout << endl;
//	}
//	for(unsigned i = 0;i < 4;i++){
//#pragma HLS UNROLL
//		for(unsigned j = 0;j < 4;j++){
//#pragma HLS UNROLL
//			for(unsigned k = 0;k < 4;k++){
//#pragma HLS UNROLL
//				unsigned off = i*16+j*4+k;
//				otemp((off+1)*16-1,off*16) = res[i][j][k];
//			}
//		}
//	}
//	return otemp;
//}

ap_int<1024> Normal_Gemm_8(ap_uint<512> a,ap_int<512> b){
	ap_int<1024> otemp;
	ap_int<256> res[4];
	ap_uint<128> A[4];
	ap_int<128> B[4];
#pragma HLS array_partition variable=res complete
#pragma HLS array_partition variable=A complete
#pragma HLS array_partition variable=B complete
	for(unsigned i = 0;i < 4;i++){
#pragma HLS UNROLL
		A[i] = a((i+1)*128-1,i*128);
		B[i] = b((i+1)*128-1,i*128);
		res[i] = 0;
	}
	for(ap_uint<4> num = 0;num < 8;num++){
#pragma HLS PIPELINE II = 1
		ap_uint<2> a_n,b_n;
		b_n = num(1,0);
		a_n = num(2,2)*2+num(0,0);
		ap_uint<2> i_off = num(2,1);
		Normal_GemmAdd<4,4,4,8,8,16>(A[a_n],B[b_n],res[i_off]);
	}
	for(unsigned i = 0;i < 4;i++){
#pragma HLS UNROLL
		otemp((i+1)*256-1,i*256) = res[i];
	}
	return otemp;
}

void top(hls::stream<ap_axis_512 >& in,hls::stream<ap_axis_512 >& out,unsigned reps = 1){
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE s_axilite port=reps bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
	#pragma HLS DATAFLOW
		hls::stream<ap_uint<512> > tin;
		hls::stream<ap_uint<512> > tout;

		DelHead_512<200>(in,tin,reps);
		for(unsigned rep = 0;rep < reps;rep++){
			for(unsigned i = 0;i < 100;i++){
				ap_uint<512> a,b;
				ap_uint<1024> res;
				a = tin.read();
				b = tin.read();
				res = Orbital_Gemm<8,8,8,8,8,16>(a,b);
//				res = Normal_Gemm_8(a,b);
//				res = Dot_Gemm<8,8,8,8,8,16>(a,b);
				tout.write(res(511,0));
				tout.write(res(1023,512));

			}
		}

		AddLast_512<200>(tout,out,reps);
}

