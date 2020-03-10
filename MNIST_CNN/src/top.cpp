#include <iostream>
//#define AP_INT_MAX_W 2048
#include <ap_int.h>
#include <hls_stream.h>
#include "../../nn-h/conv.h"
#include "../../nn-h/FcnnLayer.h"
#include "../../nn-h/util.h"
#include "../../nn-h/pooling.h"
#include "../../nn-h/conv_systolic.h"
#include "config.h"


struct ap_axis{
	ap_uint<128> data;
	ap_int<1> last;
	ap_int<16> keep;
};

template<unsigned StrSize>
void AddLast(
		hls::stream<ap_uint<128> >& in,
		hls::stream<ap_axis>& out,
		unsigned reps = 1
){
	ap_axis temp;
	temp.keep = "0xffff";
	for(unsigned i = 0;i < StrSize*reps-1;i++){
#pragma HLS PIPELINE II = 1
		temp.data = in.read();
		temp.last = 0;
		out.write(temp);
	}
	temp.data = in.read();
	temp.last = 1;
	out.write(temp);
}

template<unsigned StrSize>
void DelHead(
		hls::stream<ap_axis>& in,
		hls::stream<ap_uint<128> >& out,
		unsigned reps = 1
){
	ap_axis temp;
	for(unsigned i = 0;i < StrSize*reps;i++){
#pragma HLS PIPELINE II = 1
		temp = in.read();
		out.write(temp.data);
		if(temp.last == 1)
			break;
	}
}

void top(hls::stream<ap_axis >& in,hls::stream<ap_axis >& out,unsigned reps = 1){
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE s_axilite port=reps bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS DATAFLOW
	hls::stream<ap_uint<8> > In8;
	hls::stream<ap_uint<ABIT> > C1_in;
	hls::stream<ap_uint<ABIT*C1_OUTCHANNEL> > C1_out;
	hls::stream<ap_uint<ABIT*C2_OUTCHANNEL> > C2_out;
	hls::stream<ap_uint<ABIT*P2_CHANNEL> > P2_out;
	hls::stream<ap_uint<ABIT*C3_OUTCHANNEL> > C3_out;
	hls::stream<ap_uint<ABIT*P3_CHANNEL> > P3_out;
	hls::stream<ap_uint<ABIT*C4_OUTCHANNEL> > C4_out;
	hls::stream<ap_uint<ABIT*P4_OUTP> > P4_out;
	hls::stream<ap_uint<ABIT*F5_OUTP> > F5_out;
	hls::stream<ap_uint<ABIT*F6_OUTP> > F6_out;
	hls::stream<ap_uint<128> > tin;
	hls::stream<ap_uint<14*8> > in112;
	hls::stream<ap_uint<128> > tout;

	DelHead<56>(in,tin,reps);
	ReduceStreamWidth_Length<128,14*8,56>(tin,in112,reps);
	splitStream_Length<14*8,8,56>(in112,In8,reps);
	ReduceStreamWidth_Length<8,ABIT,28*28>(In8,C1_in,reps);

	C1:ConvLayer_NOPAD_ScaleBit<C1_KSIZE,WBIT,ABIT,C1_MBIT,C1_INCHANNEL,C1_OUTCHANNEL,C1_STRIDE,C1_SIZE,C1_INP,C1_OUTP>(C1_in,C1_out,C1_W,C1_B,C1_SCALEBIT,reps);
	C2:ConvLayer_NOPAD_ScaleBit<C2_KSIZE,WBIT,ABIT,C2_MBIT,C2_INCHANNEL,C2_OUTCHANNEL,C2_STRIDE,C2_SIZE,C2_INP,C2_OUTP>(C1_out,C2_out,C2_W,C2_B,C2_SCALEBIT,reps);
	P2:MaxPool_Channel<P2_PSIZE,ABIT,P2_SIZE,P2_SIZE,P2_CHANNEL,P2_INP,P2_OUTP>(C2_out,P2_out,reps);
	C3:ConvLayer_NOPAD_ScaleBit<C3_KSIZE,WBIT,ABIT,C3_MBIT,C3_INCHANNEL,C3_OUTCHANNEL,C3_STRIDE,C3_SIZE,C3_INP,C3_OUTP>(P2_out,C3_out,C3_W,C3_B,C3_SCALEBIT,reps);
	P3:MaxPool_Channel<P3_PSIZE,ABIT,P3_SIZE,P3_SIZE,P3_CHANNEL,P3_INP,P3_OUTP>(C3_out,P3_out,reps);
	C4:ConvLayer_NOPAD_ScaleBit<C4_KSIZE,WBIT,ABIT,C4_MBIT,C4_INCHANNEL,C4_OUTCHANNEL,C4_STRIDE,C4_SIZE,C4_INP,C4_OUTP>(P3_out,C4_out,C4_W,C4_B,C4_SCALEBIT,reps);
	P4:MaxPool_Channel_T<P4_PSIZE,ABIT,P4_SIZE,P4_SIZE,P4_CHANNEL,P4_INP,P4_OUTP>(C4_out,P4_out,reps);
	//splitStream_Length<ABIT*P4_CHANNEL,ABIT*F5_INP,1>(P4_out,F5_in);

//	ExtendStreamWidth_Length<ABIT*P4_OUTP,128,8>(P4_out,tout);
//	AddLast<8>(tout,out);

	F5:FcnnLayer_ScaleBit<F5_INP,F5_OUTP,F5_IN,F5_OUT,WBIT,ABIT,F5_MBIT>(P4_out,F5_out,F5_W,F5_B,F5_SCALEBIT,reps);
	F6:FcnnLayer_ScaleBit<F6_INP,F6_OUTP,F6_IN,F6_OUT,WBIT,ABIT,F5_MBIT>(F5_out,F6_out,F6_W,F6_B,F6_SCALEBIT,reps);

	hls::stream<ap_uint<F6_OUTP*8> > res8_str;
	EleExtend<F6_OUTP,ABIT,8,1>(F6_out,res8_str,reps);

	ExtendStreamWidth_Length<F6_OUTP*8,128,1>(res8_str,tout,reps);
	AddLast<1>(tout,out,reps);
	return;
}

