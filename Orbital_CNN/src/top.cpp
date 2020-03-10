#define AP_INT_MAX_W 8192
#include "../../nn-h/FcnnLayer.h"
#include "../../nn-h/FcnnLayer_systolic.h"
#include "../../nn-h/util.h"
#include "../../nn-h/conv.h"
#include "../../nn-h/pooling.h"
#include "../../nn-h/conv_systolic.h"
#include "config.h"
#include <ap_int.h>
#include <hls_stream.h>


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
	const unsigned Batch = 4;
	hls::stream<ap_uint<8> > In8;

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


	hls::stream<ap_uint<128> > tin;
	hls::stream<ap_uint<14*8> > in112;
	hls::stream<ap_uint<128> > tout;

	DelHead<56>(in,tin,reps);
	ReduceStreamWidth_Length<128,14*8,56>(tin,in112,reps);
	splitStream_Length<14*8,8,56>(in112,In8,reps);
	ReduceStreamWidth_Length<8,ABIT,28*28>(In8,C1_in,reps);

	C1:ConvLayer_NOPAD_Orbital<Batch,C1_KSIZE,C1_SIZE,C1_INCHANNEL,C1_OUTCHANNEL,C1_INP,C1_MIDP_I,C1_MIDP_O,C1_OUTP,C1_STRIDE,WBIT,ABIT,C1_MBIT>(C1_in,C1_out,C1_W,C1_B,C1_SCALEBIT,1);
	C2:ConvLayer_NOPAD_Orbital<Batch,C2_KSIZE,C2_SIZE,C2_INCHANNEL,C2_OUTCHANNEL,C2_INP,C2_MIDP_I,C2_MIDP_O,C2_OUTP,C2_STRIDE,WBIT,ABIT,C2_MBIT>(C1_out,C2_out,C2_W,C2_B,C2_SCALEBIT,1);
	P2:MaxPool_IOP<P2_PSIZE,ABIT,P2_SIZE,P2_CHANNEL*Batch,P2_INP*Batch,P2_OUTP*Batch>(C2_out,P2_out);
	C3:ConvLayer_NOPAD_Orbital<Batch,C3_KSIZE,C3_SIZE,C3_INCHANNEL,C3_OUTCHANNEL,C3_INP,C3_MIDP_I,C3_MIDP_O,C3_OUTP,C3_STRIDE,WBIT,ABIT,C3_MBIT>(P2_out,C3_out,C3_W,C3_B,C3_SCALEBIT,1);
	P3:MaxPool_IOP<P3_PSIZE,ABIT,P3_SIZE,P3_CHANNEL*Batch,P3_INP*Batch,P3_OUTP*Batch>(C3_out,P3_out);
	C4:ConvLayer_NOPAD_Orbital<Batch,C4_KSIZE,C4_SIZE,C4_INCHANNEL,C4_OUTCHANNEL,C4_INP,C4_MIDP_I,C4_MIDP_O,C4_OUTP,C4_STRIDE,WBIT,ABIT,C4_MBIT>(P3_out,C4_out,C4_W,C4_B,C4_SCALEBIT,1);
	P4:MaxPool_IOP<P4_PSIZE,ABIT,P4_SIZE,P4_CHANNEL*Batch,P4_INP*Batch,P4_OUTP*Batch>(C4_out,P4_out);
	//splitStream_Length<ABIT*P4_CHANNEL,ABIT*F5_INP,1>(P4_out,F5_in);

//	ExtendStreamWidth_Length<ABIT*P4_OUTP,128,8>(P4_out,tout);
//	AddLast<8>(tout,out);

	F5:FcnnLayer_Batch<Batch,F5_INPIX,F5_OUTPIX,F5_INSIZE,F5_OUTSIZE,WBIT,ABIT,F5_MBIT,F5_DEPTH,F5_WSIZE>(P4_out,F5_out,F5_W,F5_B,F5_SCALEBIT,1);
	F6:FcnnLayer_Batch<Batch,F6_INPIX,F6_OUTPIX,F6_INSIZE,F6_OUTSIZE,WBIT,ABIT,F6_MBIT,F6_DEPTH,F6_WSIZE>(F5_out,F6_out,F6_W,F6_B,F6_SCALEBIT,1);

	hls::stream<ap_uint<F6_OUTP*8> > res8_str;
	EleExtend<F6_OUTP,ABIT,8,1>(F6_out,res8_str,reps);

	ExtendStreamWidth_Length<F6_OUTP*8,128,1>(res8_str,tout,reps);
	AddLast<1>(tout,out,reps);
	return;
}

