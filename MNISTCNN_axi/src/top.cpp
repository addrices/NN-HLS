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

template<unsigned Length>
ap_uint<32> max_array(ap_uint<32> *arr){
	int max = 0;
	int max_value = arr[0];
	for(int i = 1; i < Length;i++){
		if(arr[i] > max_value){
			max_value = arr[i];
			max = i;
		}
	}
	return max;
}

int top(ap_uint<32> *in){
#pragma HLS INTERFACE m_axi depth=784 port=in bundle=inMem
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS DATAFLOW
	const unsigned Batch = 1;
	const unsigned reps = 1;
	ap_uint<32> out[10];
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

	for(int i = 0;i < 28*28;i++){
		C1_in.write(in[i]);
	}
	C1:ConvLayer_NOPAD_Normal<Batch,C1_KSIZE,C1_SIZE,C1_INCHANNEL,C1_OUTCHANNEL,C1_INP,C1_MIDP_I,C1_MIDP_O,C1_OUTP,C1_STRIDE,WBIT,ABIT,C1_MBIT>(C1_in,C1_out,C1_W,C1_B,C1_SCALEBIT,reps);
	C2:ConvLayer_NOPAD_Normal<Batch,C2_KSIZE,C2_SIZE,C2_INCHANNEL,C2_OUTCHANNEL,C2_INP,C2_MIDP_I,C2_MIDP_O,C2_OUTP,C2_STRIDE,WBIT,ABIT,C2_MBIT>(C1_out,C2_out,C2_W,C2_B,C2_SCALEBIT,reps);

	P2:MaxPool_IOP<P2_PSIZE,ABIT,P2_SIZE,P2_CHANNEL*Batch,P2_INP*Batch,P2_OUTP*Batch>(C2_out,P2_out,reps);
	C3:ConvLayer_NOPAD_Normal<Batch,C3_KSIZE,C3_SIZE,C3_INCHANNEL,C3_OUTCHANNEL,C3_INP,C3_MIDP_I,C3_MIDP_O,C3_OUTP,C3_STRIDE,WBIT,ABIT,C3_MBIT>(P2_out,C3_out,C3_W,C3_B,C3_SCALEBIT,reps);
	P3:MaxPool_IOP<P3_PSIZE,ABIT,P3_SIZE,P3_CHANNEL*Batch,P3_INP*Batch,P3_OUTP*Batch>(C3_out,P3_out,reps);
	C4:ConvLayer_NOPAD_Normal<Batch,C4_KSIZE,C4_SIZE,C4_INCHANNEL,C4_OUTCHANNEL,C4_INP,C4_MIDP_I,C4_MIDP_O,C4_OUTP,C4_STRIDE,WBIT,ABIT,C4_MBIT>(P3_out,C4_out,C4_W,C4_B,C4_SCALEBIT,reps);
	P4:MaxPool_IOP<P4_PSIZE,ABIT,P4_SIZE,P4_CHANNEL*Batch,P4_INP*Batch,P4_OUTP*Batch>(C4_out,P4_out,reps);

	F5:FcnnLayer_Batch<Batch,F5_INPIX,F5_OUTPIX,F5_INSIZE,F5_OUTSIZE,WBIT,ABIT,F5_MBIT,F5_DEPTH,F5_WSIZE>(P4_out,F5_out,F5_W,F5_B,F5_SCALEBIT,reps);
	F6:FcnnLayer_Batch<Batch,F6_INPIX,F6_OUTPIX,F6_INSIZE,F6_OUTSIZE,WBIT,ABIT,F6_MBIT,F6_DEPTH,F6_WSIZE>(F5_out,F6_out,F6_W,F6_B,F6_SCALEBIT,reps);
	for(int i = 0;i < 10;i++){
		out[i] = F6_out.read();
	}
	int res = (int)max_array<10>(out);
	return res;
}

