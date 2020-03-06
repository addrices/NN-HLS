#pragma once
#include <hls_stream.h>
#include <ap_int.h>
#include <iostream>
using namespace std;

template <	unsigned TopLeftPad,
			unsigned BottomRightPad,
			unsigned MatrixH,
			unsigned MatrixW,
			unsigned InChannel,
			unsigned Ibit>
void ZeroPad(					//add padding to image
	hls::stream<ap_uint<InChannel*Ibit> >& in,
	hls::stream<ap_uint<InChannel*Ibit> >& out)
{
	cout << TopLeftPad << "BR" << BottomRightPad << endl;
	const unsigned OutH = (MatrixH+TopLeftPad+BottomRightPad);
	const unsigned OutW = (MatrixW+TopLeftPad+BottomRightPad);
	ap_uint<InChannel*Ibit> temp_out = 0;
	for (unsigned h = 0; h < TopLeftPad; h++) {
		for (unsigned s = 0; s < OutW; s++) {
			out.write(0);
		}
	}
	
	for (unsigned h = 0; h < MatrixH; h++) {
		for ( unsigned s = 0; s < OutW; s++ ) {
#pragma HLS PIPELINE II=1
			if ( (s < TopLeftPad) || (s >= OutW-BottomRightPad) ) {
				temp_out = 0;
			}
			else {
				temp_out = in.read();
			}
			out.write(temp_out);
		}
	}

	for (unsigned h = 0; h < BottomRightPad; h++) {
		for (unsigned i = 0; i < OutW; i++) {
			out.write(0);
		}
	}
}
