#include <stdint.h>
#include <stdio.h>

#define trans(InChannel,OutChannel,Ksize,InWeight,OutWeight) 			\
	uint8_t OutWeight[OutChannel][Ksize*Ksize][InChannel];		\
	for(int i = 0; i < OutChannel;i++){						\
        for(int j = 0; j < InChannel;j++){					\
            for(int m = 0;m < Ksize;m++){					\
                for(int n = 0;n < Ksize;n++){				\
                    OutWeight[i][m*Ksize+n][j] = InWeight[i][j][m][n];\
				}											\
            }												\
        }													\
    }														\
															\
	printf("{");											\
	for(int i = 0;i < OutChannel;i++){						\
		printf("{");										\
		for(int j = 0;j < Ksize*Ksize;j++){					\
			printf("\"0x");									\
			for(int m = 0 ;m < InChannel;m++){				\
				printf("%02x",OutWeight[i][j][m]);				\
			}												\
			if(j != Ksize*Ksize-1)										\
				printf("\",");								\
			else											\
				printf("\"");								\
		}													\
		if(i != OutChannel-1)											\
			printf("},\n");									\
		else												\
			printf("}");									\
	}														\
	printf("};\n");											

#define trans_new(InChannel,OutChannel,Ksize,InWeight,OutWeight,InP,OutP) \
	{int InPack = InChannel/InP;										\
	int OutPack = OutChannel/OutP;								\
	uint8_t OutWeight[(InChannel/InP)*Ksize*Ksize*(OutChannel/OutP)][OutP][InP];\
	for(int i = 0; i < OutChannel;i++){						\
        for(int j = 0; j < InChannel;j++){					\
            for(int m = 0;m < Ksize;m++){					\
                for(int n = 0;n < Ksize;n++){				\
					int C_InPack = j / InP;					\
					int C_InP = j % InP;					\
					int C_OutPack = i / OutP;				\
					int C_OutP = i % OutP;					\
					int Offset = (m*Ksize+n)*InPack*OutPack+C_InPack*OutPack+C_OutPack;\
					OutWeight[Offset][C_OutP][C_InP] = InWeight[i][j][m][n];\
				}											\
            }												\
        }													\
    }														\
	printf("{");											\
	for(int i = 0;i < (InChannel/InP)*Ksize*Ksize*(OutChannel/OutP);i++){\
		printf("{");											\
		for(int j = 0; j < OutP;j++){						\
			printf("\"0x");									\
			for(int m = 0; m < InP;m++){					\
				printf("%02x",OutWeight[i][j][InP-1-m]);			\
			}												\
			if(j != OutP-1)									\
				printf("\",");								\
			else 											\
				printf("\"");								\
															\
		}													\
		if(i != (InChannel/InP)*Ksize*Ksize*(OutChannel/OutP)-1)\
			printf("},\n");										\
		else													\
			printf("}");											\
	}															\
	printf("};\n");}


int main(){
	printf("L1_W\n");
	trans_new(1,16,3,L1_W,L1_OUT_WEIGHT,1,4);
	printf("\n");
	printf("L2_W\n");
	trans_new(16,32,3,L2_W,L2_OUT_WEIGHT,8,16);
	printf("\n");
	printf("L3_W\n");
	trans_new(32,64,3,L3_W,L3,8,8);
	printf("\n");
	printf("L4_W\n");
	trans_new(64,64,3,L4_W,L4,4,4);

}