#ifndef CL_KERNELS_H
#define CL_KERNELS_H


#define randomSize 2048
//////////////////////////////
#define randSubSample 8
#define nchannels 1
//////////////////////////////
#define N 20
#define R 20
#define R2 400
#define R2f 400.0
#define Rf 20.0
#define cMin 2

#define BLACK 0
#define WHITE 255

#define MAX_NUM_POINTS 8192

#ifndef GPU
#include <omp.h>

void set_threadId(int th) ;
int get_global_id(int dim);
#endif

typedef struct _box
{
	int numPoints;
	int minX, maxX, minY, maxY;
	int sX, sY, eX, eY;
	int points[MAX_NUM_POINTS];
	int isClosed;
	int blobID;
} boxCL;

#if defined GPU
typedef global int* Int_ptr ;
typedef global char* Char_ptr ;
typedef global uchar* Uchar_ptr ;
typedef global short* Short_PTR;
typedef global unsigned int* Uint_ptr ;
typedef global boxCL* Box_Ptr;
#else
typedef unsigned char uchar;
typedef unsigned int* Uint_ptr ;
typedef int* Int_ptr ;
typedef char* Char_ptr ;
typedef uchar* Uchar_ptr ;
typedef short* Short_PTR;
typedef boxCL* Box_Ptr;
#endif


#ifdef GPU
kernel
#endif
void kernelInitialization(Uchar_ptr samples,Uchar_ptr input, int width , int height, Uchar_ptr blinkingMap);

#ifdef GPU
kernel
#endif
void detect(Uchar_ptr samples, Uchar_ptr input, Uchar_ptr bf, int width , int height);

#ifdef GPU
kernel
#endif
void postProcess(Uchar_ptr bf,int width,int height);

#ifdef GPU
kernel
#endif
void copyBuffer(Uchar_ptr orig,Uchar_ptr dest,int width);


#ifdef GPU
kernel
#endif
void blinking(Uchar_ptr bf,Uchar_ptr segmentPrevia,Uchar_ptr segmentActual,Uchar_ptr blinkingMap,Uchar_ptr Update
              ,int width, int height);

#ifdef GPU
kernel
#endif
void initContourData(Uchar_ptr maskP, Uchar_ptr ucopy, Uchar_ptr bfP,Uchar_ptr padImage,Uchar_ptr padMask,Box_Ptr boxes, int width, int height, int thID, int numBoxes);

#ifdef GPU
kernel
#endif
 void closing(Uchar_ptr input, Uchar_ptr out, int width, int height, int rad, int thID);


#ifdef GPU
kernel
#endif
void actualizar_modelo(Uchar_ptr input,Uchar_ptr samples,Uchar_ptr Update, int width, int height,
                       int numIteracion,Uint_ptr randomValues);

#ifdef GPU
kernel
#endif
void mooreNeighborTracing(Uchar_ptr image,Uchar_ptr maskImage,Uchar_ptr paddedImage, Uchar_ptr paddedMask, int width, int height, Box_Ptr boxes, int numBoxes, int thIndex, int numBlocks);
#endif 

