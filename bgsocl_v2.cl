#define GPU 1

#if defined GPU
typedef global int* Int_ptr;
typedef global char* Char_ptr;
typedef global uchar* Uchar_ptr;
typedef global short* Short_PTR;
typedef global unsigned int* Uint_ptr;
#else
typedef unsigned int* Uint_ptr;
typedef int* Int_ptr;
typedef char* Char_ptr;
typedef uchar* Uchar_ptr;
typedef short* Short_PTR;
#endif

#define randomSize 2048
//////////////////////////////
#define randSubSample 8
#define nchannels 1
//////////////////////////////
#define N 20
#define R 32
#define R2 400
#define R2f 400.0
#define Rf 20.0
#define cMin 2
// SHADOW Parameters
#define SH_MAX_CORRELATION 0.8
#define SH_MIN_ALFA 0.4
#define SH_MAX_ALFA 1.0
#define SH_STDC 0.15

#ifndef GPU
#include <omp.h>
int thID = -1;
void set_threadId(int th)
{
	thID = th;
}
int get_global_id(int dim)
{
	if (thID>0)
		return thID;
	else
		return omp_get_thread_num();
}
#endif




#ifdef GPU
kernel
#endif
void kernelInitialization(Uchar_ptr samples, Uchar_ptr input, int width, int height, Uchar_ptr blinkingMap)
{
	long big = width*height;

	int index = get_global_id(0);
	int x = index % width;
	int y = index / width;

	if ((x<width) && (y<height)) {
		for (int subS = 0; subS<N; subS++) {
			long base = big * subS;
			base += y * width;
			base += x;
			samples[base] = input[y * width + x];

			blinkingMap[y*width + x] = 0;
		}

	}

}



#ifdef GPU
kernel
#endif
void erode(Uchar_ptr input, Uchar_ptr output, int width, int height, int radius) {
	//Closing(3), filter BLOBs area < 15, Fill holes
	int index = get_global_id(0);
	
	if (index>= width * height) return;
	int minI = 255;
	
	int i = index % width;
	int j = index / width;

	for (int n = -radius; n < radius; n++)
		for (int m = -radius; m < radius; m++)
			if ((j + n >= 0) && (j + n < height) && (i + m >= 0) && (i + m < width))
			{
				int colorI = input[(j + n)*width + (i + m) + 0];
				minI = min(colorI, minI);
			};
	output[index] = minI;
}
#ifdef GPU
kernel
#endif
void dilate(Uchar_ptr input, Uchar_ptr output, int width, int height, int radius) {
	//Closing(3), filter BLOBs area < 15, Fill holes
	int maxI = 0;
	int index = get_global_id(0);
	int i = index % width;
	int j = index / width;

	for (int n = -radius; n < radius; n++)
		for (int m = -radius; m < radius; m++)
			if ((j + n >= 0) && (j + n < height) && (i + m >= 0) && (i + m < width))
			{
				int colorI = input[(j + n)*width + (i + m) + 0];
				maxI = max(colorI, maxI);
			};
	output[index] = maxI;
}

#ifdef GPU
kernel
#endif
void postProcess(Uchar_ptr input, Uchar_ptr output, int width, int height) {
	//Closing(3), filter BLOBs area < 15, Fill holes
	erode(input, output, width, height, 3);
	dilate(input, output, width, height, 3);

}

#ifdef GPU
kernel
#endif
void copyBuffer(Uchar_ptr orig, Uchar_ptr dest, int width) {
	int index = get_global_id(0);
	int x = index % width;
	int y = index / width;

	dest[y*width + x] = orig[y*width + x];
}


#ifdef GPU
kernel
#endif
void blinking(Uchar_ptr bf, Uchar_ptr segmentPrevia, Uchar_ptr segmentActual, Uchar_ptr blinkingMap, Uchar_ptr Update
	, int width, int height) {
	int index = get_global_id(0);
	int x = index % width;
	int y = index / width;


	if (!((x == 0) || (y == 0) || (x == width - 1) || (y == height - 1))) {

		//Actualizar nivel de blinkeo
		if (bf[y * width + x] == 0 //Clasificado como Background
			&&
			segmentPrevia[y * width + x] != segmentActual[y * width + x] //Blinkeo
																		 //Inner border of BG?
			&& ((bf[(y - 1) * width + x - 1] != 0) || (bf[y * width + x - 1] != 0) || (bf[(y - 1) * width + x] != 0) ||
				(bf[y * width + x + 1] != 0) || (bf[(y + 1) * width + x] != 0) || (bf[(y + 1) * width + x + 1] != 0) ||
				(bf[(y - 1) * width + x + 1] != 0) || (bf[(y + 1) * width + x - 1] != 0))) {

			if (blinkingMap[y * width + x] <= 135)
				blinkingMap[y * width + x] += 15;

		}
		else {
			if (blinkingMap[y * width + x] > 0)
				blinkingMap[y * width + x] -= 1;
		}

		//Actualizar modelo si no esta blinkeando y fue clasificado como Background
		if ((blinkingMap[y * width + x] < 30) && (bf[y * width + x] == 0)) {
			//ACTUALIZAR EL MODELO ! probReemplzo
			Update[y * width + x] = 0;
			//PROPAGAR !
		}
		else {
			//NO ACTUALIZAR EL MODELO
			Update[y * width + x] = 255;
		}
	}
	else {	//Borde !
		if (bf[y * width + x] == 0) {
			//ACTUALIZAR EL MODELO, CUIDADO CON LOS BORDES !
			Update[y * width + x] = 0;
		}
		else {
			//NO ACTUALIZAR EL MODELO
			Update[y * width + x] = 255;
		}
	}

}

#ifdef GPU
kernel
#endif
void computeShadowMask(Uchar_ptr inputData, Uchar_ptr meanInputData, Uchar_ptr _bgrModelData, int width, int height,int radius)
{
	int index = get_global_id(0);
	int i = index % width;
	int j = index / width;

	float C = 0;
	float Varianzai = 0;
	float Varianzaf = 0;
	float Varianzac = 0;
	float mediac = 0;
	float stdc = 0;
	float alfa = 0;
	int vecinos = 0;
	float colori = 0;
	float colorf = 0;
	int vecinosSh = 0;

	int limsupY = height - 1;
	int liminfY = 1;
	int limsupX = width - 1;
	int liminfX = 1;

	if (_bgrModelData[j * width + i + 0] != 0)
	{
		C = 0;
		Varianzai = 0;
		Varianzaf = 0;
		Varianzac = 0;
		mediac = 0;
		stdc = 0;
		alfa = 0;
		vecinos = 0;
		colori = 0;
		colorf = 0; //outputData[j, i, 0] = 128;
		for (int n = -radius; n < radius; n++)
			for (int m = -radius; m < radius; m++)
				if ((j + n >= liminfY) && (j + n < limsupY) && (i + m >= liminfX) && (i + m < limsupX))
				{
					colori = inputData[(j + n)*width + (i + m) + 0];
					colorf = meanInputData[(j + n)*width + (i + m) + 0];
					C += colori * colorf;
					Varianzai += colori * colori;
					Varianzaf += colorf * colorf;
					Varianzac += (colori / colorf) * (colori / colorf);
					mediac += (colori / colorf);
					vecinos++;
					if (_bgrModelData[(j + n)*width + (i + m) + 0] != 0)
						vecinosSh++;
				};

		C = (C / (vecinos + 0.001)) / (sqrt(Varianzai / (vecinos + 0.001)) * sqrt(Varianzaf / (vecinos + 0.001)));
		Varianzac = Varianzac / vecinos; mediac = mediac / vecinos;
		stdc = sqrt(Varianzac - mediac * mediac);
		colori = inputData[(j)*width + (i)+0];
		colorf = meanInputData[(j)*width + (i)+0];
		alfa = colori / colorf;
		if ((C >= SH_MAX_CORRELATION) && (Varianzai < Varianzaf) && (vecinosSh>0))
			if ((stdc < SH_STDC) && (alfa >= SH_MIN_ALFA) && (alfa < SH_MAX_ALFA))
			{
				_bgrModelData[j * width + i + 0] = 1; // Aumentar para visualizar
			}
	}; //endif mascara

}; //end area


int randomfunc2(Int_ptr randomValues, int numIteracion, int x, int y, int* iv)
{
	(*iv)++;
	return  randomValues[(numIteracion* x * 11 + (*iv) * 5) % 2048] + randomValues[(numIteracion *  y * 32 + (*iv) * 3) % 1024];

}

#define RADIUS_N 7


#ifdef GPU
kernel
#endif
void detect(Uchar_ptr samples, Uchar_ptr input, Uchar_ptr bf, int width, int height)
{
	int big = width*height;

	int index = get_global_id(0);
	
	if (index >= width * height) return;
	
	int x = index % width;
	int y = index / width;

	int count = 0;
	int ind = 0;
	int I0i = input[index];

	while ((count<cMin) && (ind<N)) 
	{

		int base = big * ind + index;
				
		int S0i = samples[base];

		if (abs(I0i - S0i) < R)
		{
			count++;
		}

		ind++;
	}
	if (count >= cMin)
		bf[index] = 0;//BACKGROUND
	else
		bf[index] = 255;//FOREGROUND


}

#ifdef GPU
kernel
#endif
void update_model(Uchar_ptr input, Uchar_ptr samples, Uchar_ptr bf, int width, int height,
	int numIteracion, Int_ptr randomValues)
{
	int index = get_global_id(0);
	int lind = get_local_id(0);
	if (index >= width * height) return;
	int x = index % width;
	int y = index / width;

	int big = width*height;
	int X_off[9] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
	int Y_off[9] = { -1, -1,-1, 1, 1, 1, 0, 0, 0 };
	int invocation = 0;

     
    if (numIteracion<N)
	{
	    int base =  width*height * numIteracion + index;
		samples[base] = input[index];
		return;
	}
	
		

	//Si fue considerado foreground
	//if (bf[index] == 1)
	{

		int random = (numIteracion+lind*3) % randSubSample;
		int nIndex = (y + Y_off[numIteracion % 9]) * width + x + X_off[numIteracion % 9];
		if ((random == 0) && (nIndex >= 0) && (nIndex < width*height))
		{
			int randSubS = numIteracion % N;
			int base = width * height * randSubS + index;
			samples[base] = input[nIndex];

		}



	}


}

