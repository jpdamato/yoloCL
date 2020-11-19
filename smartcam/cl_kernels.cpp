#include "cl_kernels.h"

#ifndef GPU
#include <omp.h>

#define abs(v) v<0? -v:v


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
	void kernelInitialization(Uchar_ptr samples,Uchar_ptr input, int width , int height, Uchar_ptr blinkingMap)
{
	long big = width*height;

	int index = get_global_id(0);
	int x = index % width;
	int y = index / width;

	if ((x<width)&&(y<height)){
		for (int subS=0;subS<N;subS++){
			long base = big * subS;
			base+= y * width;
			base+= x;
			samples[base] =  input[y * width  + x];

			blinkingMap[y*width + x]=0;
		}

	}

}

#ifdef GPU
kernel
#endif
	void detect(Uchar_ptr samples, Uchar_ptr input, Uchar_ptr bf, int width , int height)
{
	long big = width*height;

	int index = get_global_id(0);
	int x = index % width;
	int y = index / width;

	int count=0;
	int ind=0;
	while ((count<cMin)&&(ind<N)){

		long base = big * ind;
		base+= y * width;
		base+= x;


		uchar I0i = input[y * width  + x];
		uchar S0i = samples[base];

		if (abs(I0i-S0i) < R)
			count++;

		ind++;
	}

	if (count >= cMin)
		bf[y * width + x]=0;//BACKGROUND
	else
		bf[y * width + x]=255;//FOREGROUND


}

#ifdef GPU
kernel
#endif
	void postProcess(Uchar_ptr bf,int width,int height){
		//Closing(3), filter BLOBs area < 15, Fill holes
}

#ifdef GPU
kernel
#endif
	void copyBuffer(Uchar_ptr orig,Uchar_ptr dest,int width){
		int index = get_global_id(0);
		int x = index % width;
		int y = index / width;

		dest[y*width + x]=orig[y*width + x];
}


#ifdef GPU
kernel
#endif
	void blinking(Uchar_ptr bf,Uchar_ptr segmentPrevia,Uchar_ptr segmentActual,Uchar_ptr blinkingMap,Uchar_ptr Update
	,int width, int height){
		int index = get_global_id(0);
		int x = index % width;
		int y = index / width;


		if (!( (x==0)||(y==0)||(x==width-1)||(y==height-1) )){

			//Actualizar nivel de blinkeo
			if ( bf[y * width + x]==0 //Clasificado como Background
				&&
				segmentPrevia[y * width + x] != segmentActual[y * width + x] //Blinkeo
			//Inner border of BG?
			&& ((bf[(y-1) * width + x-1] != 0) || (bf[y * width + x-1] != 0) || (bf[(y-1) * width + x] != 0) ||
				(bf[y * width + x+1] != 0) || (bf[(y+1) * width + x] != 0) || (bf[(y+1) * width + x+1] != 0) ||
				(bf[(y-1) * width + x+1] != 0) || (bf[(y+1) * width + x-1] != 0)) ){

					if (blinkingMap[y * width + x] <= 135 )
						blinkingMap[y * width + x] += 15;

			}
			else {
				if (blinkingMap[y * width + x] > 0 )
					blinkingMap[y * width + x] -= 1;
			}

			//Actualizar modelo si no esta blinkeando y fue clasificado como Background
			if ( (blinkingMap[y * width + x] < 30) && (bf[y * width + x] == 0) ){
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
			if (bf[y * width + x] == 0){
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
	void initContourData(Uchar_ptr maskP, Uchar_ptr ucopy, Uchar_ptr bfP,Uchar_ptr padImage,Uchar_ptr padMask,Box_Ptr boxes, int width, int height, int thID, int numBoxes)
{
	int index ;
	if (thID<0)     index = get_global_id(0);
	else index = thID;

	int x = index % width;
	int y = index / width;

	if (index<numBoxes)
	{		
		boxes[index].minX = 10000;
		boxes[index].minY = 10000;
		boxes[index].maxX = -10000;
		boxes[index].maxY = -10000;
		boxes[index].numPoints = 0;
		boxes[index].isClosed = 0;
	}



	if ((x<0) || (y<0) ||  (x>=width) || (y>=height))  return;
	maskP[index] = 0; 
	ucopy[index] = bfP[index];

	padImage[(x+1)+(y+1)*(width+2)] = bfP[x + y*width];
	padMask[(x+1)+(y+1)*(width+2)] = WHITE;


}

#ifdef GPU
kernel
#endif
	void closing(Uchar_ptr input, Uchar_ptr out, int width, int height, int rad, int thID)
{
	int index ;
	if (thID<0)     index = get_global_id(0);
	else index = thID;


	int x = index % width;
	int y = index / width;

	if ((x<rad) || (y<rad) ||  (x>=width-rad) || (y>=height-rad))  return; 

	if (rad<=0) 
	{
		out[index] = input[index];		  
		return ; 
	}		

	int imax = 0;

	for (int i=-rad; i<=rad; i++)
		for (int j=-rad; j<=rad; j++)
			if ( (((y+i)*width +x+j) >=0) && ((y+i)*width +x+j < width * height) )
			{
				if (input[ (y+i)*width +x+j]>imax)
					imax = input[ (y+i)*width +x+j];
			}
			out[y * width + x] = imax;
}


#ifdef GPU
kernel
#endif
	void actualizar_modelo(Uchar_ptr input,Uchar_ptr samples,Uchar_ptr Update, int width, int height,
	int numIteracion,Uint_ptr randomValues)
{

	int index = get_global_id(0);
	int x = index % width;
	int y = index / width;

	long big= width*height;
	int X_off[9] = {-1, 0, 1, -1, 1, -1, 0, 1, 0};
	int Y_off[9] = {-1, -1, -1, 1, 1, 1, 0, 0, 0};

	if (!((x>=0) && (y>=0) && (x<=width-1) && (y<=height-1)) ) return;

	//Si fue considerado background
	if (Update[y*width + x] == 0)
	{

		if (randomValues[numIteracion*index*47 % 2048] % randSubSample == 0)
		{
			int randSubS = (randomValues[(numIteracion*index*7) %randomSize]) %N;//...

			long base = big * randSubS;
			base+= y * width;
			base+= x;

			samples[base] = input[y * width + x ];
		}

		//Propagar si esta fuera del borde
		if ((x>0) && (y>0) && (x<width-1) && (y<height-1))
		{
			//Propagar a algun vecino!
			int random= (randomValues[(numIteracion*index*11) %randomSize]) % randSubSample;

			if (random==0)
			{
				//Random de los N samples
				int randSubP = (numIteracion+1) %N;
				long baseP = big * randSubP;

				//Random {X,Y}
				int Yr = y+Y_off[(randomValues[(numIteracion*index*15) %randomSize])%9];
				int Xr = x+X_off[(randomValues[(numIteracion*index*17) %randomSize])%9];

				if ((Xr>=0) && (Yr>=0) && (Xr<width) && (Yr<height))
				{
					baseP+= Yr * width;
					baseP+= Xr ;

					samples[baseP] = input[Yr * width  + Xr ];
				}
			}
		}


	}


}

#define MIN(a,b) a<b?a:b
#define MAX(a,b) a<b?b:a

	
#ifdef GPU
kernel
#endif
void mooreNeighborTracing(Uchar_ptr image,Uchar_ptr maskImage,Uchar_ptr paddedImage, Uchar_ptr paddedMask, int width, int height, Box_Ptr boxes, int numBoxes, int thIndex, int numBlocks)
{
	
	if (thIndex<0) thIndex = get_global_id(0) ; 

	bool inside = false;
	int pos = 0;
	int index = thIndex*16;
	int jmp = 0;

	int checkLocationNr = 1;  // The neighbor number of the location we want to check for a new border point
	int checkPosition;      // The corresponding absolute array address of checkLocationNr
	int newCheckLocationNr;   // Variable that holds the neighborhood position we want to check if we find a new border at checkLocationNr
	int startPos = pos;      // Set start position
	int counter = 0;       // Counter is used for the jacobi stop criterion
	int counter2 = 0;       // Counter2 is used to determine if the point we have discovered is one single point

	// Defines the neighborhood offset position from current position and the neighborhood
	// position we want to check next if we find a new border at checkLocationNr
	int neighborhood[16] = {            -1,7,             -3-width,7,
	-width-2,1,             -1-width,1,            1,3,             3+width,3,
	width+2,5,             1+width,5           };


	int iX = thIndex % numBlocks;
	int iY = thIndex / numBlocks;

	int startX = iX*(width+2) / numBlocks;
	int startY = iY*(height+2) / numBlocks;
	int endX = (iX+1)*(width+2) / numBlocks;
	int endY = (iY+1)*(height+2) / numBlocks;

	for(int y = startY; y < (MIN(height,endY)); y ++)
	{
		for(int x = startX; x < (MIN(width,endX)); x ++)
		{
			pos = x + y*(width+2);
			uchar val = paddedImage[pos];

			// Scan for BLACK pixel
			if(paddedMask[pos] == BLACK && !inside)    // Entering an already discovered border
			{
				inside = true;
			}
			else if(val == BLACK && inside)  // Already discovered border point
			{
				continue;
			}
			else if(val == WHITE && inside)  // Leaving a border
			{
				inside = false;
			}
			else if(val == BLACK && !inside)  // Undiscovered border point
			{
				paddedMask[pos] = BLACK;
				checkLocationNr = 1;  
				counter = 0;       
				counter2 = 0;      
				startPos = pos;
				// Trace around the neighborhood
				while(true)
				{
					checkPosition = pos + neighborhood[(checkLocationNr-1)*2+0];
					newCheckLocationNr = neighborhood[(checkLocationNr-1)*2+1];

					if (jmp==0)
					{
						if (index>=16*numBlocks*numBlocks) break;
						if (boxes[index].numPoints>=2048) break;
						int xi = pos % (width+2), yi = pos / (width+2);
						boxes[index].points[boxes[index].numPoints] = yi*width + xi;
						boxes[index].minX = MIN(boxes[index].minX, xi);
						boxes[index].maxX = MAX(boxes[index].maxX, xi);
						boxes[index].minY = MIN(boxes[index].minY, yi);
						boxes[index].maxY = MAX(boxes[index].maxY, yi);

						boxes[index].numPoints++;
					}
					jmp = (jmp+1) % 8;

					if(paddedImage[checkPosition] == BLACK) // Next border point found
					{
						if(checkPosition == startPos)
						{
							counter ++;

							// Stopping criterion (jacob)
							if(newCheckLocationNr == 1 || counter >= 3)
							{
								// Close loop
								inside = true; // Since we are starting the search at were we first started we must set inside to true
								break;
							}
						}

						checkLocationNr = newCheckLocationNr; // Update which neighborhood position we should check next
						pos = checkPosition;
						counter2 = 0;             // Reset the counter that keeps track of how many neighbors we have visited
						paddedMask[checkPosition] = BLACK; // Set the border pixel
					}
					else
					{
						// Rotate clockwise in the neighborhood
						checkLocationNr = 1 + (checkLocationNr % 8);
						if(counter2 > 8)
						{
							// If counter2 is above 8 we have traced around the neighborhood and
							// therefor the border is a single black pixel and we can exit
							counter2 = 0;
							break;
						}
						else
						{
							counter2 ++;
						}
					}
				}

				// Guardo los puntos, solo si tengo muchos
				if ((boxes[index].numPoints>50) && (boxes[index].numPoints<MAX_NUM_POINTS))
				{
					index++ ;
					jmp = 0;
					if (index>=numBoxes) return ;
					boxes[index].numPoints = 0;
					boxes[index].minX = boxes[index].minY = 100000;
					boxes[index].maxX = boxes[index].maxY = -100000;

				}
				else
				{
					boxes[index].numPoints = 0;
					boxes[index].minX = boxes[index].minY = 100000;
					boxes[index].maxX = boxes[index].maxY = -100000;
				}
			}
		}
	}


	return ;
}