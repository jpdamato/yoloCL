#ifndef CONTOUR_DETECTION_H
#define CONTOUR_DETECTION_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;


#include <cl/cl.hpp>

#include "../gpuUtils/cl_utils.h"


 #include "cl_kernels.h"



static int indexes[16] = {-1,0, -1,1, 0,1, 1,1, 1,0, 1,-1, 0,-1, -1,-1};
static int rotations[24] = {-1,1, 0,1, 1, 1 , // Top
	1,1 , 1,0, 1,-1, // Right
	1,-1, 0,-1, -1,-1,// Bottom
	-1,-1, -1,0, -1,1 };// Left};





class ContourDetection
{
public:

	double cntTime, fillTime, otherTime ;

	cl::Kernel clkInitData,clkClose, clkMoore;
    cl::Buffer clMask,clCopy, clBoxes, clPadImage, clPadMask;

	vector<vector<Point>> contours;


	static void addPoint(boxCL *b,int x, int y)
	{
		if (b->minX > x) b->minX = x;
		if (b->minY > y) b->minY = y;
		if (b->maxX < x) b->maxX = x;
		if (b->maxY < y) b->maxY = y;
	}

	static bool isInside(boxCL *b,int x, int y)
	{
		return (b->minX<x) && (b->maxX>x)  && (b->minY<y)  && (b->maxY>y) ;
	}

	static bool outOfBorder(vector<Point> contour,int height,int width)
	{
		//Chequear si el blob toca el borde
		for (unsigned int i=0;i<contour.size();i++){
			if ((contour[i].x==1) || (contour[i].x==width-1) || (contour[i].y==1) || (contour[i].y==height-1))
				return false;
			if ((contour[i].x==0) || (contour[i].x==width) || (contour[i].y==0) || (contour[i].y==height))
				return false;
		}
		return true;
	}

	static void matToUchar(Mat m, uchar* uc, int width, int height)
	{	
#pragma omp parallel for
		for (int y=0; y<height;y++)
		{
			for (int x = 0 ; x<width ; x++)
			{
				uc[(long)y * (long)width + x] = m.at<uchar>(y,x) ;

			}
		}
	}

	static void ucharToMat(uchar* uc, Mat m, int width, int height)
	{
#pragma omp parallel for
		for (int y=0; y<height;y++)
		{
			for (int x = 0 ; x<width ; x++)
			{
				m.at<uchar>(y,x) = uc[(long)y * (long)width + x] ;
			}
		}
	}

	/**
	* This algorithm is called Moore Neighbor Tracing
	* An explanation of the algorithm can be found here:
	* http://www.imageprocessingplace.com/downloads_V3/root_downloads/tutorials/contour_tracing_Abeer_George_Ghuneim/moore.html
	*/


	/**
	* Pads an image represented by a 1D pixel array with 1 pixel with a color
	* specified by paddingColor
	*/
	void unpadImage(uchar* dstImage, int width, int height)
	{
	
#pragma omp parallel for
		for(int x = 0; x < width; x ++)
		{
			for(int y = 0; y < height; y ++)
			{
				dstImage[x+y*width] = 255 - paddedMask[x+1+(y+1)*(width+2)];
			}
		}
	}

	void padImage(uchar * image, int width, int height, uchar paddingColor)
	{
#pragma omp parallel for
		for(int x = 0; x < width+2; x ++)
		{
			for(int y = 0; y < height+2; y ++)
			{
				if(x == 0 || y == 0 || x == width+1 || y == height+1)
				{
					paddedImage[x + y*(width+2)] = paddingColor;
				}
				else
				{
					paddedImage[x+y*(width+2)] = image[x-1 + (y-1)*width];
				}
			}
		}
	}



	void bhm_line(uchar* pixels, int x1,int y1,int x2,int y2,int c, int width, int height, vector<Point>& contour)
	{
		int x,y,dx,dy,dx1,dy1,px,py,xe,ye,i;
		dx=x2-x1;
		dy=y2-y1;
		dx1=abs((int)dx);
		dy1=abs((int)dy);
		px=2*dy1-dx1;
		py=2*dx1-dy1;
		if(dy1<=dx1)
		{
			if(dx>=0)
			{
				x=x1;
				y=y1;
				xe=x2;
			}
			else
			{
				x=x2;
				y=y2;
				xe=x1;
			}
			pixels[y*width +x] = c; contour.push_back(Point(x, y));
			
			for(i=0;x<xe;i++)
			{
				x=x+1;
				if(px<0)
				{
					px=px+2*dy1;
				}
				else
				{
					if((dx<0 && dy<0) || (dx>0 && dy>0))
					{
						y=y+1;
					}
					else
					{
						y=y-1;
					}
					px=px+2*(dy1-dx1);
				}
				pixels[y*width +x] = c; contour.push_back(Point(x, y));
 			}
		}
		else
		{
			if(dy>=0)
			{
				x=x1;
				y=y1;
				ye=y2;
			}
			else
			{
				x=x2;
				y=y2;
				ye=y1;
			}
			pixels[y*width +x] = c; contour.push_back(Point(x, y));
			for(i=0;y<ye;i++)
			{
				y=y+1;
				if(py<=0)
				{
					py=py+2*dx1;
				}
				else
				{
					if((dx<0 && dy<0) || (dx>0 && dy>0))
					{
						x=x+1;
					}
					else
					{
						x=x-1;
					}
					py=py+2*(dx1-dy1);
				}
				pixels[y*width +x] = c; contour.push_back(Point(x, y));
			}
		}
	}

	static void fillPolygons(uchar* bf, uchar* dst, boxCL* boxes, int width, int height, int np)
	{
		#pragma omp parallel for 
		// para todos los blobs
		for (int i = 0; i<np ; i++)
		{
			if (boxes[i].numPoints==0) continue;

			// Es un borde 
			if ((boxes[i].minY <= 1 ) && ( boxes[i].maxY>=height)) continue;
			// dentro del bounding box
			for (int ry= boxes[i].minY; ry<=boxes[i].maxY ; ry++)				
			{
				//----------
				bool isInside = false;
				int rmnx , rmxx ;
				rmnx = 10000;
				rmxx = -10000;

				for (int rx = boxes[i].minX; rx <= boxes[i].maxX; rx++)
				{
					// es el contorno que le corresponde
					if  (bf[ry*width + rx]==boxes[i].blobID)
					{
						rmnx = MIN(rmnx, rx);
						rmxx = MAX(rmxx, rx);
					}
				}

				for (int rx = rmnx; rx <= rmxx; rx++)
					dst[ry*width + rx] = 255;
			}
		}

	}

	

	
	uchar *maskP , *ucopy, *paddedImage, *paddedMask, *bFP;
	boxCL* boxes;
	int numBoxes;

	int debug ;

	ContourDetection(int width, int height)
	{
		maskP = NULL; 
		ucopy= NULL;
		numBoxes = 256*32;
		boxes = new boxCL[256*32];
		cntTime =  fillTime  = otherTime = 0;

		paddedImage = new uchar[ (height+2) * (width+2)];
		ucopy = new uchar[width * height];
		paddedMask = new uchar[(width+2) * (height+2)];
		maskP = new uchar[width * height];
		bFP = new uchar[width * height * 4];

	}

	~ContourDetection()
	{
		free(maskP);
		free(ucopy);
		free(boxes);
		free(bFP);
	}

	void fillContours(Mat &m,  int minAreaBlob, int mode, int numFrame)
	{
		matToUchar(m, bFP, m.cols, m.rows);
		fillContours(bFP, m.rows, m.cols, minAreaBlob,mode, numFrame);
	}

	void findContour(uchar* bfP, int height,int width, int numFrame)
	{

		double st = cl_utils::sft_clock();
		//---------		
		#pragma omp parallel for 
		for (int i=0;i<width * height;i++) 
			initContourData((Uchar_ptr)maskP,(Uchar_ptr)ucopy,(Uchar_ptr)bfP,paddedImage, paddedMask, boxes, width, height, i,numBoxes);
		 
		#pragma omp parallel for 
		for (int i=0;i<width * height;i++) 
			closing(ucopy,bfP, width, height,1,i);

		otherTime += cl_utils::sft_clock() - st; 
		st = cl_utils::sft_clock();
		int blockSize = 16;
		//---------
		#pragma omp parallel for 
		for (int i = 0; i < blockSize*blockSize; i++)
		{
			mooreNeighborTracing(bfP, maskP, paddedImage, paddedMask, width, height, boxes, numBoxes, i, blockSize);
		}
		int blobCount = 256*16;
        
		// Remove white padding and return it
		unpadImage(maskP, width, height);
		
		cntTime += cl_utils::sft_clock() - st; 
		

	

		
		st = cl_utils::sft_clock();
		//
		// Queda guardado en el maskImage
#pragma omp parallel for 
		for (int x=0; x<width*height ; x++) { 		maskP[x] = 0; 	}

		// Remove white padding and return it
		int blobID = 1;
		for (int j=0; j<blobCount;j++)
		   if ( boxes[j].numPoints-1>0) 
			{
			    boxes[j].blobID = blobID;
				blobID++;
		   }

		debug = 1;
		cv::Mat cntMask(height, width, CV_8SC3, cvScalar(0));
		cv::Mat cntMask2(height, width, CV_8SC3, cvScalar(0));
		if (debug)
		{
			ucharToMat(maskP, cntMask, width, height);
			imshow("Masked", cntMask);
		}

		contours.clear();		
		contours.reserve(blobCount);

#pragma omp parallel for 
		for (int j=0; j<blobCount;j++)
		{	
			vector<Point> row;
			contours.push_back(row);

			for(int i = 0; i < boxes[j].numPoints-1; i ++)
			{
				int x0 = boxes[j].points[i] % width;
				int y0 = boxes[j].points[i] / width;

				int x1 = boxes[j].points[i+1] % width;
				int y1 = boxes[j].points[i+1] / width;

				if ((x0<0)|| (x0>=width) ||(y0<0) ||(y0>=height)) continue;
				if ((x1<0)|| (x1>=width) ||(y1<0) ||(y1>=height)) continue;
				row.push_back(Point(x0,y0));
				// draw contours
			//	bhm_line(maskP, x0,y0,x1,y1,boxes[j].blobID,width, height, row);

			}
			cv::polylines(cntMask2, row, false, cv::Scalar(j*71 % 255, (j+331) * 51 % 255, (j + 123) * 33 % 255), 3);
		}
	    //fillPolygons(maskP,maskP,  boxes, width, height, blobCount);

		fillTime += cl_utils::sft_clock() - st; 
		/*
		if (debug)
		{
			for (int j = 0; j < blobCount; j++)
			{
				//if (boxes[j].numPoints > 0)
				//	cv::rectangle(cntMask, cv::Point(boxes[j].minX, boxes[j].minY), cv::Point(boxes[j].maxX, boxes[j].maxY),
				//		Scalar(200, 202, 200));
				//cv::polylines(cntMask, contours[j], true, cv::Scalar(255,0,0), 2);
			}
			cv::imshow("contour extraction before", cntMask2);
		}
			//		matToUchar(cntMask, maskP,width, height);

			*/
		

		if (debug)
		{
			ucharToMat(maskP,cntMask,width, height);
		//	cv::imshow("contour extraction after",cntMask);
			cv::imshow("contour extraction before", cntMask2);
			cout<<" Time Contour : " <<otherTime/numFrame<<" " << fillTime/numFrame<< " " << cntTime/numFrame<< "\n";
			cv::waitKey(1);
		}
	}

	int initCL(cl::Context context, cl::Program program, cl::Buffer bf, int width, int height)
	{
		 // Data
      clMask=cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(uchar)*width * height);
      clCopy=cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(uchar)*width * height);
	  clPadImage =cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(uchar)*(height+2)*(width+2));
	  clPadMask =cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(uchar)*(height+2)*(width+2));
	  clBoxes=cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(boxCL)*numBoxes*16);
    // Kernels

      clkInitData=cl::Kernel(program,"initContourData");
      clkClose=cl::Kernel(program,"closing");
	  clkMoore=cl::Kernel(program,"mooreNeighborTracing");

	  
	  int rad = 1;
	  int thID = -1;
	  int numBlocks = 16;
	 //void initContourData(Uchar_ptr maskP,  //0
	  // Uchar_ptr ucopy,  //1
	  // Uchar_ptr bfP,   //2
	  // Uchar_ptr padImage, //3
	  // Uchar_ptr padMask, //4
	  // Box_Ptr boxes, //5
	  // int width, int height, int thID, int numBoxes)
	  int iclError = clkInitData.setArg(0,clMask);
      iclError |= clkInitData.setArg(1,clCopy);
      iclError |= clkInitData.setArg(2,bf);
	  iclError |= clkInitData.setArg(3,clPadImage);
	  iclError |= clkInitData.setArg(4,clPadMask);
	  iclError |= clkInitData.setArg(5,clBoxes);
      iclError |= clkInitData.setArg(6,sizeof(int),(void*)&width);
      iclError |= clkInitData.setArg(7,sizeof(int),(void*)&height);
	  iclError |= clkInitData.setArg(8,sizeof(int),(void*)&thID);
	  int nb = numBoxes * 16;
	  iclError |= clkInitData.setArg(9,sizeof(int),(void*)&nb);
	  
	  
	  iclError |= clkClose.setArg(0,clCopy);
      iclError |= clkClose.setArg(1,bf);      
      iclError |= clkClose.setArg(2,sizeof(int),(void*)&width);
      iclError |= clkClose.setArg(3,sizeof(int),(void*)&height);
	  iclError |= clkClose.setArg(4,sizeof(int),(void*)&rad);
	  iclError |= clkClose.setArg(5,sizeof(int),(void*)&thID);

//void mooreNeighborTracing(Uchar_ptr image,Uchar_ptr maskImage,Uchar_ptr paddedImage, Uchar_ptr paddedMask, int width, int height, Box_Ptr boxes, int numBoxes, int thIndex)
	  iclError |= clkMoore.setArg(0,bf);
      iclError |= clkMoore.setArg(1,clMask);      
	  iclError |= clkMoore.setArg(2,clPadImage);      
	  iclError |= clkMoore.setArg(3,clPadMask);      
      iclError |= clkMoore.setArg(4,sizeof(int),(void*)&width);
      iclError |= clkMoore.setArg(5,sizeof(int),(void*)&height);
	  iclError |= clkMoore.setArg(6,clBoxes);    
	  iclError |= clkMoore.setArg(7,sizeof(int),(void*)&numBoxes);
 	  iclError |= clkMoore.setArg(8,sizeof(int),(void*)&thID);  
	  iclError |= clkMoore.setArg(9,sizeof(int),(void*)&numBlocks);  

	  return iclError;
	}
	
	
	void fillContours(uchar* datos, int height,int width, int minAreaBlob, int mode, int numFrame)
	{

		if (mode == 1)
		{
			findContour(datos, height, width,numFrame);
		}
		else
		{
			double st = cl_utils::sft_clock();
			Mat_<uchar> bf(height,width);

			for (int y=0; y<height;y++)
			{
				for (int x = 0 ; x<width ; x++)
				{
					bf(y,x) = datos[y * width + x] ;
				}
			}

			vector<vector<Point> > contours;vector<Vec4i> hierarchy;

			cv::morphologyEx(bf,bf,MORPH_CLOSE,getStructuringElement(MORPH_ELLIPSE,Size(3,3)));
			otherTime += cl_utils::sft_clock() - st;
			st = cl_utils::sft_clock() ;

			findContours(bf,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE,Point(0, 0));

			//bf=0;
			//Filtro blobs con area menor a minAreaBlob (10)
			vector<vector<Point>>::iterator contours_it=contours.begin();
			while (contours_it != contours.end())
			{
				//Si no cumple con el área minima y está fuera del borde, eliminarlo
				if(contourArea(*contours_it)<minAreaBlob && outOfBorder(*contours_it,bf.rows,bf.cols)){
					// erase returns the new iterator
					contours_it = contours.erase(contours_it);
				}
				else{
					++contours_it;
				}
			}
			cntTime += cl_utils::sft_clock() - st;
			st = cl_utils::sft_clock() ;

			bf=0;
			//CV_FILLED NO IMPORTA QUE AREA

			drawContours(bf,contours,-1,Scalar(128,200,0),2,8);
			fillTime += cl_utils::sft_clock() - st;

			//OR DE LA IMAGEN!
			//cv::imshow("contours extraction",bf);cv::waitKey(1);

		}

	}
	
};
#endif