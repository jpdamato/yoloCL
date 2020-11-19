#include <CL\cl.hpp>

#include "liteCLBlast.h"



#include "../gpuUtils/cl_utils.h"
#include "../gpuUtils/gpu_param.h"
#include "../u_ProcessTime.h"

//Xgemm
std::vector<std::string> params_names = { "GEMMK", "KREG", "KWG", "KWI", "MDIMA", "MDIMC", "MWG", "NDIMB", "NDIMC", "NWG", "SA", "SB", "STRM", "STRN", "VWM", "VWN",
//XgemmDirectSingle
"KWID", "MDIMAD", "MDIMCD", "NDIMBD", "NDIMCD", "PADA", "PADB",  "VWMD", "VWND", "WGD" ,
	// xComm
	"KWID", "MDIMAD", "MDIMCD", "NDIMBD", "NDIMCD", "PADA", "PADB", "VWMD", "VWND", "WGD" };

std::vector<int> params_values = { 0, 1, 16, 2, 16, 16, 128, 16, 16, 64, 1, 1, 1, 1, 2, 4, 2, 16, 8, 8, 8, 1, 1, 1, 2, 32, 0, 0, 0, 0, 0, 0,
                                   1, 8, 16, 16, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 };

namespace liteClBlast
{

	

	int LiteClBlast::init(std::string cl_prog, int localWG, int platformProcessingIndex, int deviceProcessingIndex)
	{
		int error = CL_SUCCESS;
		blastProgram = clUtils::loadCLSources(cl_prog, localWG, platformProcessingIndex, deviceProcessingIndex, &error);

		return error;
	}

	int LiteClBlast::init(std::string cl_prog, int localWG, cl::Context ctx, cl::Device dev)
	{
		int error = CL_SUCCESS;
		blastProgram = clUtils::loadCLSources(cl_prog, ctx, dev, &error);
		this->context = ctx;
		this->dev = dev;
		return error;
	}


	// Rounding functions performing ceiling and division operations
	size_t CeilDiv(const size_t x, const size_t y) {
		return 1 + ((x - 1) / y);
	}

	size_t Ceil(const size_t x, const size_t y) {
		return CeilDiv(x, y)*y;
	}

	int LiteClBlast::GemmF(const size_t m, const size_t n, const size_t k,
		const float alpha,
		const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
		const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
		const float beta,
		const cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
		const bool a_do_transpose, const bool b_do_transpose, const bool c_do_transpose,
		const bool a_conjugate, const bool b_conjugate, cl::CommandQueue cqueue, const bool checkIsZero) {

		// Retrieves the proper XgemmDirect kernel from the compiled binary
		int error = 0;

		cl::Kernel kernel = cl::Kernel(blastProgram,"XgemmDirectTN", &error);

		if (error != CL_SUCCESS)
		{
			return error;
		}

		// fprintf(stderr, name.c_str());

		 // Sets the kernel arguments
		kernel.setArg(0, static_cast<int>(m));
		kernel.setArg(1, static_cast<int>(n));
		kernel.setArg(2, static_cast<int>(k));
		kernel.setArg(3, alpha);
		kernel.setArg(4, beta);
		kernel.setArg(5, a_buffer);
		kernel.setArg(6, static_cast<int>(a_offset));
		kernel.setArg(7, static_cast<int>(a_ld));
		kernel.setArg(8, b_buffer);
		kernel.setArg(9, static_cast<int>(b_offset));
		kernel.setArg(10, static_cast<int>(b_ld));
		kernel.setArg(11, c_buffer);
		kernel.setArg(12, static_cast<int>(c_offset));
		kernel.setArg(13, static_cast<int>(c_ld));
		kernel.setArg(14, static_cast<int>(c_do_transpose));
		kernel.setArg(15, static_cast<int>(a_conjugate));
		kernel.setArg(16, static_cast<int>(b_conjugate));
		kernel.setArg(17, static_cast<int>(checkIsZero));

		size_t  WGD = 32;
		size_t  MDIMCD = 8;
		size_t  NDIMCD = 8;

		// Computes the global and local thread sizes
		const auto m_ceiled = Ceil(m, WGD);
		const auto n_ceiled = Ceil(n, WGD);
		const cl::NDRange global = cl::NDRange{		(m_ceiled * MDIMCD) / WGD,			(n_ceiled * NDIMCD) / WGD	};
		const cl::NDRange  local = cl::NDRange{  MDIMCD, NDIMCD };

		// Launches the kernel
		error = cqueue.enqueueNDRangeKernel(kernel, 0, global, local);

		return error;
	}







	// Set the locations of the OpenCL kernel files
#define CL_KERNEL_FILE "D:\\Sdks\\Yolo\\darknet\\liteGemm.cl"


// OpenCL settings
#define MAX_NUM_DEVICES 16
#define MAX_DEVICE_NAME 1024
#define CURRENT_DEVICE 0

#define KERNEL 10

// Define OpenCL compiler options, such as "-cl-nv-maxrregcount=127"
#define COMPILER_OPTIONS ""

// =================================================================================================


	void LiteclBlastv2::initBuffers(cl::Program prog, cl::Context ctx, int K, int M, int N)
	{
		int err = 0;

		int K_XL, M_XL, N_XL;
		// In case of myGEMM10, compute matrix sizes K, M, N as rounded-up to form complete tiles
		K_XL = CEIL_DIV(K, TSK) * TSK;
		M_XL = CEIL_DIV(M, TSM) * TSM;
		N_XL = CEIL_DIV(N, TSN) * TSN;

		host_Assign = std::vector<int>(M_XL*N_XL);

		// Prepare OpenCL memory objects
		// Create extra objects for rounded-up sizes (only needed in case of myGEMM10)
		bufB_TR = clCreateBuffer(ctx(), CL_MEM_READ_ONLY, N*K * sizeof(float), NULL, &err);
		bufA_XL = clCreateBuffer(ctx(), CL_MEM_READ_ONLY, M_XL*K_XL * sizeof(float), NULL, &err);
		bufB_TR_XL = clCreateBuffer(ctx(), CL_MEM_READ_ONLY, N_XL*K_XL * sizeof(float), NULL, &err);
		bufC_XL = clCreateBuffer(ctx(), CL_MEM_READ_WRITE, M_XL*N_XL * sizeof(float), NULL, &err);

		bufAssignment_XL = cl::Buffer(ctx, CL_MEM_READ_WRITE, M_XL*N_XL * sizeof(int));

		checkError(err, __LINE__);

		// Configure the myGEMM kernel
		char kernelname[100];
		sprintf(kernelname, "myGEMM%d", KERNEL);
		kernel1 = clCreateKernel(prog(), kernelname, &err);
		kTranspose = clCreateKernel(prog(), "transpose", &err);
		kernel3a = clCreateKernel(prog(), "paddingAddZeroes", &err);
		kernel3b = clCreateKernel(prog(), "paddingAddZeroes", &err);
		kernel3c = clCreateKernel(prog(), "paddingRemoveZeroes", &err);

		checkError(err, __LINE__);

	}
	// Matrix-multiplication using a custom OpenCL SGEMM kernel. This function also copies the input
	// matrices to the GPU, runs SGEMM, and copies the output matrix back to the CPU.
	void LiteclBlastv2::myclblas(cl::Buffer bufA, cl::Buffer bufB, cl::Buffer bufC,
		int K, int M, int N,
		cl::Program prog, cl::Context ctx, cl::CommandQueue cqueue) {

		// Define OpenCL variables
		cl_int err;
		cl_context context = 0;
		cl_command_queue queue = 0;
		cl_event event = NULL;
		cl_program program = NULL;
		
		int K_XL, M_XL, N_XL;
		// In case of myGEMM10, compute matrix sizes K, M, N as rounded-up to form complete tiles
		K_XL = CEIL_DIV(K, TSK) * TSK;
		M_XL = CEIL_DIV(M, TSM) * TSM;
		N_XL = CEIL_DIV(N, TSN) * TSN;

		program = prog();
		context = ctx();
		queue = cqueue();

		
		// Set the arguments of the myGEMM kernel
		err = clSetKernelArg(kernel1, 0, sizeof(int), (void*)&M_XL);
		err = clSetKernelArg(kernel1, 1, sizeof(int), (void*)&N_XL);
		err = clSetKernelArg(kernel1, 2, sizeof(int), (void*)&K_XL);
		err = clSetKernelArg(kernel1, 3, sizeof(cl_mem), (void*)&bufA_XL);
		err = clSetKernelArg(kernel1, 4, sizeof(cl_mem), (void*)&bufB_TR_XL);
		err = clSetKernelArg(kernel1, 5, sizeof(cl_mem), (void*)&bufC_XL);
		// Configure the supporting transpose kernel and set its arguments (only for certain myGEMMs)

		
		const size_t tLocal[2] = { TRANSPOSEX, TRANSPOSEY };
		const size_t tGlobal[2] = { (size_t)K, (size_t)N };

		// Configure the supporting padding kernels and set their arguments (only for myGEMM10)

		err = clSetKernelArg(kernel3a, 0, sizeof(int), (void*)&M);
		err = clSetKernelArg(kernel3a, 1, sizeof(int), (void*)&K);
		err = clSetKernelArg(kernel3a, 2, sizeof(cl_mem), (void*)&bufA);
		err = clSetKernelArg(kernel3a, 3, sizeof(int), (void*)&M_XL);
		err = clSetKernelArg(kernel3a, 4, sizeof(int), (void*)&K_XL);
		err = clSetKernelArg(kernel3a, 5, sizeof(cl_mem), (void*)&bufA_XL);

		err = clSetKernelArg(kernel3b, 0, sizeof(int), (void*)&N);
		err = clSetKernelArg(kernel3b, 1, sizeof(int), (void*)&K);
		err = clSetKernelArg(kernel3b, 2, sizeof(cl_mem), (void*)&bufB_TR);
		err = clSetKernelArg(kernel3b, 3, sizeof(int), (void*)&N_XL);
		err = clSetKernelArg(kernel3b, 4, sizeof(int), (void*)&K_XL);
		err = clSetKernelArg(kernel3b, 5, sizeof(cl_mem), (void*)&bufB_TR_XL);

		err = clSetKernelArg(kernel3c, 0, sizeof(int), (void*)&M_XL);
		err = clSetKernelArg(kernel3c, 1, sizeof(int), (void*)&N_XL);
		err = clSetKernelArg(kernel3c, 2, sizeof(cl_mem), (void*)&bufC_XL);
		err = clSetKernelArg(kernel3c, 3, sizeof(int), (void*)&M);
		err = clSetKernelArg(kernel3c, 4, sizeof(int), (void*)&N);
		err = clSetKernelArg(kernel3c, 5, sizeof(cl_mem), (void*)&bufC);
		
	
		checkError(err, __LINE__);

		const size_t pLocal[2] = { PADDINGX, PADDINGY };
		const size_t pAGlobal[2] = { (size_t)M_XL, (size_t)K_XL };
		const size_t pBGlobal[2] = { (size_t)N_XL, (size_t)K_XL };
		const size_t pCGlobal[2] = { (size_t)M, (size_t)N };

		const size_t local[2] = { TSM / WPTM, TSN / WPTN };
		const size_t global[2] = { (size_t)(M_XL / WPTM), (size_t)(N_XL / WPTN) };

		int NUM_RUNS = 1;

		// Start the timed loop
		double startTime = 0;

		auto start_time = std::chrono::steady_clock::now();


		// Run the transpose B kernel first
		err = clSetKernelArg(kTranspose, 0, sizeof(int), (void*)&K);
		err = clSetKernelArg(kTranspose, 1, sizeof(int), (void*)&N);
		err = clSetKernelArg(kTranspose, 2, sizeof(cl_mem), (void*)&bufB);
		err = clSetKernelArg(kTranspose, 3, sizeof(cl_mem), (void*)&bufB_TR);
		err = clSetKernelArg(kTranspose, 4, sizeof(cl_mem), (void*)&bufAssignment_XL);

		err = clEnqueueNDRangeKernel(queue, kTranspose, 2, NULL, tGlobal, tLocal, 0, NULL, &event);
		
		// Make the inputs extra large with padded zeros
		err = clEnqueueNDRangeKernel(queue, kernel3a, 2, NULL, pAGlobal, pLocal, 0, NULL, &event);

		err = clEnqueueNDRangeKernel(queue, kernel3b, 2, NULL, pBGlobal, pLocal, 0, NULL, &event);
		cqueue.finish();

		auto elapsed_time = std::chrono::steady_clock::now() - start_time;
		auto time_ms = std::chrono::duration<double, std::milli>(elapsed_time).count();

		printf(" ...... Execution PADD   32F in %.3lf ms with status %d\n", time_ms, err);
		start_time = std::chrono::steady_clock::now();

		// Run the myGEMM kernel
		err = clEnqueueNDRangeKernel(queue, kernel1, 2, NULL, global, local, 0, NULL, &event);

		// Remove padded zeroes from the larger output
		err = clEnqueueNDRangeKernel(queue, kernel3c, 2, NULL, pCGlobal, pLocal, 0, NULL, &event);

		cqueue.finish();

		elapsed_time = std::chrono::steady_clock::now() - start_time;
		time_ms = std::chrono::duration<double, std::milli>(elapsed_time).count();

		// Example completed. See "clblast.h" for status codes (0 -> success).
		printf(" ...... Execution MULT  32F in %.3lf ms with status %d\n", time_ms, err);


		
		
		cqueue.enqueueReadBuffer(bufAssignment_XL, CL_TRUE, 0, host_Assign.size() * sizeof(int), host_Assign.data());
		cqueue.finish();

	}

	void LiteclBlastv2::release()
	{
		clReleaseMemObject(bufB_TR);
		clReleaseMemObject(bufA_XL);
		clReleaseMemObject(bufB_TR_XL);
		clReleaseMemObject(bufC_XL);

		clReleaseKernel(kernel1);
#if KERNEL == 5 || KERNEL == 6 || KERNEL == 7 || KERNEL == 8 || KERNEL == 9 || KERNEL == 10
		clReleaseKernel(kTranspose);
#endif
#if KERNEL == 10
		clReleaseKernel(kernel3a);
		clReleaseKernel(kernel3b);
		clReleaseKernel(kernel3c);
#endif
	}


	int LiteclBlastv2::init(std::string cl_prog , cl::Context ctx, cl::Device dev)
	{
		int error = CL_SUCCESS;
		blastProgram = clUtils::loadCLSources(cl_prog, ctx, dev, &error);
		this->context = ctx;
		this->dev = dev;
		return error;
		
	}
	// =================================================================================================

	// Print an error message to screen (only if it occurs)
	void LiteclBlastv2::checkError(cl_int error, int line) {
		if (error != CL_SUCCESS) {
			switch (error) {
			case CL_DEVICE_NOT_FOUND:                 printf("-- Error at %d:  Device not found.\n", line); break;
			case CL_DEVICE_NOT_AVAILABLE:             printf("-- Error at %d:  Device not available\n", line); break;
			case CL_COMPILER_NOT_AVAILABLE:           printf("-- Error at %d:  Compiler not available\n", line); break;
			case CL_MEM_OBJECT_ALLOCATION_FAILURE:    printf("-- Error at %d:  Memory object allocation failure\n", line); break;
			case CL_OUT_OF_RESOURCES:                 printf("-- Error at %d:  Out of resources\n", line); break;
			case CL_OUT_OF_HOST_MEMORY:               printf("-- Error at %d:  Out of host memory\n", line); break;
			case CL_PROFILING_INFO_NOT_AVAILABLE:     printf("-- Error at %d:  Profiling information not available\n", line); break;
			case CL_MEM_COPY_OVERLAP:                 printf("-- Error at %d:  Memory copy overlap\n", line); break;
			case CL_IMAGE_FORMAT_MISMATCH:            printf("-- Error at %d:  Image format mismatch\n", line); break;
			case CL_IMAGE_FORMAT_NOT_SUPPORTED:       printf("-- Error at %d:  Image format not supported\n", line); break;
			case CL_BUILD_PROGRAM_FAILURE:            printf("-- Error at %d:  Program build failure\n", line); break;
			case CL_MAP_FAILURE:                      printf("-- Error at %d:  Map failure\n", line); break;
			case CL_INVALID_VALUE:                    printf("-- Error at %d:  Invalid value\n", line); break;
			case CL_INVALID_DEVICE_TYPE:              printf("-- Error at %d:  Invalid device type\n", line); break;
			case CL_INVALID_PLATFORM:                 printf("-- Error at %d:  Invalid platform\n", line); break;
			case CL_INVALID_DEVICE:                   printf("-- Error at %d:  Invalid device\n", line); break;
			case CL_INVALID_CONTEXT:                  printf("-- Error at %d:  Invalid context\n", line); break;
			case CL_INVALID_QUEUE_PROPERTIES:         printf("-- Error at %d:  Invalid queue properties\n", line); break;
			case CL_INVALID_COMMAND_QUEUE:            printf("-- Error at %d:  Invalid command queue\n", line); break;
			case CL_INVALID_HOST_PTR:                 printf("-- Error at %d:  Invalid host pointer\n", line); break;
			case CL_INVALID_MEM_OBJECT:               printf("-- Error at %d:  Invalid memory object\n", line); break;
			case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  printf("-- Error at %d:  Invalid image format descriptor\n", line); break;
			case CL_INVALID_IMAGE_SIZE:               printf("-- Error at %d:  Invalid image size\n", line); break;
			case CL_INVALID_SAMPLER:                  printf("-- Error at %d:  Invalid sampler\n", line); break;
			case CL_INVALID_BINARY:                   printf("-- Error at %d:  Invalid binary\n", line); break;
			case CL_INVALID_BUILD_OPTIONS:            printf("-- Error at %d:  Invalid build options\n", line); break;
			case CL_INVALID_PROGRAM:                  printf("-- Error at %d:  Invalid program\n", line); break;
			case CL_INVALID_PROGRAM_EXECUTABLE:       printf("-- Error at %d:  Invalid program executable\n", line); break;
			case CL_INVALID_KERNEL_NAME:              printf("-- Error at %d:  Invalid kernel name\n", line); break;
			case CL_INVALID_KERNEL_DEFINITION:        printf("-- Error at %d:  Invalid kernel definition\n", line); break;
			case CL_INVALID_KERNEL:                   printf("-- Error at %d:  Invalid kernel\n", line); break;
			case CL_INVALID_ARG_INDEX:                printf("-- Error at %d:  Invalid argument index\n", line); break;
			case CL_INVALID_ARG_VALUE:                printf("-- Error at %d:  Invalid argument value\n", line); break;
			case CL_INVALID_ARG_SIZE:                 printf("-- Error at %d:  Invalid argument size\n", line); break;
			case CL_INVALID_KERNEL_ARGS:              printf("-- Error at %d:  Invalid kernel arguments\n", line); break;
			case CL_INVALID_WORK_DIMENSION:           printf("-- Error at %d:  Invalid work dimensionsension\n", line); break;
			case CL_INVALID_WORK_GROUP_SIZE:          printf("-- Error at %d:  Invalid work group size\n", line); break;
			case CL_INVALID_WORK_ITEM_SIZE:           printf("-- Error at %d:  Invalid work item size\n", line); break;
			case CL_INVALID_GLOBAL_OFFSET:            printf("-- Error at %d:  Invalid global offset\n", line); break;
			case CL_INVALID_EVENT_WAIT_LIST:          printf("-- Error at %d:  Invalid event wait list\n", line); break;
			case CL_INVALID_EVENT:                    printf("-- Error at %d:  Invalid event\n", line); break;
			case CL_INVALID_OPERATION:                printf("-- Error at %d:  Invalid operation\n", line); break;
			case CL_INVALID_GL_OBJECT:                printf("-- Error at %d:  Invalid OpenGL object\n", line); break;
			case CL_INVALID_BUFFER_SIZE:              printf("-- Error at %d:  Invalid buffer size\n", line); break;
			case CL_INVALID_MIP_LEVEL:                printf("-- Error at %d:  Invalid mip-map level\n", line); break;
			case -1024:                               printf("-- Error at %d:  *clBLAS* Functionality is not implemented\n", line); break;
			case -1023:                               printf("-- Error at %d:  *clBLAS* Library is not initialized yet\n", line); break;
			case -1022:                               printf("-- Error at %d:  *clBLAS* Matrix A is not a valid memory object\n", line); break;
			case -1021:                               printf("-- Error at %d:  *clBLAS* Matrix B is not a valid memory object\n", line); break;
			case -1020:                               printf("-- Error at %d:  *clBLAS* Matrix C is not a valid memory object\n", line); break;
			case -1019:                               printf("-- Error at %d:  *clBLAS* Vector X is not a valid memory object\n", line); break;
			case -1018:                               printf("-- Error at %d:  *clBLAS* Vector Y is not a valid memory object\n", line); break;
			case -1017:                               printf("-- Error at %d:  *clBLAS* An input dimension (M,N,K) is invalid\n", line); break;
			case -1016:                               printf("-- Error at %d:  *clBLAS* Leading dimension A must not be less than the size of the first dimension\n", line); break;
			case -1015:                               printf("-- Error at %d:  *clBLAS* Leading dimension B must not be less than the size of the second dimension\n", line); break;
			case -1014:                               printf("-- Error at %d:  *clBLAS* Leading dimension C must not be less than the size of the third dimension\n", line); break;
			case -1013:                               printf("-- Error at %d:  *clBLAS* The increment for a vector X must not be 0\n", line); break;
			case -1012:                               printf("-- Error at %d:  *clBLAS* The increment for a vector Y must not be 0\n", line); break;
			case -1011:                               printf("-- Error at %d:  *clBLAS* The memory object for Matrix A is too small\n", line); break;
			case -1010:                               printf("-- Error at %d:  *clBLAS* The memory object for Matrix B is too small\n", line); break;
			case -1009:                               printf("-- Error at %d:  *clBLAS* The memory object for Matrix C is too small\n", line); break;
			case -1008:                               printf("-- Error at %d:  *clBLAS* The memory object for Vector X is too small\n", line); break;
			case -1007:                               printf("-- Error at %d:  *clBLAS* The memory object for Vector Y is too small\n", line); break;
			case -1001:                               printf("-- Error at %d:  Code -1001: no GPU available?\n", line); break;
			default:                                  printf("-- Error at %d:  Unknown with code %d\n", line, error);
			}
			exit(1);
		}
	}

}