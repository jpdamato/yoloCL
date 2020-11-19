#pragma once
#include <cstdio>
#include <chrono>
#include <vector>

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS // to disable deprecation warnings
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS // to disable deprecation warnings

// Includes the C++ OpenCL API. If not yet available, it can be found here:
// https://www.khronos.org/registry/cl/api/1.1/cl.hpp
#include <CL\cl.hpp>


// Select a kernel
#define KERNEL 8

// Constants for kernels 1 -- 5
#define TS 32                        // The square-root of the 2D tile-size (== work-group dims)

// Constants for kernels 3, 5
#define WPT 8                        // The amount of work-per-thread, i.e. the thread-coarsening factor
#define RTS (TS/WPT)                 // The reduced tile-size in one dimension

// Constants for kernels 4, 7 -- 10
#define WIDTH 4                      // The vector-width (in number of floats)

// Constants for kernel 5
#define TSDK 16                      // The tile-size in dimension K (for kernel 5 only)
#define LPT ((TSDK*WPT)/(TS))        // The amount of loads-per-thread (assume TSN==TSM)

// Constants for kernels 6 -- 10
#define TSM 128                      // The tile-size in dimension M
#define TSN 128                      // The tile-size in dimension N
#define TSK 16                       // The tile-size in dimension K
#define WPTM 8                       // The amount of work-per-thread in dimension M
#define WPTN 8                       // The amount of work-per-thread in dimension N
#define RTSM (TSM/WPTM)              // The reduced tile-size in dimension M (== number of threads)
#define RTSN (TSN/WPTN)              // The reduced tile-size in dimension N (== number of threads)
#define LPTA ((TSK*WPTM*WPTN)/(TSN)) // The amount of loads-per-thread for A
#define LPTB ((TSK*WPTM*WPTN)/(TSM)) // The amount of loads-per-thread for B

// Constraints on settings for kernels 6 -- 10
// Note: TSM/WPTM has to be integer
// Note: TSN/WPTN has to be integer
// Note: TSM/WIDTH has to be integer
// Note: TSN/WIDTH has to be integer
// Note: (TSK*WPTM*WPTN)/(TSN*WIDTH) has to be integer
// Note: (TSK*WPTM*WPTN)/(TSM*WIDTH) has to be integer

// Constants for kernel 11 (mimicing clBlas)
#define THREADSX 8
#define THREADSY 8
#define RX 8
#define RY 4
#define RK (RY)

// Constants for the supporting transpose kernel
#define TRANSPOSEX 32
#define TRANSPOSEY 32

// Constants for the supporting padding kernels
#define PADDINGX 32
#define PADDINGY 32

// Macros for host and kernel code
#define MIN(a,b) ((a) > (b)) ? (b) : (a)
#define MAX(a,b) ((a) > (b)) ? (a) : (b)
#define CEIL_DIV(x,y) (((x) + (y) - 1) / (y))
#define MOD2(x,y) ((x) % (y))
#define DIV2(x,y) ((x) / (y))

namespace liteClBlast {
	// =================================================================================================

	// Status codes. These codes can be returned by functions declared in this header file. The error
	// codes match either the standard OpenCL error codes or the clBLAS error codes. 
	enum class StatusCode {

		// Status codes in common with the OpenCL standard
		kSuccess = 0, // CL_SUCCESS
		kOpenCLCompilerNotAvailable = -3, // CL_COMPILER_NOT_AVAILABLE
		kTempBufferAllocFailure = -4, // CL_MEM_OBJECT_ALLOCATION_FAILURE
		kOpenCLOutOfResources = -5, // CL_OUT_OF_RESOURCES
		kOpenCLOutOfHostMemory = -6, // CL_OUT_OF_HOST_MEMORY
		kOpenCLBuildProgramFailure = -11, // CL_BUILD_PROGRAM_FAILURE: OpenCL compilation error
		kInvalidValue = -30, // CL_INVALID_VALUE
		kInvalidCommandQueue = -36, // CL_INVALID_COMMAND_QUEUE
		kInvalidMemObject = -38, // CL_INVALID_MEM_OBJECT
		kInvalidBinary = -42, // CL_INVALID_BINARY
		kInvalidBuildOptions = -43, // CL_INVALID_BUILD_OPTIONS
		kInvalidProgram = -44, // CL_INVALID_PROGRAM
		kInvalidProgramExecutable = -45, // CL_INVALID_PROGRAM_EXECUTABLE
		kInvalidKernelName = -46, // CL_INVALID_KERNEL_NAME
		kInvalidKernelDefinition = -47, // CL_INVALID_KERNEL_DEFINITION
		kInvalidKernel = -48, // CL_INVALID_KERNEL
		kInvalidArgIndex = -49, // CL_INVALID_ARG_INDEX
		kInvalidArgValue = -50, // CL_INVALID_ARG_VALUE
		kInvalidArgSize = -51, // CL_INVALID_ARG_SIZE
		kInvalidKernelArgs = -52, // CL_INVALID_KERNEL_ARGS
		kInvalidLocalNumDimensions = -53, // CL_INVALID_WORK_DIMENSION: Too many thread dimensions
		kInvalidLocalThreadsTotal = -54, // CL_INVALID_WORK_GROUP_SIZE: Too many threads in total
		kInvalidLocalThreadsDim = -55, // CL_INVALID_WORK_ITEM_SIZE: ... or for a specific dimension
		kInvalidGlobalOffset = -56, // CL_INVALID_GLOBAL_OFFSET
		kInvalidEventWaitList = -57, // CL_INVALID_EVENT_WAIT_LIST
		kInvalidEvent = -58, // CL_INVALID_EVENT
		kInvalidOperation = -59, // CL_INVALID_OPERATION
		kInvalidBufferSize = -61, // CL_INVALID_BUFFER_SIZE
		kInvalidGlobalWorkSize = -63, // CL_INVALID_GLOBAL_WORK_SIZE

		// Status codes in common with the clBLAS library
		kNotImplemented = -1024, // Routine or functionality not implemented yet
		kInvalidMatrixA = -1022, // Matrix A is not a valid OpenCL buffer
		kInvalidMatrixB = -1021, // Matrix B is not a valid OpenCL buffer
		kInvalidMatrixC = -1020, // Matrix C is not a valid OpenCL buffer
		kInvalidVectorX = -1019, // Vector X is not a valid OpenCL buffer
		kInvalidVectorY = -1018, // Vector Y is not a valid OpenCL buffer
		kInvalidDimension = -1017, // Dimensions M, N, and K have to be larger than zero
		kInvalidLeadDimA = -1016, // LD of A is smaller than the matrix's first dimension
		kInvalidLeadDimB = -1015, // LD of B is smaller than the matrix's first dimension
		kInvalidLeadDimC = -1014, // LD of C is smaller than the matrix's first dimension
		kInvalidIncrementX = -1013, // Increment of vector X cannot be zero
		kInvalidIncrementY = -1012, // Increment of vector Y cannot be zero
		kInsufficientMemoryA = -1011, // Matrix A's OpenCL buffer is too small
		kInsufficientMemoryB = -1010, // Matrix B's OpenCL buffer is too small
		kInsufficientMemoryC = -1009, // Matrix C's OpenCL buffer is too small
		kInsufficientMemoryX = -1008, // Vector X's OpenCL buffer is too small
		kInsufficientMemoryY = -1007, // Vector Y's OpenCL buffer is too small

		// Custom additional status codes for CLBlast
		kInsufficientMemoryTemp = -2050, // Temporary buffer provided to GEMM routine is too small
		kInvalidBatchCount = -2049, // The batch count needs to be positive
		kInvalidOverrideKernel = -2048, // Trying to override parameters for an invalid kernel
		kMissingOverrideParameter = -2047, // Missing override parameter(s) for the target kernel
		kInvalidLocalMemUsage = -2046, // Not enough local memory available on this device
		kNoHalfPrecision = -2045, // Half precision (16-bits) not supported by the device
		kNoDoublePrecision = -2044, // Double precision (64-bits) not supported by the device
		kInvalidVectorScalar = -2043, // The unit-sized vector is not a valid OpenCL buffer
		kInsufficientMemoryScalar = -2042, // The unit-sized vector's OpenCL buffer is too small
		kDatabaseError = -2041, // Entry for the device was not found in the database
		kUnknownError = -2040, // A catch-all error code representing an unspecified error
		kUnexpectedError = -2039, // A catch-all error code representing an unexpected exception
	};

	// Matrix layout and transpose types
	enum class Layout { kRowMajor = 101, kColMajor = 102 };
	enum class Transpose { kNo = 111, kYes = 112, kConjugate = 113 };
	enum class Triangle { kUpper = 121, kLower = 122 };
	enum class Diagonal { kNonUnit = 131, kUnit = 132 };
	enum class Side { kLeft = 141, kRight = 142 };
	enum class KernelMode { kCrossCorrelation = 151, kConvolution = 152 };

	// Precision scoped enum (values in bits)
	enum class Precision {
		kHalf = 16, kSingle = 32, kDouble = 64,
		kComplexSingle = 3232, kComplexDouble = 6464, kAny = -1
	};

	class LiteClBlast
	{
	public :
		cl::Program blastProgram;
		cl::Context context;
		cl::Device dev;
		int init(std::string cl_prog, int localWG, int platformProcessingIndex, int deviceProcessingIndex);
		int init(std::string cl_prog, int localWG, cl::Context ctx, cl::Device dev);

		int GemmF(const size_t m, const size_t n, const size_t k,
			const float alpha,
			const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
			const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
			const float beta,
			const cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
			const bool a_do_transpose, const bool b_do_transpose, const bool c_do_transpose,
			const bool a_conjugate, const bool b_conjugate, cl::CommandQueue cqueue, const bool checkIsZero);
	};

	class LiteclBlastv2
	{
	public : 
		cl_mem bufB_TR, bufA_XL, bufB_TR_XL, bufC_XL ;
		cl::Buffer bufAssignment_XL;
		cl_kernel kernel1 , kTranspose, kernel3a, kernel3b, kernel3c;
		std::vector<int> host_Assign;
		cl::Program blastProgram;
		cl::Context context;
		cl::Device dev;

		int init(std::string cl_prog,  cl::Context ctx, cl::Device dev);
		void initBuffers(cl::Program prog, cl::Context ctx, int K, int M, int N);
		void myclblas(cl::Buffer bA, cl::Buffer bB, cl::Buffer bC, int K, int M, int N, cl::Program prog, cl::Context ctx, cl::CommandQueue cqueue);
		void checkError(cl_int error, int line);

		void release();
	};
}