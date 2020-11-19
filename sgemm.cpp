
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file demonstrates the use of the SGEMM routine. It is a stand-alone example, but it does
// require the Khronos C++ OpenCL API header file (downloaded by CMake). The example uses C++
// features, but CLBlast can also be used using the regular C-style OpenCL API.
//
// Note that this example is meant for illustration purposes only. CLBlast provides other programs
// for performance benchmarking ('client_xxxxx') and for correctness testing ('test_xxxxx').
//
// =================================================================================================

#include <cstdio>
#include <chrono>
#include <vector>

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS // to disable deprecation warnings
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS // to disable deprecation warnings

// Includes the C++ OpenCL API. If not yet available, it can be found here:
// https://www.khronos.org/registry/cl/api/1.1/cl.hpp
#include <CL\cl.hpp>

// Includes the CLBlast library
#include <clblast.h>

#include "utilities\utilities.hpp"
#include "liteBlast/liteCLBlast.h"

using half = clblast::half;

// Example SGEMM arguments
const size_t m = 256	;// 32;
const size_t n = 23104;//92416;
const size_t k = 128;//64;
float alpha = 1.0f;
float beta = 1.0f;
const auto a_ld = k;
const auto b_ld = n;
const auto c_ld = n;

liteClBlast::LiteClBlast lclB;

//liteClBlast::LiteclBlastv2 liteB;

std::chrono::time_point<std::chrono::steady_clock> start_time;

// =================================================================================================
int testDouble(cl::Context context, cl::CommandQueue queue, std::vector<float>& res)
{
	auto event = cl_event{ nullptr };
	// Populate host matrices with some example data
	auto host_a = std::vector<double>(m*k);
	auto host_b = std::vector<double>(n*k);
	auto host_c = std::vector<double>(m*n);
	for (double &item : host_a) { item = 12.193; }
	for (double &item : host_b) { item = -8.199; }
	for (double &item : host_c) { item = 0.0; }

	// Start the timer
	start_time = std::chrono::steady_clock::now();

	// Copy the matrices to the device
	auto device_a = cl::Buffer(context, CL_MEM_READ_WRITE, host_a.size()*sizeof(double));
	auto device_b = cl::Buffer(context, CL_MEM_READ_WRITE, host_b.size()*sizeof(double));
	auto device_c = cl::Buffer(context, CL_MEM_READ_WRITE, host_c.size()*sizeof(double));
	queue.enqueueWriteBuffer(device_a, CL_TRUE, 0, host_a.size()*sizeof(double), host_a.data());
	queue.enqueueWriteBuffer(device_b, CL_TRUE, 0, host_b.size()*sizeof(double), host_b.data());
	queue.enqueueWriteBuffer(device_c, CL_TRUE, 0, host_c.size()*sizeof(double), host_c.data());


	// Call the SGEMM routine. Note that the type of alpha and beta (float) determine the precision.
	auto queue_plain = queue();
	auto status = clblast::Gemm<double>(clblast::Layout::kRowMajor,
		clblast::Transpose::kNo, clblast::Transpose::kNo,
		m, n, k,
		alpha,
		device_a(), 0, a_ld,
		device_b(), 0, b_ld,
		beta,
		device_c(), 0, c_ld,
		&queue_plain, &event);

	queue.enqueueReadBuffer(device_c, CL_TRUE, 0, host_c.size()*sizeof(double), host_c.data());

	for (double &item : host_c)
	{
		res.push_back(item);
	}

	// Record the execution time
	if (status == clblast::StatusCode::kSuccess) {
		clWaitForEvents(1, &event);
		clReleaseEvent(event);
	}

	return static_cast<int>(status);
}

// =================================================================================================
int test16F(cl::Context context, cl::CommandQueue queue, std::vector<float>& res)
{
	auto event = cl_event{ nullptr };
	// Populate host matrices with some example data
	auto host_a = std::vector<half>(m*k);
	auto host_b = std::vector<half>(n*k);
	auto host_c = std::vector<half>(m*n);
	for (half &item : host_a) { item = FloatToHalf(12.193f); }
	for (half &item : host_b) { item = FloatToHalf( -8.199f); }
	for (half &item : host_c) { item = FloatToHalf(0.0f); }

	// Start the timer
	start_time = std::chrono::steady_clock::now();
		
	// Copy the matrices to the device
	auto device_a = cl::Buffer(context, CL_MEM_READ_WRITE, host_a.size()*sizeof(half));
	auto device_b = cl::Buffer(context, CL_MEM_READ_WRITE, host_b.size()*sizeof(half));
	auto device_c = cl::Buffer(context, CL_MEM_READ_WRITE, host_c.size()*sizeof(half));
	queue.enqueueWriteBuffer(device_a, CL_TRUE, 0, host_a.size()*sizeof(half), host_a.data());
	queue.enqueueWriteBuffer(device_b, CL_TRUE, 0, host_b.size()*sizeof(half), host_b.data());
	queue.enqueueWriteBuffer(device_c, CL_TRUE, 0, host_c.size()*sizeof(half), host_c.data());


	// Call the SGEMM routine. Note that the type of alpha and beta (float) determine the precision.
	auto queue_plain = queue();
	auto status = clblast::Gemm<half>(clblast::Layout::kRowMajor,
		clblast::Transpose::kNo, clblast::Transpose::kNo,
		m, n, k,
		alpha + 0.0001,
		device_a(), 0, a_ld,
		device_b(), 0, b_ld,
		beta + 0.0001,
		device_c(), 0, c_ld,
		&queue_plain, &event);

	queue.enqueueReadBuffer(device_c, CL_TRUE, 0, host_c.size()*sizeof(half), host_c.data());

	for (half &item : host_c) 
	{ 
		res.push_back(HalfToFloat(item));
	}

	// Record the execution time
	if (status == clblast::StatusCode::kSuccess) {
		clWaitForEvents(1, &event);
		clReleaseEvent(event);
	}

	return static_cast<int>(status);
}

int get_local_id(int dim)
{
	return 1;
}

int get_group_id(int dim)
{
	return 1;
}


void gemm_nn(int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float *C, int ldc)
{
	int i, j, k;
#pragma omp parallel for
	for (i = 0; i < M; ++i)
	{
		for (k = 0; k < K; ++k)
		{
			register float A_PART = ALPHA * A[i*lda + k];
			for (j = 0; j < N; ++j)
			{
				C[i*ldc + j] += A_PART * B[k*ldb + j];
			}
		}
	}
}
// =================================================================================================
int test32F(cl::Context context, cl::CommandQueue queue, std::vector<float>& res)
{
	auto event = cl_event{ nullptr };
	// Populate host matrices with some example data
	auto host_a = std::vector<float>(m*k);
	auto host_b = std::vector<float>(n*k);
	auto host_c0 = std::vector<float>(m*n);
	auto host_c1 = std::vector<float>(m*n);
	auto host_c2 = std::vector<float>(m*n);
	int index = 0;
	// initialize
	for (int x = 0; x < m; x++) 
		for (int y = 0; y < k; y++)
		{
			if (x < 32 && y < 32)
			{
				host_a[x * k + y] = 11.0f;
			}
		} // 8.73f + 3.0f * (1.0 - rand() / RAND_MAX); }
  
	// initialize
	for (int x = 0; x < n; x++)
		for (int y = 0; y < k; y++)
		{
			host_b[x * k + y] = -1.0f;
		}

	for (auto &item : host_c0) { item = 0.0f; }
	for (auto &item : host_c1) { item = 0.0f; }
	for (auto &item : host_c2) { item = 0.0f; }

	// Start the timer
	start_time = std::chrono::steady_clock::now();

	// Copy the matrices to the device
	auto device_a = cl::Buffer(context, CL_MEM_READ_WRITE, host_a.size() * sizeof(float));
	auto device_b = cl::Buffer(context, CL_MEM_READ_WRITE, host_b.size() * sizeof(float));
	auto device_c0 = cl::Buffer(context, CL_MEM_READ_WRITE, host_c1.size() * sizeof(float));
	auto device_c1 = cl::Buffer(context, CL_MEM_READ_WRITE, host_c1.size() * sizeof(float));
	auto device_c2 = cl::Buffer(context, CL_MEM_READ_WRITE, host_c2.size() * sizeof(float));

	queue.enqueueWriteBuffer(device_a, CL_TRUE, 0, host_a.size() * sizeof(float), host_a.data());
	queue.enqueueWriteBuffer(device_b, CL_TRUE, 0, host_b.size() * sizeof(float), host_b.data());
	queue.enqueueWriteBuffer(device_c0, CL_TRUE, 0, host_c0.size() * sizeof(float), host_c0.data());
	queue.enqueueWriteBuffer(device_c1, CL_TRUE, 0, host_c1.size() * sizeof(float), host_c1.data());
	queue.enqueueWriteBuffer(device_c2, CL_TRUE, 0, host_c2.size() * sizeof(float), host_c2.data());


	// My CL BLAS .. kernel 10.. Method 3. not working
	/*
	liteB.initBuffers(liteB.blastProgram, context, k, m, n);
	

	{
		auto start_time = std::chrono::steady_clock::now();

		liteB.myclblas(device_a, device_b, device_c2, k, m, n, lclB.blastProgram, context, queue);
		
	
		auto elapsed_time = std::chrono::steady_clock::now() - start_time;
		auto time_ms = std::chrono::duration<double, std::milli>(elapsed_time).count();

		// Example completed. See "clblast.h" for status codes (0 -> success).
		printf("Completed myGEMM10 32F in %.3lf ms with status %d\n", time_ms, 1);
	}
	
	liteB.release();
	*/
	clblast::StatusCode status;
	auto queue_plain = queue();
	
	//Modified clBlast-.. Method 1
	{
		auto start_time = std::chrono::steady_clock::now();

		int error = lclB.GemmF(	m, n, k,
			alpha,
			device_a(), 0, a_ld,
			device_b(), 0, b_ld,
			beta,
			device_c0(), 0, c_ld ,true, false,true,false, false, queue,false);
		queue.finish();
		
		
		auto elapsed_time = std::chrono::steady_clock::now() - start_time;
		auto time_ms = std::chrono::duration<double, std::milli>(elapsed_time).count();

		// Example completed. See "clblast.h" for status codes (0 -> success).
		printf("Completed modified CLBLAST 32F in %.3lf ms with status %d\n", time_ms, error);
	}
		
	// Original clBlast .. Method 2
	{
		auto start_time = std::chrono::steady_clock::now();

		// Call the SGEMM routine. Note that the type of alpha and beta (float) determine the precision.
		status = clblast::Gemm<float>(clblast::Layout::kRowMajor,
			clblast::Transpose::kNo, clblast::Transpose::kNo,
			m, n, k,
			alpha ,
			device_a(), 0, a_ld,
			device_b(), 0, b_ld,
			beta ,
			device_c1(), 0, c_ld,
			&queue_plain, &event);

		// Record the execution time
		if (status == clblast::StatusCode::kSuccess) {
			clWaitForEvents(1, &event);
			clReleaseEvent(event);
		}
		queue.finish();
	
	
		auto elapsed_time = std::chrono::steady_clock::now() - start_time;
		auto time_ms = std::chrono::duration<double, std::milli>(elapsed_time).count();

		// Example completed. See "clblast.h" for status codes (0 -> success).
		printf("Completed clBLAST 32F in %.3lf ms with status %d\n", time_ms, status);

		queue.enqueueReadBuffer(device_c0, CL_TRUE, 0, host_c0.size() * sizeof(float), host_c0.data());
		queue.enqueueReadBuffer(device_c1, CL_TRUE, 0, host_c1.size() * sizeof(float), host_c1.data());
		queue.enqueueReadBuffer(device_c2, CL_TRUE, 0, host_c2.size() * sizeof(float), host_c2.data());
		queue.finish();

	}


	float error = 0.0f;
	float error2 = 0.0f;
	float error3 = 0.0f;

	for (int i = 0; i < host_c0.size(); i++)
	{
		error3 += fabs(host_c0[i] - host_c2[i]);
		error2 += fabs(host_c1[i] - host_c2[i]);
		error += fabs(host_c1[i] - host_c0[i]);
	}

	
	printf("  First values  Method 1 %.3lf , Method 2 %.3lf , Method 3 %.3lf between methods  \n", host_c0[0], host_c1[0], host_c2[0]);
	printf("  Last values  Method 1 %.3lf , Method 2 %.3lf , Method 3 %.3lf between methods  \n", host_c0[host_c0.size()-2], host_c1[host_c1.size() - 2], host_c2[host_c2.size() - 2]);
	printf("  Computed error Method 1 %.3lf , Method 2 %.3lf , Method 3 %.3lf between methods  \n", error , error2, error3);


	res.swap(host_c1);

	return static_cast<int>(status);
}
// Example use of the single-precision Xgemm routine SGEMM
int main() {

  // OpenCL platform/device settings
  
  // Initializes the OpenCL platform
  auto platforms = std::vector<cl::Platform>();
  cl::Platform::get(&platforms);
  
  for (int platform_id = 0; platform_id < platforms.size(); platform_id++)
  {
	  if (platforms.size() == 0 || platform_id >= platforms.size()) { continue; }

	  printf("*****************************\n");
	  printf( platforms[platform_id].getInfo<CL_PLATFORM_NAME>().c_str() );
	  auto platform = platforms[platform_id];

	  // Initializes the OpenCL device
	  auto devices = std::vector<cl::Device>();
	  platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	  for (int device_id = 0; device_id < devices.size(); device_id++)
	  {
		  if (devices.size() == 0 || device_id >= devices.size()) { continue; }
		  		  
		  auto device = devices[device_id];
		  printf("    ");
		  printf(devices[device_id].getInfo<CL_DEVICE_NAME>().c_str());
		  printf("    \n");



		  // Creates the OpenCL context, queue, and an event
		  auto device_as_vector = std::vector<cl::Device>{ device };
		  auto context = cl::Context(device_as_vector);
		  auto queue = cl::CommandQueue(context, device);


		  int clError = 0;
		  clError = lclB.init("clBlast_Kernels.cl", 256, context, device);

		//  clError |= liteB.init("D:\\Sdks\\Yolo\\darknet\\liteGemm.cl", context, device);;

		
		

		  std::vector<float> resF;
		  std::vector<float> resHalf;
		  std::vector<float> resdouble;
		  int status;
		  for (int i = 0; i < 10; i++)
		  {
			  status = test32F(context, queue, resF);
		  }

		  auto elapsed_time = std::chrono::steady_clock::now() - start_time;
		  auto time_ms = std::chrono::duration<double, std::milli>(elapsed_time).count();

		  // Example completed. See "clblast.h" for status codes (0 -> success).
		  printf("Completed SGEMM 32F in %.3lf ms with status %d\n", time_ms, status);


		  status = test16F(context, queue, resHalf);
		  elapsed_time = std::chrono::steady_clock::now() - start_time;
		  time_ms = std::chrono::duration<double, std::milli>(elapsed_time).count();
		  // Example completed. See "clblast.h" for status codes (0 -> success).
		  printf("Completed SGEMMF 16F in %.3lf ms with status %d\n", time_ms, status);

		  status = testDouble(context, queue, resdouble);
		  elapsed_time = std::chrono::steady_clock::now() - start_time;
		  time_ms = std::chrono::duration<double, std::milli>(elapsed_time).count();
		  // Example completed. See "clblast.h" for status codes (0 -> success).
		  printf("Completed SGEMMF double in %.3lf ms with status %d\n", time_ms, status);

		  printf("*****************************\n");

		  float error = 0.0f;
		  float error2 = 0.0f;

		  if (resF.size() == resHalf.size())
		  {
			  for (int i = 0; i < resF.size(); i++)
			  {
				  error += fabs(resF[i] - resHalf[i]);
				  error2 += fabs(resF[i] - resdouble[i]);
			  }
			  printf("  Computed error %.3lf between formats Float and Half \n", error);
			  printf("  Computed error %.3lf between formats Float and Double \n", error2);
		  }
		  printf("*****************************\n");
	  }
  }
  return 0;
}

// =================================================================================================
