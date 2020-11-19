//////////////////////////////////////////////////////////
///////////////////////////////////////////////////////
#define PRECISION 32
#define ROUTINE_GEMM
#define USE_INLINE_KEYWORD 1
#define USE_SUBGROUP_SHUFFLING 1
#define SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA 1

// =================================================================================================
  // Enable support for double-precision
#if PRECISION == 16
#pragma OPENCL EXTENSION cl_khr_fp16: enable
#endif

// Enable support for double-precision
#if PRECISION == 64 || PRECISION == 6464
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

#define ZERO 0.0f
#define ONE 1.0f
#define SMALLEST -1.0e37f


typedef float real;
typedef float2 real2;
typedef float4 real4;
typedef float8 real8;
typedef float16 real16;
// Data-widths
typedef real4 realC;


// Single-element version of a complex number
#if PRECISION == 3232
typedef float singlereal;
#elif PRECISION == 6464
typedef double singlereal;
#else
typedef real singlereal;
#endif

// Converts a 'real argument' value to a 'real' value as passed to the kernel. Normally there is no
// conversion, but half-precision is not supported as kernel argument so it is converted from float.
#if PRECISION == 16
typedef float real_arg;
#define GetRealArg(x) (half)x
#else
typedef real real_arg;
#define GetRealArg(x) x
#endif

// Pointers to local memory objects (using a define because CUDA doesn't need them)
#ifndef LOCAL_PTR
#define LOCAL_PTR __local
#endif

// =================================================================================================

// Don't use the non-IEEE754 compliant OpenCL built-in mad() instruction per default. For specific
// devices, this is enabled (see src/routine.cpp).
#ifndef USE_CL_MAD
#define USE_CL_MAD 0
#endif

// Sets a variable to zero
#if PRECISION == 3232 || PRECISION == 6464
#define SetToZero(a) a.x = ZERO; a.y = ZERO
#else
#define SetToZero(a) a = ZERO
#endif

// Sets a variable to zero (only the imaginary part)
#if PRECISION == 3232 || PRECISION == 6464
#define ImagToZero(a) a.y = ZERO
#else
#define ImagToZero(a) 
#endif

// Sets a variable to one
#if PRECISION == 3232 || PRECISION == 6464
#define SetToOne(a) a.x = ONE; a.y = ZERO
#else
#define SetToOne(a) a = ONE
#endif

// Determines whether a variable is zero
#if PRECISION == 3232 || PRECISION == 6464
#define IsZero(a) ((a.x == ZERO) && (a.y == ZERO))
#else
#define IsZero(a) (a == ZERO)
#endif

// The absolute value (component-wise)
#if PRECISION == 3232 || PRECISION == 6464
#define AbsoluteValue(value) value.x = fabs(value.x); value.y = fabs(value.y)
#else
#define AbsoluteValue(value) value = fabs(value)
#endif

// Negation (component-wise)
#if PRECISION == 3232 || PRECISION == 6464
#define Negate(value) value.x = -(value.x); value.y = -(value.y)
#else
#define Negate(value) value = -(value)
#endif

// Adds two complex variables
#if PRECISION == 3232 || PRECISION == 6464
#define Add(c,a,b) c.x = a.x + b.x; c.y = a.y + b.y
#else
#define Add(c,a,b) c = a + b
#endif

// Subtracts two complex variables
#if PRECISION == 3232 || PRECISION == 6464
#define Subtract(c,a,b) c.x = a.x - b.x; c.y = a.y - b.y
#else
#define Subtract(c,a,b) c = a - b
#endif

// Multiply two complex variables (used in the defines below)
#if PRECISION == 3232 || PRECISION == 6464
#define MulReal(a,b) a.x*b.x - a.y*b.y
#define MulImag(a,b) a.x*b.y + a.y*b.x
#endif

// The scalar multiply function
#if PRECISION == 3232 || PRECISION == 6464
#define Multiply(c,a,b) c.x = MulReal(a,b); c.y = MulImag(a,b)
#else
#define Multiply(c,a,b) c = a * b
#endif

// The scalar multiply-add function
#if PRECISION == 3232 || PRECISION == 6464
#define MultiplyAdd(c,a,b) c.x += MulReal(a,b); c.y += MulImag(a,b)
#else
#if USE_CL_MAD == 1
#define MultiplyAdd(c,a,b) c = mad(a, b, c)
#else
#define MultiplyAdd(c,a,b) c += a * b
#endif
#endif

// The scalar multiply-subtract function
#if PRECISION == 3232 || PRECISION == 6464
#define MultiplySubtract(c,a,b) c.x -= MulReal(a,b); c.y -= MulImag(a,b)
#else
#define MultiplySubtract(c,a,b) c -= a * b
#endif

// The scalar division function: full division
#if PRECISION == 3232 || PRECISION == 6464
#define DivideFull(c,a,b) singlereal num_x = (a.x * b.x) + (a.y * b.y); singlereal num_y = (a.y * b.x) - (a.x * b.y); singlereal denom = (b.x * b.x) + (b.y * b.y); c.x = num_x / denom; c.y = num_y / denom
#else
#define DivideFull(c,a,b) c = a / b
#endif

// The scalar AXPBY function
#if PRECISION == 3232 || PRECISION == 6464
#define AXPBY(e,a,b,c,d) e.x = MulReal(a,b) + MulReal(c,d); e.y = MulImag(a,b) + MulImag(c,d)
#else
#define AXPBY(e,a,b,c,d) e = a*b + c*d
#endif

// The complex conjugate operation for complex transforms
#if PRECISION == 3232 || PRECISION == 6464
#define COMPLEX_CONJUGATE(value) value.x = value.x; value.y = -value.y
#else
#define COMPLEX_CONJUGATE(value) 
#endif

// =================================================================================================

// Force inlining functions or not: some compilers don't support the inline keyword
#ifdef USE_INLINE_KEYWORD
#define INLINE_FUNC inline
#else
#define INLINE_FUNC
#endif

// =================================================================================================

// Shuffled workgroup indices to avoid partition camping, see below. For specific devices, this is
// enabled (see src/routine.cc).
#ifndef USE_STAGGERED_INDICES
#define USE_STAGGERED_INDICES 0
#endif

// Staggered/shuffled group indices to avoid partition camping (AMD GPUs). Formula's are taken from:
// http://docs.nvidia.com/cuda/samples/6_Advanced/transpose/doc/MatrixTranspose.pdf
// More details: https://github.com/CNugteren/CLBlast/issues/53
INLINE_FUNC int GetGroupID1() { return get_group_id(1); }
INLINE_FUNC int GetGroupID0() { return get_group_id(0); }

// =================================================================================================

// End of the C++11 raw string literal
#define COPY_DIMX 8
#define COPY_DIMY 32
#define COPY_VW 4
#define COPY_WPT 1
#define PAD_DIMX 16
#define PAD_DIMY 32
#define PAD_WPTX 1
#define PAD_WPTY 2
#define TRA_DIM 8
#define TRA_PAD 0
#define TRA_SHUFFLE 0
#define TRA_WPT 4
#define PADTRA_PAD 1
#define PADTRA_TILE 16
#define PADTRA_WPT 2
#define GEMMK 0
#define KREG 1
#define KWG 32
#define KWI 2
#define MDIMA 16
#define MDIMC 16
#define MWG 64
#define NDIMB 8
#define NDIMC 8
#define NWG 64
#define SA 1
#define SB 1
#define STRM 0
#define STRN 0
#define VWM 4
#define VWN 4
#define KWID 2
#define MDIMAD 8
#define MDIMCD 8
#define NDIMBD 8
#define NDIMCD 8
#define PADA 1
#define PADB 1
#define VWMD 4
#define VWND 2
#define WGD 32
#define XGEMM_MIN_INDIRECT_SIZE 1536


// =================================================================================================

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.

// For the 'fast' copy kernel
#ifndef COPY_DIMX
#define COPY_DIMX 8      // Local workgroup size in the first dimension (x)
#endif

#ifndef COPY_DIMY
#define COPY_DIMY 8      // Local workgroup size in the second dimension (y)
#endif

#ifndef COPY_WPT
#define COPY_WPT 1       // Work per thread in the first dimension (x)
#endif

// For the padding/copy kernels and the conversion kernels
#ifndef PAD_DIMX
#define PAD_DIMX 8      // Local workgroup size in the first dimension (x)
#endif
#ifndef PAD_DIMY
#define PAD_DIMY 8      // Local workgroup size in the second dimension (y)
#endif
#ifndef PAD_WPTX
#define PAD_WPTX 1      // Work per thread in the first dimension (x)
#endif
#ifndef PAD_WPTY
#define PAD_WPTY 1      // Work per thread in the second dimension (y)
#endif

// For the 'fast' transpose kernel
#ifndef TRA_DIM
#define TRA_DIM 8       // Number of local threads in the two dimensions (x,y)
#endif
#ifndef TRA_WPT
#define TRA_WPT 1       // Work per thread in one dimension and vector-width in the other
#endif
#ifndef TRA_PAD
#define TRA_PAD 0       // Padding of the local memory to avoid bank-conflicts
#endif
#ifndef TRA_SHUFFLE
#define TRA_SHUFFLE 0   // Shuffling of the global indices to avoid global memory bank-conflicts
#endif

// For the padding/transpose kernels
#ifndef PADTRA_TILE
#define PADTRA_TILE 8   // Number of local threads in the two dimensions (x,y)
#endif
#ifndef PADTRA_WPT
#define PADTRA_WPT 1    // Amount of work per thread
#endif
#ifndef PADTRA_PAD
#define PADTRA_PAD 0    // Padding of the local memory to avoid bank-conflicts
#endif


// Data-widths
#if TRA_WPT == 1
typedef float realT;
#elif TRA_WPT == 2
typedef float2 realT;
#elif TRA_WPT == 4
typedef float4 realT;
#elif TRA_WPT == 8
typedef float8 realT;
#elif TRA_WPT == 16
typedef float16 realT;
#endif

// =================================================================================================
// End of the C++11 raw string literal
// =================================================================================================


// =================================================================================================

// Fast copy kernel. Requires 'ld' and the number of threads in dimension 0 to be a multiple of
// COPY_VW. Also requires both matrices to be of the same dimensions and without offset.
__kernel void CopyMatrixFast(const int ld,
	__global const realC* restrict src,
	__global realC* dest,
	const real_arg arg_alpha) {
	const real alpha = GetRealArg(arg_alpha);
#pragma unroll
	for (int _w_one = 0; _w_one < COPY_WPT; _w_one += 1) {
		const int id_one = get_global_id(0);
		const int id_two = (get_group_id(1)*COPY_WPT + _w_one) * COPY_DIMY + get_local_id(1);
		const int id = id_two * (ld / COPY_VW) + id_one;
		realC result;
#if COPY_VW == 1
		Multiply(result, alpha, src[id]);
#elif COPY_VW == 2
		Multiply(result.x, alpha, src[id].x);
		Multiply(result.y, alpha, src[id].y);
#elif COPY_VW == 4
		Multiply(result.x, alpha, src[id].x);
		Multiply(result.y, alpha, src[id].y);
		Multiply(result.z, alpha, src[id].z);
		Multiply(result.w, alpha, src[id].w);
#elif COPY_VW == 8
		Multiply(result.s0, alpha, src[id].s0);
		Multiply(result.s1, alpha, src[id].s1);
		Multiply(result.s2, alpha, src[id].s2);
		Multiply(result.s3, alpha, src[id].s3);
		Multiply(result.s4, alpha, src[id].s4);
		Multiply(result.s5, alpha, src[id].s5);
		Multiply(result.s6, alpha, src[id].s6);
		Multiply(result.s7, alpha, src[id].s7);
#elif COPY_VW == 16
		Multiply(result.s0, alpha, src[id].s0);
		Multiply(result.s1, alpha, src[id].s1);
		Multiply(result.s2, alpha, src[id].s2);
		Multiply(result.s3, alpha, src[id].s3);
		Multiply(result.s4, alpha, src[id].s4);
		Multiply(result.s5, alpha, src[id].s5);
		Multiply(result.s6, alpha, src[id].s6);
		Multiply(result.s7, alpha, src[id].s7);
		Multiply(result.s8, alpha, src[id].s8);
		Multiply(result.s9, alpha, src[id].s9);
		Multiply(result.sA, alpha, src[id].sA);
		Multiply(result.sB, alpha, src[id].sB);
		Multiply(result.sC, alpha, src[id].sC);
		Multiply(result.sD, alpha, src[id].sD);
		Multiply(result.sE, alpha, src[id].sE);
		Multiply(result.sF, alpha, src[id].sF);
#endif
		dest[id] = result;;
	}
}

// =================================================================================================

// End of the C++11 raw string literal


// =================================================================================================

// Copies a matrix from source to destination. The output is padded with zero values in case the
// destination matrix dimensions are larger than the source matrix dimensions. Additionally, the ld
// value and offset can be different.
INLINE_FUNC void _CopyPadMatrix(const int src_one, const int src_two,
	const int src_ld, const int src_offset,
	__global const real* restrict src,
	const int dest_one, const int dest_two,
	const int dest_ld, const int dest_offset,
	__global real* dest,
	const real alpha,
	const int do_conjugate) {

	// Loops over the work per thread in both dimensions
#pragma unroll
	for (int _w_one = 0; _w_one < PAD_WPTX; _w_one += 1)
	{
		const int id_one = (get_group_id(0)*PAD_WPTX + _w_one) * PAD_DIMX + get_local_id(0);
#pragma unroll
		for (int _w_two = 0; _w_two < PAD_WPTY; _w_two += 1)
		{
			const int id_two = (get_group_id(1)*PAD_WPTY + _w_two) * PAD_DIMY + get_local_id(1);
			if ((id_two < dest_two) && (id_one < dest_one))
			{

				// Loads data if the thread IDs are within bounds of the source matrix. Otherwise, set the
				// value to be written to zero.
				real value;
				SetToZero(value);
				if ((id_two < src_two) && (id_one < src_one))
				{
					value = src[id_two*src_ld + id_one + src_offset];
				}

				// Stores the value in the destination matrix
				if (do_conjugate == 1) { COMPLEX_CONJUGATE(value); }
				Multiply(dest[id_two*dest_ld + id_one + dest_offset], alpha, value);
			}
		}
	}
}

// Interface to the above function
__kernel void CopyPadMatrix(const int src_one, const int src_two,
	const int src_ld, const int src_offset,
	__global const real* restrict src,
	const int dest_one, const int dest_two,
	const int dest_ld, const int dest_offset,
	__global real* dest,
	const real_arg arg_alpha,
	const int do_conjugate) {
	const real alpha = GetRealArg(arg_alpha);
	_CopyPadMatrix(src_one, src_two, src_ld, src_offset, src,
		dest_one, dest_two, dest_ld, dest_offset, dest,
		alpha, do_conjugate);
}

// =================================================================================================

// Same as above, but now un-pads a matrix. This kernel reads data from a padded source matrix, but
// writes only the actual data back to the destination matrix. Again, the ld value and offset can
// be different.
INLINE_FUNC void _CopyMatrix(const int src_one, const int src_two,
	const int src_ld, const int src_offset,
	__global const real* restrict src,
	const int dest_one, const int dest_two,
	const int dest_ld, const int dest_offset,
	__global real* dest,
	const real alpha,
	const int upper, const int lower,
	const int diagonal_imag_zero) {

	// Loops over the work per thread in both dimensions
#pragma unroll
	for (int _w_one = 0; _w_one < PAD_WPTX; _w_one += 1) {
		const int id_one = (get_group_id(0)*PAD_WPTX + _w_one) * PAD_DIMX + get_local_id(0);
#pragma unroll
		for (int _w_two = 0; _w_two < PAD_WPTY; _w_two += 1) {
			const int id_two = (get_group_id(1)*PAD_WPTY + _w_two) * PAD_DIMY + get_local_id(1);

			// Masking in case of triangular matrices: updates only the upper or lower part
			bool condition = true;
#if defined(ROUTINE_SYRK) || defined(ROUTINE_HERK) || defined(ROUTINE_SYR2K) || defined(ROUTINE_HER2K)
			if (upper == 1) { condition = (id_two >= id_one); }
			else if (lower == 1) { condition = (id_two <= id_one); }
#endif
			if (condition) {

				// Copies the value into the destination matrix. This is always within bounds of the source
				// matrix, as we know that the destination matrix is smaller or equal to the source.
				if (id_two < dest_two && id_one < dest_one) {
					real value = src[id_two*src_ld + id_one + src_offset];
					if (diagonal_imag_zero == 1 && id_one == id_two) { ImagToZero(value); }
					Multiply(dest[id_two*dest_ld + id_one + dest_offset], alpha, value);
				}
			}
		}
	}
}

// Interface to the above function
__kernel void CopyMatrix(const int src_one, const int src_two,
	const int src_ld, const int src_offset,
	__global const real* restrict src,
	const int dest_one, const int dest_two,
	const int dest_ld, const int dest_offset,
	__global real* dest,
	const real_arg arg_alpha,
	const int upper, const int lower,
	const int diagonal_imag_zero) {
	const real alpha = GetRealArg(arg_alpha);
	_CopyMatrix(src_one, src_two, src_ld, src_offset, src,
		dest_one, dest_two, dest_ld, dest_offset, dest,
		alpha, upper, lower, diagonal_imag_zero);
}

// End of the C++11 raw string literal


// =================================================================================================


// =================================================================================================

// Transposes and copies a matrix. Requires both matrices to be of the same dimensions and without
// offset. A more general version is available in 'padtranspose.opencl'.
__kernel void TransposeMatrixFast(const int ld,
	__global const realT* restrict src,
	__global realT* dest,
	const real_arg arg_alpha) {
	const real alpha = GetRealArg(arg_alpha);

	// Sets the group identifiers. They might be 'shuffled' around to distribute work in a different
	// way over workgroups, breaking memory-bank dependencies.
	const int gid0 = get_group_id(0);
#if TRA_SHUFFLE == 1
	const int gid1 = (get_group_id(0) + get_group_id(1)) % get_num_groups(0);
#else
	const int gid1 = get_group_id(1);
#endif

	// Local memory to store a tile of the matrix (for coalescing)
	__local realT tile[TRA_WPT*TRA_DIM][TRA_DIM + TRA_PAD];

	// Loops over the work per thread
#pragma unroll
	for (int _w_one = 0; _w_one < TRA_WPT; _w_one += 1) {

		// Computes the identifiers for the source matrix. Note that the local and global dimensions
		// do not correspond to each other!
		const int id_one = gid1 * TRA_DIM + get_local_id(0);
		const int id_two = (gid0 * TRA_DIM + get_local_id(1))*TRA_WPT + _w_one;

		// Loads data into the local memory
		realT value = src[id_two*(ld / TRA_WPT) + id_one];
		tile[get_local_id(0)*TRA_WPT + _w_one][get_local_id(1)] = value;
	}

	// Synchronizes all threads in a workgroup
	barrier(CLK_LOCAL_MEM_FENCE);

	// Loads transposed data from the local memory
#pragma promote_to_registers
	realT vpm[TRA_WPT];
#pragma unroll
	for (int _w_one = 0; _w_one < TRA_WPT; _w_one += 1) {
		vpm[_w_one] = tile[get_local_id(1)*TRA_WPT + _w_one][get_local_id(0)];
	}

	// Performs the register-level transpose of the vectorized data
#pragma promote_to_registers
	realT results[TRA_WPT];
#if TRA_WPT == 1
	results[0] = vpm[0];
#elif TRA_WPT == 2
	results[0].x = vpm[0].x; results[0].y = vpm[1].x;
	results[1].x = vpm[0].y; results[1].y = vpm[1].y;
#elif TRA_WPT == 4
	results[0].x = vpm[0].x; results[0].y = vpm[1].x; results[0].z = vpm[2].x; results[0].w = vpm[3].x;
	results[1].x = vpm[0].y; results[1].y = vpm[1].y; results[1].z = vpm[2].y; results[1].w = vpm[3].y;
	results[2].x = vpm[0].z; results[2].y = vpm[1].z; results[2].z = vpm[2].z; results[2].w = vpm[3].z;
	results[3].x = vpm[0].w; results[3].y = vpm[1].w; results[3].z = vpm[2].w; results[3].w = vpm[3].w;
#elif TRA_WPT == 8
	results[0].s0 = vpm[0].s0; results[0].s1 = vpm[1].s0; results[0].s2 = vpm[2].s0; results[0].s3 = vpm[3].s0; results[0].s4 = vpm[4].s0; results[0].s5 = vpm[5].s0; results[0].s6 = vpm[6].s0; results[0].s7 = vpm[7].s0;
	results[1].s0 = vpm[0].s1; results[1].s1 = vpm[1].s1; results[1].s2 = vpm[2].s1; results[1].s3 = vpm[3].s1; results[1].s4 = vpm[4].s1; results[1].s5 = vpm[5].s1; results[1].s6 = vpm[6].s1; results[1].s7 = vpm[7].s1;
	results[2].s0 = vpm[0].s2; results[2].s1 = vpm[1].s2; results[2].s2 = vpm[2].s2; results[2].s3 = vpm[3].s2; results[2].s4 = vpm[4].s2; results[2].s5 = vpm[5].s2; results[2].s6 = vpm[6].s2; results[2].s7 = vpm[7].s2;
	results[3].s0 = vpm[0].s3; results[3].s1 = vpm[1].s3; results[3].s2 = vpm[2].s3; results[3].s3 = vpm[3].s3; results[3].s4 = vpm[4].s3; results[3].s5 = vpm[5].s3; results[3].s6 = vpm[6].s3; results[3].s7 = vpm[7].s3;
	results[4].s0 = vpm[0].s4; results[4].s1 = vpm[1].s4; results[4].s2 = vpm[2].s4; results[4].s3 = vpm[3].s4; results[4].s4 = vpm[4].s4; results[4].s5 = vpm[5].s4; results[4].s6 = vpm[6].s4; results[4].s7 = vpm[7].s4;
	results[5].s0 = vpm[0].s5; results[5].s1 = vpm[1].s5; results[5].s2 = vpm[2].s5; results[5].s3 = vpm[3].s5; results[5].s4 = vpm[4].s5; results[5].s5 = vpm[5].s5; results[5].s6 = vpm[6].s5; results[5].s7 = vpm[7].s5;
	results[6].s0 = vpm[0].s6; results[6].s1 = vpm[1].s6; results[6].s2 = vpm[2].s6; results[6].s3 = vpm[3].s6; results[6].s4 = vpm[4].s6; results[6].s5 = vpm[5].s6; results[6].s6 = vpm[6].s6; results[6].s7 = vpm[7].s6;
	results[7].s0 = vpm[0].s7; results[7].s1 = vpm[1].s7; results[7].s2 = vpm[2].s7; results[7].s3 = vpm[3].s7; results[7].s4 = vpm[4].s7; results[7].s5 = vpm[5].s7; results[7].s6 = vpm[6].s7; results[7].s7 = vpm[7].s7;
#elif TRA_WPT == 16
	results[0].s0 = vpm[0].s0; results[0].s1 = vpm[1].s0; results[0].s2 = vpm[2].s0; results[0].s3 = vpm[3].s0; results[0].s4 = vpm[4].s0; results[0].s5 = vpm[5].s0; results[0].s6 = vpm[6].s0; results[0].s7 = vpm[7].s0; results[0].s8 = vpm[8].s0; results[0].s9 = vpm[9].s0; results[0].sA = vpm[10].s0; results[0].sB = vpm[11].s0; results[0].sC = vpm[12].s0; results[0].sD = vpm[13].s0; results[0].sE = vpm[14].s0; results[0].sF = vpm[15].s0;
	results[1].s0 = vpm[0].s1; results[1].s1 = vpm[1].s1; results[1].s2 = vpm[2].s1; results[1].s3 = vpm[3].s1; results[1].s4 = vpm[4].s1; results[1].s5 = vpm[5].s1; results[1].s6 = vpm[6].s1; results[1].s7 = vpm[7].s1; results[1].s8 = vpm[8].s1; results[1].s9 = vpm[9].s1; results[1].sA = vpm[10].s1; results[1].sB = vpm[11].s1; results[1].sC = vpm[12].s1; results[1].sD = vpm[13].s1; results[1].sE = vpm[14].s1; results[1].sF = vpm[15].s1;
	results[2].s0 = vpm[0].s2; results[2].s1 = vpm[1].s2; results[2].s2 = vpm[2].s2; results[2].s3 = vpm[3].s2; results[2].s4 = vpm[4].s2; results[2].s5 = vpm[5].s2; results[2].s6 = vpm[6].s2; results[2].s7 = vpm[7].s2; results[2].s8 = vpm[8].s2; results[2].s9 = vpm[9].s2; results[2].sA = vpm[10].s2; results[2].sB = vpm[11].s2; results[2].sC = vpm[12].s2; results[2].sD = vpm[13].s2; results[2].sE = vpm[14].s2; results[2].sF = vpm[15].s2;
	results[3].s0 = vpm[0].s3; results[3].s1 = vpm[1].s3; results[3].s2 = vpm[2].s3; results[3].s3 = vpm[3].s3; results[3].s4 = vpm[4].s3; results[3].s5 = vpm[5].s3; results[3].s6 = vpm[6].s3; results[3].s7 = vpm[7].s3; results[3].s8 = vpm[8].s3; results[3].s9 = vpm[9].s3; results[3].sA = vpm[10].s3; results[3].sB = vpm[11].s3; results[3].sC = vpm[12].s3; results[3].sD = vpm[13].s3; results[3].sE = vpm[14].s3; results[3].sF = vpm[15].s3;
	results[4].s0 = vpm[0].s4; results[4].s1 = vpm[1].s4; results[4].s2 = vpm[2].s4; results[4].s3 = vpm[3].s4; results[4].s4 = vpm[4].s4; results[4].s5 = vpm[5].s4; results[4].s6 = vpm[6].s4; results[4].s7 = vpm[7].s4; results[4].s8 = vpm[8].s4; results[4].s9 = vpm[9].s4; results[4].sA = vpm[10].s4; results[4].sB = vpm[11].s4; results[4].sC = vpm[12].s4; results[4].sD = vpm[13].s4; results[4].sE = vpm[14].s4; results[4].sF = vpm[15].s4;
	results[5].s0 = vpm[0].s5; results[5].s1 = vpm[1].s5; results[5].s2 = vpm[2].s5; results[5].s3 = vpm[3].s5; results[5].s4 = vpm[4].s5; results[5].s5 = vpm[5].s5; results[5].s6 = vpm[6].s5; results[5].s7 = vpm[7].s5; results[5].s8 = vpm[8].s5; results[5].s9 = vpm[9].s5; results[5].sA = vpm[10].s5; results[5].sB = vpm[11].s5; results[5].sC = vpm[12].s5; results[5].sD = vpm[13].s5; results[5].sE = vpm[14].s5; results[5].sF = vpm[15].s5;
	results[6].s0 = vpm[0].s6; results[6].s1 = vpm[1].s6; results[6].s2 = vpm[2].s6; results[6].s3 = vpm[3].s6; results[6].s4 = vpm[4].s6; results[6].s5 = vpm[5].s6; results[6].s6 = vpm[6].s6; results[6].s7 = vpm[7].s6; results[6].s8 = vpm[8].s6; results[6].s9 = vpm[9].s6; results[6].sA = vpm[10].s6; results[6].sB = vpm[11].s6; results[6].sC = vpm[12].s6; results[6].sD = vpm[13].s6; results[6].sE = vpm[14].s6; results[6].sF = vpm[15].s6;
	results[7].s0 = vpm[0].s7; results[7].s1 = vpm[1].s7; results[7].s2 = vpm[2].s7; results[7].s3 = vpm[3].s7; results[7].s4 = vpm[4].s7; results[7].s5 = vpm[5].s7; results[7].s6 = vpm[6].s7; results[7].s7 = vpm[7].s7; results[7].s8 = vpm[8].s7; results[7].s9 = vpm[9].s7; results[7].sA = vpm[10].s7; results[7].sB = vpm[11].s7; results[7].sC = vpm[12].s7; results[7].sD = vpm[13].s7; results[7].sE = vpm[14].s7; results[7].sF = vpm[15].s7;
	results[8].s0 = vpm[0].s8; results[8].s1 = vpm[1].s8; results[8].s2 = vpm[2].s8; results[8].s3 = vpm[3].s8; results[8].s4 = vpm[4].s8; results[8].s5 = vpm[5].s8; results[8].s6 = vpm[6].s8; results[8].s7 = vpm[7].s8; results[8].s8 = vpm[8].s8; results[8].s9 = vpm[9].s8; results[8].sA = vpm[10].s8; results[8].sB = vpm[11].s8; results[8].sC = vpm[12].s8; results[8].sD = vpm[13].s8; results[8].sE = vpm[14].s8; results[8].sF = vpm[15].s8;
	results[9].s0 = vpm[0].s9; results[9].s1 = vpm[1].s9; results[9].s2 = vpm[2].s9; results[9].s3 = vpm[3].s9; results[9].s4 = vpm[4].s9; results[9].s5 = vpm[5].s9; results[9].s6 = vpm[6].s9; results[9].s7 = vpm[7].s9; results[9].s8 = vpm[8].s9; results[9].s9 = vpm[9].s9; results[9].sA = vpm[10].s9; results[9].sB = vpm[11].s9; results[9].sC = vpm[12].s9; results[9].sD = vpm[13].s9; results[9].sE = vpm[14].s9; results[9].sF = vpm[15].s9;
	results[10].s0 = vpm[0].sA; results[10].s1 = vpm[1].sA; results[10].s2 = vpm[2].sA; results[10].s3 = vpm[3].sA; results[10].s4 = vpm[4].sA; results[10].s5 = vpm[5].sA; results[10].s6 = vpm[6].sA; results[10].s7 = vpm[7].sA; results[10].s8 = vpm[8].sA; results[10].s9 = vpm[9].sA; results[10].sA = vpm[10].sA; results[10].sB = vpm[11].sA; results[10].sC = vpm[12].sA; results[10].sD = vpm[13].sA; results[10].sE = vpm[14].sA; results[10].sF = vpm[15].sA;
	results[11].s0 = vpm[0].sB; results[11].s1 = vpm[1].sB; results[11].s2 = vpm[2].sB; results[11].s3 = vpm[3].sB; results[11].s4 = vpm[4].sB; results[11].s5 = vpm[5].sB; results[11].s6 = vpm[6].sB; results[11].s7 = vpm[7].sB; results[11].s8 = vpm[8].sB; results[11].s9 = vpm[9].sB; results[11].sA = vpm[10].sB; results[11].sB = vpm[11].sB; results[11].sC = vpm[12].sB; results[11].sD = vpm[13].sB; results[11].sE = vpm[14].sB; results[11].sF = vpm[15].sB;
	results[12].s0 = vpm[0].sC; results[12].s1 = vpm[1].sC; results[12].s2 = vpm[2].sC; results[12].s3 = vpm[3].sC; results[12].s4 = vpm[4].sC; results[12].s5 = vpm[5].sC; results[12].s6 = vpm[6].sC; results[12].s7 = vpm[7].sC; results[12].s8 = vpm[8].sC; results[12].s9 = vpm[9].sC; results[12].sA = vpm[10].sC; results[12].sB = vpm[11].sC; results[12].sC = vpm[12].sC; results[12].sD = vpm[13].sC; results[12].sE = vpm[14].sC; results[12].sF = vpm[15].sC;
	results[13].s0 = vpm[0].sD; results[13].s1 = vpm[1].sD; results[13].s2 = vpm[2].sD; results[13].s3 = vpm[3].sD; results[13].s4 = vpm[4].sD; results[13].s5 = vpm[5].sD; results[13].s6 = vpm[6].sD; results[13].s7 = vpm[7].sD; results[13].s8 = vpm[8].sD; results[13].s9 = vpm[9].sD; results[13].sA = vpm[10].sD; results[13].sB = vpm[11].sD; results[13].sC = vpm[12].sD; results[13].sD = vpm[13].sD; results[13].sE = vpm[14].sD; results[13].sF = vpm[15].sD;
	results[14].s0 = vpm[0].sE; results[14].s1 = vpm[1].sE; results[14].s2 = vpm[2].sE; results[14].s3 = vpm[3].sE; results[14].s4 = vpm[4].sE; results[14].s5 = vpm[5].sE; results[14].s6 = vpm[6].sE; results[14].s7 = vpm[7].sE; results[14].s8 = vpm[8].sE; results[14].s9 = vpm[9].sE; results[14].sA = vpm[10].sE; results[14].sB = vpm[11].sE; results[14].sC = vpm[12].sE; results[14].sD = vpm[13].sE; results[14].sE = vpm[14].sE; results[14].sF = vpm[15].sE;
	results[15].s0 = vpm[0].sF; results[15].s1 = vpm[1].sF; results[15].s2 = vpm[2].sF; results[15].s3 = vpm[3].sF; results[15].s4 = vpm[4].sF; results[15].s5 = vpm[5].sF; results[15].s6 = vpm[6].sF; results[15].s7 = vpm[7].sF; results[15].s8 = vpm[8].sF; results[15].s9 = vpm[9].sF; results[15].sA = vpm[10].sF; results[15].sB = vpm[11].sF; results[15].sC = vpm[12].sF; results[15].sD = vpm[13].sF; results[15].sE = vpm[14].sF; results[15].sF = vpm[15].sF;
#endif

	// Multiplies by alpha and then stores the results into the destination matrix
#pragma unroll
	for (int _w_two = 0; _w_two < TRA_WPT; _w_two += 1) {
		realT result;
#if TRA_WPT == 1
		Multiply(result, alpha, results[_w_two]);
#elif TRA_WPT == 2
		Multiply(result.x, alpha, results[_w_two].x);
		Multiply(result.y, alpha, results[_w_two].y);
#elif TRA_WPT == 4
		Multiply(result.x, alpha, results[_w_two].x);
		Multiply(result.y, alpha, results[_w_two].y);
		Multiply(result.z, alpha, results[_w_two].z);
		Multiply(result.w, alpha, results[_w_two].w);
#elif TRA_WPT == 8
		Multiply(result.s0, alpha, results[_w_two].s0);
		Multiply(result.s1, alpha, results[_w_two].s1);
		Multiply(result.s2, alpha, results[_w_two].s2);
		Multiply(result.s3, alpha, results[_w_two].s3);
		Multiply(result.s4, alpha, results[_w_two].s4);
		Multiply(result.s5, alpha, results[_w_two].s5);
		Multiply(result.s6, alpha, results[_w_two].s6);
		Multiply(result.s7, alpha, results[_w_two].s7);
#elif TRA_WPT == 16
		Multiply(result.s0, alpha, results[_w_two].s0);
		Multiply(result.s1, alpha, results[_w_two].s1);
		Multiply(result.s2, alpha, results[_w_two].s2);
		Multiply(result.s3, alpha, results[_w_two].s3);
		Multiply(result.s4, alpha, results[_w_two].s4);
		Multiply(result.s5, alpha, results[_w_two].s5);
		Multiply(result.s6, alpha, results[_w_two].s6);
		Multiply(result.s7, alpha, results[_w_two].s7);
		Multiply(result.s8, alpha, results[_w_two].s8);
		Multiply(result.s9, alpha, results[_w_two].s9);
		Multiply(result.sA, alpha, results[_w_two].sA);
		Multiply(result.sB, alpha, results[_w_two].sB);
		Multiply(result.sC, alpha, results[_w_two].sC);
		Multiply(result.sD, alpha, results[_w_two].sD);
		Multiply(result.sE, alpha, results[_w_two].sE);
		Multiply(result.sF, alpha, results[_w_two].sF);
#endif
		const int id_one = gid0 * TRA_DIM + get_local_id(0);
		const int id_two = (gid1*TRA_DIM + get_local_id(1))*TRA_WPT + _w_two;
		dest[id_two*(ld / TRA_WPT) + id_one] = result;
	}
}

// =================================================================================================

// End of the C++11 raw string literal


// =================================================================================================

// Transposes a matrix from source to destination. The output is padded with zero values in case the
// destination matrix dimensions are larger than the transposed source matrix dimensions.
INLINE_FUNC void _TransposePadMatrix(LOCAL_PTR real* tile,
	const int src_one, const int src_two,
	const int src_ld, const int src_offset,
	__global const real* restrict src,
	const int dest_one, const int dest_two,
	const int dest_ld, const int dest_offset,
	__global real* dest,
	const real alpha,
	const int do_conjugate) {

	// Loop over the work per thread
#pragma unroll
	for (int _w_one = 0; _w_one < PADTRA_WPT; _w_one += 1) {
#pragma unroll
		for (int _w_two = 0; _w_two < PADTRA_WPT; _w_two += 1) {

			// Computes the identifiers for the source matrix. Note that the local and global dimensions
			// do not correspond to each other!
			const int id_src_one = (get_group_id(1)*PADTRA_WPT + _w_two) * PADTRA_TILE + get_local_id(0);
			const int id_src_two = (get_group_id(0)*PADTRA_WPT + _w_one) * PADTRA_TILE + get_local_id(1);

			// Loads data into the local memory if the thread IDs are within bounds of the source matrix.
			// Otherwise, set the local memory value to zero.
			real value;
			SetToZero(value);
			if (id_src_two < src_two && id_src_one < src_one) {
				value = src[id_src_two*src_ld + id_src_one + src_offset];
			}
			const int tile_id0 = get_local_id(0)*PADTRA_WPT + _w_one;
			const int tile_id1 = get_local_id(1)*PADTRA_WPT + _w_two;
			tile[tile_id1 * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD) + tile_id0] = value;
		}
	}

	// Synchronizes all threads in a workgroup
	barrier(CLK_LOCAL_MEM_FENCE);

	// Loop over the work per thread
#pragma unroll
	for (int _w_one = 0; _w_one < PADTRA_WPT; _w_one += 1) {
#pragma unroll
		for (int _w_two = 0; _w_two < PADTRA_WPT; _w_two += 1) {

			// Computes the identifiers for the destination matrix
			const int id_dest_one = (get_group_id(0)*PADTRA_WPT + _w_one) * PADTRA_TILE + get_local_id(0);
			const int id_dest_two = (get_group_id(1)*PADTRA_WPT + _w_two) * PADTRA_TILE + get_local_id(1);

			// Stores the transposed value in the destination matrix
			if ((id_dest_one < dest_one) && (id_dest_two < dest_two)) {
				const int tile_id0 = get_local_id(1)*PADTRA_WPT + _w_one;
				const int tile_id1 = get_local_id(0)*PADTRA_WPT + _w_two;
				real value = tile[tile_id1 * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD) + tile_id0];
				if (do_conjugate == 1) { COMPLEX_CONJUGATE(value); }
				Multiply(dest[id_dest_two*dest_ld + id_dest_one + dest_offset], alpha, value);
			}
		}
	}
}

// Interface to the above function
__kernel void TransposePadMatrix(const int src_one, const int src_two,
	const int src_ld, const int src_offset,
	__global const real* restrict src,
	const int dest_one, const int dest_two,
	const int dest_ld, const int dest_offset,
	__global real* dest,
	const real_arg arg_alpha,
	const int do_conjugate) {
	const real alpha = GetRealArg(arg_alpha);
	__local real tile[(PADTRA_WPT*PADTRA_TILE) * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD)];
	_TransposePadMatrix(tile, src_one, src_two, src_ld, src_offset, src,
		dest_one, dest_two, dest_ld, dest_offset, dest,
		alpha, do_conjugate);
}

// =================================================================================================

// Transposes a matrix, while considering possible padding in the source matrix. Data is read from a
// padded source matrix, but only the actual data is written back to the transposed destination
// matrix. This kernel optionally checks for upper/lower triangular matrices.
INLINE_FUNC void _TransposeMatrix(LOCAL_PTR real* tile,
	const int src_one, const int src_two,
	const int src_ld, const int src_offset,
	__global const real* restrict src,
	const int dest_one, const int dest_two,
	const int dest_ld, const int dest_offset,
	__global real* dest,
	const real alpha,
	const int upper, const int lower,
	const int diagonal_imag_zero) {

	// Loop over the work per thread
#pragma unroll
	for (int _w_one = 0; _w_one < PADTRA_WPT; _w_one += 1) {
#pragma unroll
		for (int _w_two = 0; _w_two < PADTRA_WPT; _w_two += 1) {

			// Computes the identifiers for the source matrix. Note that the local and global dimensions
			// do not correspond to each other!
			const int id_src_one = (get_group_id(1)*PADTRA_WPT + _w_two) * PADTRA_TILE + get_local_id(0);
			const int id_src_two = (get_group_id(0)*PADTRA_WPT + _w_one) * PADTRA_TILE + get_local_id(1);

			// Loads data into the local memory if the thread IDs are within bounds of the source matrix.
			if ((id_src_one < src_one) && (id_src_two < src_two)) {
				real value = src[id_src_two*src_ld + id_src_one + src_offset];
				const int tile_id0 = get_local_id(0)*PADTRA_WPT + _w_one;
				const int tile_id1 = get_local_id(1)*PADTRA_WPT + _w_two;
				tile[tile_id1 * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD) + tile_id0] = value;
			}
		}
	}

	// Synchronizes all threads in a workgroup
	barrier(CLK_LOCAL_MEM_FENCE);

	// Loop over the work per thread
#pragma unroll
	for (int _w_one = 0; _w_one < PADTRA_WPT; _w_one += 1) {
#pragma unroll
		for (int _w_two = 0; _w_two < PADTRA_WPT; _w_two += 1) {

			// Computes the identifiers for the destination matrix
			const int id_dest_one = (get_group_id(0)*PADTRA_WPT + _w_one) * PADTRA_TILE + get_local_id(0);
			const int id_dest_two = (get_group_id(1)*PADTRA_WPT + _w_two) * PADTRA_TILE + get_local_id(1);

			// Masking in case of triangular matrices: updates only the upper or lower part
			bool condition = true;
#if defined(ROUTINE_SYRK) || defined(ROUTINE_HERK) || defined(ROUTINE_SYR2K) || defined(ROUTINE_HER2K)
			if (upper == 1) { condition = (id_dest_one >= id_dest_two); }
			else if (lower == 1) { condition = (id_dest_one <= id_dest_two); }
#endif
			if (condition) {

				// Stores the transposed value in the destination matrix
				if ((id_dest_one < dest_one) && (id_dest_two < dest_two)) {
					const int tile_id0 = get_local_id(1)*PADTRA_WPT + _w_one;
					const int tile_id1 = get_local_id(0)*PADTRA_WPT + _w_two;
					real value = tile[tile_id1 * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD) + tile_id0];
					if (diagonal_imag_zero == 1 && id_dest_one == id_dest_two) { ImagToZero(value); }
					Multiply(dest[id_dest_two*dest_ld + id_dest_one + dest_offset], alpha, value);
				}
			}
		}
	}
}

// Interface to the above function
__kernel void TransposeMatrix(const int src_one, const int src_two,
	const int src_ld, const int src_offset,
	__global const real* restrict src,
	const int dest_one, const int dest_two,
	const int dest_ld, const int dest_offset,
	__global real* dest,
	const real_arg arg_alpha,
	const int upper, const int lower,
	const int diagonal_imag_zero) {
	const real alpha = GetRealArg(arg_alpha);
	__local real tile[(PADTRA_WPT*PADTRA_TILE) * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD)];
	_TransposeMatrix(tile, src_one, src_two, src_ld, src_offset, src,
		dest_one, dest_two, dest_ld, dest_offset, dest,
		alpha, upper, lower, diagonal_imag_zero);
}

// =================================================================================================
#if defined(ROUTINE_GEMMBATCHED)

// Batched version of the above
__kernel void TransposePadMatrixBatched(const int src_one, const int src_two,
	const int src_ld, const __constant int* src_offsets,
	__global const real* restrict src,
	const int dest_one, const int dest_two,
	const int dest_ld, const __constant int* dest_offsets,
	__global real* dest,
	const int do_conjugate) {
	const int batch = get_group_id(2);
	const int src_offset = src_offsets[batch];
	const int dest_offset = dest_offsets[batch];
	real alpha; SetToOne(alpha);
	__local real tile[(PADTRA_WPT*PADTRA_TILE) * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD)];
	_TransposePadMatrix(tile, src_one, src_two, src_ld, src_offset, src,
		dest_one, dest_two, dest_ld, dest_offset, dest,
		alpha, do_conjugate);
}

// Batched version of the above
__kernel void TransposeMatrixBatched(const int src_one, const int src_two,
	const int src_ld, const __constant int* src_offsets,
	__global const real* restrict src,
	const int dest_one, const int dest_two,
	const int dest_ld, const __constant int* dest_offsets,
	__global real* dest) {
	const int batch = get_group_id(2);
	const int src_offset = src_offsets[batch];
	const int dest_offset = dest_offsets[batch];
	real alpha; SetToOne(alpha);
	__local real tile[(PADTRA_WPT*PADTRA_TILE) * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD)];
	_TransposeMatrix(tile, src_one, src_two, src_ld, src_offset, src,
		dest_one, dest_two, dest_ld, dest_offset, dest,
		alpha, 0, 0, 0);
}

#endif
// =================================================================================================
#if defined(ROUTINE_GEMMSTRIDEDBATCHED)

// Strided-batched version of the above
__kernel void TransposePadMatrixStridedBatched(const int src_one, const int src_two,
	const int src_ld, const int src_offset,
	const int src_stride, __global const real* restrict src,
	const int dest_one, const int dest_two,
	const int dest_ld, const int dest_offset,
	const int dest_stride, __global real* dest,
	const int do_conjugate) {
	const int batch = get_group_id(2);
	const int src_offset_batch = src_offset + src_stride * batch;
	const int dest_offset_batch = dest_offset + dest_stride * batch;
	real alpha; SetToOne(alpha);
	__local real tile[(PADTRA_WPT*PADTRA_TILE) * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD)];
	_TransposePadMatrix(tile, src_one, src_two, src_ld, src_offset_batch, src,
		dest_one, dest_two, dest_ld, dest_offset_batch, dest,
		alpha, do_conjugate);
}

// Strided-batched version of the above
__kernel void TransposeMatrixStridedBatched(const int src_one, const int src_two,
	const int src_ld, const int src_offset,
	const int src_stride, __global const real* restrict src,
	const int dest_one, const int dest_two,
	const int dest_ld, const int dest_offset,
	const int dest_stride, __global real* dest) {
	const int batch = get_group_id(2);
	const int src_offset_batch = src_offset + src_stride * batch;
	const int dest_offset_batch = dest_offset + dest_stride * batch;
	real alpha; SetToOne(alpha);
	__local real tile[(PADTRA_WPT*PADTRA_TILE) * (PADTRA_WPT*PADTRA_TILE + PADTRA_PAD)];
	_TransposeMatrix(tile, src_one, src_two, src_ld, src_offset_batch, src,
		dest_one, dest_two, dest_ld, dest_offset_batch, dest,
		alpha, 0, 0, 0);
}

#endif
// =================================================================================================

// End of the C++11 raw string literal


// =================================================================================================
#if defined(ROUTINE_SYMM)

// Kernel to populate a squared symmetric matrix, given that the triangle which holds the data is
// stored as the lower-triangle of the input matrix. This uses the padding kernel's parameters.
__kernel void SymmLowerToSquared(const int src_dim,
	const int src_ld, const int src_offset,
	__global const real* restrict src,
	const int dest_dim,
	const int dest_ld, const int dest_offset,
	__global real* dest) {

	// Loops over the work per thread in both dimensions
#pragma unroll
	for (int _w_one = 0; _w_one < PAD_WPTX; _w_one += 1) {
		const int id_one = (get_group_id(0)*PAD_WPTX + _w_one) * PAD_DIMX + get_local_id(0);
#pragma unroll
		for (int _w_two = 0; _w_two < PAD_WPTY; _w_two += 1) {
			const int id_two = (get_group_id(1)*PAD_WPTY + _w_two) * PAD_DIMY + get_local_id(1);
			if (id_two < dest_dim && id_one < dest_dim) {

				// Loads data from the lower-symmetric matrix
				real result;
				SetToZero(result);
				if (id_two < src_dim && id_one < src_dim) {
					if (id_two <= id_one) { result = src[id_two*src_ld + id_one + src_offset]; }
					else { result = src[id_one*src_ld + id_two + src_offset]; }
				}

				// Stores the result in the destination matrix
				dest[id_two*dest_ld + id_one + dest_offset] = result;
			}
		}
	}
}

// Same as above, but now the matrix' data is stored in the upper-triangle
__kernel void SymmUpperToSquared(const int src_dim,
	const int src_ld, const int src_offset,
	__global const real* restrict src,
	const int dest_dim,
	const int dest_ld, const int dest_offset,
	__global real* dest) {

	// Loops over the work per thread in both dimensions
#pragma unroll
	for (int _w_one = 0; _w_one < PAD_WPTX; _w_one += 1) {
		const int id_one = (get_group_id(0)*PAD_WPTX + _w_one) * PAD_DIMX + get_local_id(0);
#pragma unroll
		for (int _w_two = 0; _w_two < PAD_WPTY; _w_two += 1) {
			const int id_two = (get_group_id(1)*PAD_WPTY + _w_two) * PAD_DIMY + get_local_id(1);
			if (id_two < dest_dim && id_one < dest_dim) {

				// Loads data from the upper-symmetric matrix
				real result;
				SetToZero(result);
				if (id_two < src_dim && id_one < src_dim) {
					if (id_one <= id_two) { result = src[id_two*src_ld + id_one + src_offset]; }
					else { result = src[id_one*src_ld + id_two + src_offset]; }
				}

				// Stores the result in the destination matrix
				dest[id_two*dest_ld + id_one + dest_offset] = result;
			}
		}
	}
}

#endif
// =================================================================================================

// End of the C++11 raw string literal


// =================================================================================================
#if defined(ROUTINE_TRMM)

// Kernel to populate a squared triangular matrix, given that the triangle which holds the data is
// stored as the lower-triangle of the input matrix. This uses the padding kernel's parameters.
__kernel void TriaLowerToSquared(const int src_dim,
	const int src_ld, const int src_offset,
	__global const real* restrict src,
	const int dest_dim,
	const int dest_ld, const int dest_offset,
	__global real* dest,
	const int unit_diagonal) {

	// Loops over the work per thread in both dimensions
#pragma unroll
	for (int _w_one = 0; _w_one < PAD_WPTX; _w_one += 1) {
		const int id_one = (get_group_id(0)*PAD_WPTX + _w_one) * PAD_DIMX + get_local_id(0);
#pragma unroll
		for (int _w_two = 0; _w_two < PAD_WPTY; _w_two += 1) {
			const int id_two = (get_group_id(1)*PAD_WPTY + _w_two) * PAD_DIMY + get_local_id(1);
			if (id_two < dest_dim && id_one < dest_dim) {

				// Loads data from the lower-triangular matrix
				real result;
				SetToZero(result);
				if (id_two < src_dim && id_one < src_dim) {
					if (id_two <= id_one) { result = src[id_two*src_ld + id_one + src_offset]; }
					if (id_two == id_one && unit_diagonal) { SetToOne(result); }
					// Else: result is zero
				}

				// Stores the result in the destination matrix
				dest[id_two*dest_ld + id_one + dest_offset] = result;
			}
		}
	}
}

// Same as above, but now the matrix' data is stored in the upper-triangle
__kernel void TriaUpperToSquared(const int src_dim,
	const int src_ld, const int src_offset,
	__global const real* restrict src,
	const int dest_dim,
	const int dest_ld, const int dest_offset,
	__global real* dest,
	const int unit_diagonal) {

	// Loops over the work per thread in both dimensions
#pragma unroll
	for (int _w_one = 0; _w_one < PAD_WPTX; _w_one += 1) {
		const int id_one = (get_group_id(0)*PAD_WPTX + _w_one) * PAD_DIMX + get_local_id(0);
#pragma unroll
		for (int _w_two = 0; _w_two < PAD_WPTY; _w_two += 1) {
			const int id_two = (get_group_id(1)*PAD_WPTY + _w_two) * PAD_DIMY + get_local_id(1);
			if (id_two < dest_dim && id_one < dest_dim) {

				// Loads data from the upper-triangular matrix
				real result;
				SetToZero(result);
				if (id_two < src_dim && id_one < src_dim) {
					if (id_one <= id_two) { result = src[id_two*src_ld + id_one + src_offset]; }
					if (id_one == id_two && unit_diagonal) { SetToOne(result); }
					// Else: result is zero
				}

				// Stores the result in the destination matrix
				dest[id_two*dest_ld + id_one + dest_offset] = result;
			}
		}
	}
}

#endif
// =================================================================================================

// End of the C++11 raw string literal


// =================================================================================================
#if defined(ROUTINE_HEMM)
#if PRECISION == 3232 || PRECISION == 6464

// Kernel to populate a squared hermitian matrix, given that the triangle which holds the data is
// stored as the lower-triangle of the input matrix. This uses the padding kernel's parameters.
__kernel void HermLowerToSquared(const int src_dim,
	const int src_ld, const int src_offset,
	__global const real* restrict src,
	const int dest_dim,
	const int dest_ld, const int dest_offset,
	__global real* dest) {

	// Loops over the work per thread in both dimensions
#pragma unroll
	for (int _w_one = 0; _w_one < PAD_WPTX; _w_one += 1) {
		const int id_one = (get_group_id(0)*PAD_WPTX + _w_one) * PAD_DIMX + get_local_id(0);
#pragma unroll
		for (int _w_two = 0; _w_two < PAD_WPTY; _w_two += 1) {
			const int id_two = (get_group_id(1)*PAD_WPTY + _w_two) * PAD_DIMY + get_local_id(1);
			if (id_two < dest_dim && id_one < dest_dim) {

				// Loads data from the lower-hermitian matrix
				real result;
				SetToZero(result);
				if (id_two < src_dim && id_one < src_dim) {
					if (id_two <= id_one) {
						result = src[id_two*src_ld + id_one + src_offset];
						if (id_one == id_two) { result.y = ZERO; }
					}
					else {
						result = src[id_one*src_ld + id_two + src_offset];
						COMPLEX_CONJUGATE(result);
					}
				}

				// Stores the result in the destination matrix
				dest[id_two*dest_ld + id_one + dest_offset] = result;
			}
		}
	}
}

// Same as above, but now the matrix' data is stored in the upper-triangle
__kernel void HermUpperToSquared(const int src_dim,
	const int src_ld, const int src_offset,
	__global const real* restrict src,
	const int dest_dim,
	const int dest_ld, const int dest_offset,
	__global real* dest) {

	// Loops over the work per thread in both dimensions
#pragma unroll
	for (int _w_one = 0; _w_one < PAD_WPTX; _w_one += 1) {
		const int id_one = (get_group_id(0)*PAD_WPTX + _w_one) * PAD_DIMX + get_local_id(0);
#pragma unroll
		for (int _w_two = 0; _w_two < PAD_WPTY; _w_two += 1) {
			const int id_two = (get_group_id(1)*PAD_WPTY + _w_two) * PAD_DIMY + get_local_id(1);
			if (id_two < dest_dim && id_one < dest_dim) {

				// Loads data from the upper-hermitian matrix
				real result;
				SetToZero(result);
				if (id_two < src_dim && id_one < src_dim) {
					if (id_one <= id_two) {
						result = src[id_two*src_ld + id_one + src_offset];
						if (id_one == id_two) { result.y = ZERO; }
					}
					else {
						result = src[id_one*src_ld + id_two + src_offset];
						COMPLEX_CONJUGATE(result);
					}
				}

				// Stores the result in the destination matrix
				dest[id_two*dest_ld + id_one + dest_offset] = result;
			}
		}
	}
}

#endif
#endif
// =================================================================================================

// End of the C++11 raw string literal


// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library. Note that all parameters here have a
// suffix 'D' to denote that they are for the 'direct' version of the GEMM kernel.
#ifndef WGD
#define WGD 8      // Tile-size in dimension M, N, and K (e.g. 8, 16, 32, 64)
#endif
#ifndef MDIMCD
#define MDIMCD 8    // Threads per workgroup in M-dimension (e.g. 8, 16, 32)
#endif
#ifndef NDIMCD
#define NDIMCD 8    // Threads per workgroup in N-dimension (e.g. 8, 16, 32)
#endif
#ifndef MDIMAD
#define MDIMAD 8    // Re-shaped tile dimension of matrix A: KDIMAD * MDIMAD
#endif
#ifndef NDIMBD
#define NDIMBD 8    // Re-shaped tile dimension of matrix B: KDIMBD * NDIMBD
#endif
#ifndef KWID
#define KWID 1      // Unroll factor of the WGD loop (smaller or equal than WGD)
#endif
#ifndef VWMD
#define VWMD 1      // Vector width of matrices A and C
#endif
#ifndef VWND
#define VWND 1      // Vector width of matrix B
#endif
#ifndef PADA
#define PADA 1      // Local memory padding for matrix A
#endif
#ifndef PADB
#define PADB 1      // Local memory padding for matrix B
#endif

// Helper parameters based on the above tuning parameters
#define MWID (WGD/MDIMCD)                // Work per work-item (M-dimension)
#define NWID (WGD/NDIMCD)                // Work per work-item (N-dimension)
#define KDIMAD ((MDIMCD*NDIMCD)/(MDIMAD)) // Re-shaped tile dimension of matrix A: KDIMAD * MDIMAD
#define KDIMBD ((MDIMCD*NDIMCD)/(NDIMBD)) // Re-shaped tile dimension of matrix B: KDIMBD * NDIMBD
#define MWAD (WGD/MDIMAD)                // Amount of loads-per-thread for matrix A (M-dimension)
#define KWAD (WGD/KDIMAD)                // Amount of loads-per-thread for matrix A (K-dimension)
#define KWBD (WGD/KDIMBD)                // Amount of loads-per-thread for matrix B (K-dimension)
#define NWBD (WGD/NDIMBD)                // Amount of loads-per-thread for matrix B (N-dimension)

// =================================================================================================

// Data-widths in dimension M
#if VWMD == 1
typedef real realMD;
#elif VWMD == 2
typedef real2 realMD;
#elif VWMD == 4
typedef real4 realMD;
#elif VWMD == 8
typedef real8 realMD;
#elif VWMD == 16
typedef real16 realMD;
#endif

// Data-widths in dimension N
#if VWND == 1
typedef real realND;
#elif VWND == 2
typedef real2 realND;
#elif VWND == 4
typedef real4 realND;
#elif VWND == 8
typedef real8 realND;
#elif VWND == 16
typedef real16 realND;
#endif

// =================================================================================================

// Loads global off-chip memory into thread-private register files. This function is specific for
// loading the A input matrix.
INLINE_FUNC real GlobalToPrivateDirectA(const __global real* restrict agms, const int _mi,
	const int a_ld, const int a_offset, const int idm, const int idk,
	const int a_transpose, const int a_conjugate) {
	const int a_index = (a_transpose) ? (idm + _mi)*a_ld + idk : idk * a_ld + (idm + _mi);
	real result = agms[a_index + a_offset];
	if (a_conjugate) { COMPLEX_CONJUGATE(result); }
	return result;
}

// Same as above, but now for the B input matrix
INLINE_FUNC real GlobalToPrivateDirectB(const __global real* restrict bgms, const int _ni,
	const int b_ld, const int b_offset, const int idn, const int idk,
	const int b_transpose, const int b_conjugate) {
	const int b_index = (b_transpose) ? (idn + _ni)*b_ld + idk : idk * b_ld + (idn + _ni);
	real result = bgms[b_index + b_offset];
	if (b_conjugate) { COMPLEX_CONJUGATE(result); }
	return result;
}

// Loads global off-chip memory into thread-private register files. This function is specific for
// loading the A input matrix. This is the same as above but now includes a bounds check.
INLINE_FUNC real GlobalToPrivateCheckedA(const __global real* restrict agms, const int _mi,
	const int a_ld, const int a_offset, const int idm, const int idk,
	const int a_transpose, const int a_conjugate,
	const int kSizeM) {
	real result;
	if (idm + _mi < kSizeM) {
		const int a_index = (a_transpose) ? (idm + _mi)*a_ld + idk : idk * a_ld + (idm + _mi);
		result = agms[a_index + a_offset];
		if (a_conjugate) { COMPLEX_CONJUGATE(result); }
	}
	else {
		SetToZero(result);
	}
	return result;
}

// Same as above, but now for the B input matrix
INLINE_FUNC real GlobalToPrivateCheckedB(const __global real* restrict bgms, const int _ni,
	const int b_ld, const int b_offset, const int idn, const int idk,
	const int b_transpose, const int b_conjugate,
	const int kSizeN) {
	real result;
	if (idn + _ni < kSizeN) {
		const int b_index = (b_transpose) ? (idn + _ni)*b_ld + idk : idk * b_ld + (idn + _ni);
		result = bgms[b_index + b_offset];
		if (b_conjugate) { COMPLEX_CONJUGATE(result); }
	}
	else {
		SetToZero(result);
	}
	return result;
}

// =================================================================================================

// Caches on-chip local memory into per-thread private memory (registers). This function is specific
// for caching the A input matrix.
INLINE_FUNC real LocalToPrivateDirectA(LOCAL_PTR real* alm, const int _mi, const int kg,
	const int a_transpose) {
	const int mg = _mi + get_local_id(0)*MWID;
	const int index = (a_transpose) ? mg * (WGD + PADA) + kg : kg * (WGD + PADA) + mg;
	return alm[index];
}

// Same as above, but now for the B input matrix
INLINE_FUNC real LocalToPrivateDirectB(LOCAL_PTR real* blm, const int _ni, const int kg,
	const int b_transpose) {
	const int ng = _ni + get_local_id(1)*NWID;
	const int index = (b_transpose) ? ng * (WGD + PADB) + kg : kg * (WGD + PADB) + ng;
	return blm[index];
}

// =================================================================================================

// Merges the results in Cpm with the global array in Cgm. This also performs the multiplication
// with the constants: Cgm = alpha*A*B + beta*Cgm = alpha*Cpm + beta*Cgm
INLINE_FUNC void StoreResultsDirect(__global real* cgm, const real c_value,
	const int _mi, const int _ni, const int idm, const int idn,
	const real alpha, const real beta,
	const int c_ld, const int c_offset, const int c_transpose) {

	// Determines the destination index
	int c_index = (c_transpose) ? (idm + _mi)*c_ld + (idn + _ni) : (idn + _ni)*c_ld + (idm + _mi);

	// The final multiplication with alpha (in case beta == 0)
	real result;
	if (IsZero(beta)) {
		Multiply(result, alpha, c_value);
	}
	// The final multiplication with alpha and the addition with beta*C
	else {
		AXPBY(result, alpha, c_value, beta, cgm[c_index + c_offset]);
	}
	cgm[c_index + c_offset] = result;
}

// Merges the results in Cpm with the global array in Cgm. This also performs the multiplication
// with the constants: Cgm = alpha*A*B + beta*Cgm = alpha*Cpm + beta*Cgm
INLINE_FUNC void StoreResultsChecked(__global real* cgm, const real c_value,
	const int _mi, const int _ni, const int idm, const int idn,
	const int kSizeM, const int kSizeN,
	const real alpha, const real beta,
	const int c_ld, const int c_offset, const int c_transpose) {
	if ((idm + _mi) < kSizeM && (idn + _ni) < kSizeN) {

		// Deter_mines the destination index
		int c_index = (c_transpose) ? (idm + _mi)*c_ld + (idn + _ni) : (idn + _ni)*c_ld + (idm + _mi);

		// The final multiplication with alpha (in case beta == 0)
		real result;
		if (IsZero(beta)) {
			Multiply(result, alpha, c_value);
		}
		// The final multiplication with alpha and the addition with beta*C
		else {
			AXPBY(result, alpha, c_value, beta, cgm[c_index + c_offset]);
		}
		cgm[c_index + c_offset] = result;
	}
}

// =================================================================================================

// End of the C++11 raw string literal



#if VWMD == 1
#define IsZeroX(a) ( a== ZERO )
#elif VWMD == 2
#define IsZeroX(a) ((a.x == ZERO) && (a.y == ZERO))
#elif VWMD == 4
#define IsZeroX(a) ((a.x == ZERO) && (a.y == ZERO) && (a.z == ZERO) && (a.w == ZERO))
#elif VWMD == 8
#define IsZeroX(a) ((a.s0 == ZERO) && (a.s1 == ZERO) && (a.s2 == ZERO) && (a.s3 == ZERO) && (a.s4 == ZERO) && (a.s5 == ZERO) && (a.s6 == ZERO) && (a.s7 == ZERO))
#elif VWMD == 16
#define IsZeroX(a) ((a.s0 == ZERO) && (a.s1 == ZERO) && (a.s2 == ZERO) && (a.s3 == ZERO) && (a.s4 == ZERO) && (a.s5 == ZERO) && (a.s6 == ZERO) && (a.s7 == ZERO) && (a.s8 == ZERO) && (a.s9 == ZERO) && (a.sA == ZERO) && (a.sB == ZERO) && (a.sC == ZERO) && (a.sD == ZERO) && (a.sE == ZERO) && (a.sF == ZERO) )
#endif


// =================================================================================================

// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix.
INLINE_FUNC int GlobalToLocalDirectA(const __global realMD* restrict agm, LOCAL_PTR real* alm,
	const int a_ld, const int a_offset, const int kwg,
	const int a_transpose, const int a_conjugate) {
#if MDIMCD == MDIMAD
	const int la0 = get_local_id(0);
	const int la1 = get_local_id(1);
#else
	const int tid = get_local_id(0) + MDIMCD * get_local_id(1);
	const int la0 = tid % MDIMAD;
	const int la1 = tid / MDIMAD;
#endif

	int isAllZero = 1;

	//#pragma unroll
	for (int _mia = 0; _mia < MWAD / VWMD; _mia += 1) {
		//#pragma unroll
		for (int _kia = 0; _kia < KWAD; _kia += 1) {

			// Computes the indices for the global memory
			int mg = _mia + la0 * (MWAD / VWMD);
			int kg = _kia + la1 * KWAD;
			int idm = (a_transpose) ? mg + kwg / VWMD : mg + GetGroupID0()*(WGD / VWMD);
			int idk = (a_transpose) ? kg + GetGroupID0()*WGD : kg + kwg;

			// Loads the data from global memory into the local memory
			const realMD avec = agm[idk*(a_ld / VWMD) + idm + (a_offset / VWMD)];
			// check if all data is ZERO

			if ((avec.x == ZERO) && (avec.y == ZERO) && (avec.z == ZERO) && (avec.w == ZERO)) continue;
			isAllZero = 0;


#if VWMD == 1
			alm[kg*(WGD + PADA) + mg] = avec;
#elif VWMD == 2
			alm[kg*(WGD + PADA) + mg * VWMD + 0] = avec.x;
			alm[kg*(WGD + PADA) + mg * VWMD + 1] = avec.y;
#elif VWMD == 4
			alm[kg*(WGD + PADA) + mg * VWMD + 0] = avec.x;
			alm[kg*(WGD + PADA) + mg * VWMD + 1] = avec.y;
			alm[kg*(WGD + PADA) + mg * VWMD + 2] = avec.z;
			alm[kg*(WGD + PADA) + mg * VWMD + 3] = avec.w;
#elif VWMD == 8
			alm[kg*(WGD + PADA) + mg * VWMD + 0] = avec.s0;
			alm[kg*(WGD + PADA) + mg * VWMD + 1] = avec.s1;
			alm[kg*(WGD + PADA) + mg * VWMD + 2] = avec.s2;
			alm[kg*(WGD + PADA) + mg * VWMD + 3] = avec.s3;
			alm[kg*(WGD + PADA) + mg * VWMD + 4] = avec.s4;
			alm[kg*(WGD + PADA) + mg * VWMD + 5] = avec.s5;
			alm[kg*(WGD + PADA) + mg * VWMD + 6] = avec.s6;
			alm[kg*(WGD + PADA) + mg * VWMD + 7] = avec.s7;
#elif VWMD == 16
			alm[kg*(WGD + PADA) + mg * VWMD + 0] = avec.s0;
			alm[kg*(WGD + PADA) + mg * VWMD + 1] = avec.s1;
			alm[kg*(WGD + PADA) + mg * VWMD + 2] = avec.s2;
			alm[kg*(WGD + PADA) + mg * VWMD + 3] = avec.s3;
			alm[kg*(WGD + PADA) + mg * VWMD + 4] = avec.s4;
			alm[kg*(WGD + PADA) + mg * VWMD + 5] = avec.s5;
			alm[kg*(WGD + PADA) + mg * VWMD + 6] = avec.s6;
			alm[kg*(WGD + PADA) + mg * VWMD + 7] = avec.s7;
			alm[kg*(WGD + PADA) + mg * VWMD + 8] = avec.s8;
			alm[kg*(WGD + PADA) + mg * VWMD + 9] = avec.s9;
			alm[kg*(WGD + PADA) + mg * VWMD + 10] = avec.sA;
			alm[kg*(WGD + PADA) + mg * VWMD + 11] = avec.sB;
			alm[kg*(WGD + PADA) + mg * VWMD + 12] = avec.sC;
			alm[kg*(WGD + PADA) + mg * VWMD + 13] = avec.sD;
			alm[kg*(WGD + PADA) + mg * VWMD + 14] = avec.sE;
			alm[kg*(WGD + PADA) + mg * VWMD + 15] = avec.sF;
#endif
			if (a_conjugate) {
				for (int vm = 0; vm < VWMD; ++vm) {
					COMPLEX_CONJUGATE(alm[kg*(WGD + PADA) + mg * VWMD + vm]);
				}
			}
		}
	}

	return isAllZero;
}

// Same as above, but now for the B input matrix
INLINE_FUNC int GlobalToLocalDirectB(const __global realND* restrict bgm, LOCAL_PTR real* blm,
	const int b_ld, const int b_offset, const int kwg,
	const int b_transpose, const int b_conjugate) {
#if MDIMCD == NDIMBD
	const int lb0 = get_local_id(0);
	const int lb1 = get_local_id(1);
#else
	const int tid = get_local_id(0) + MDIMCD * get_local_id(1);
	const int lb0 = tid % NDIMBD;
	const int lb1 = tid / NDIMBD;
#endif

	int isAllZero = 1;

	// #pragma unroll
	for (int _kib = 0; _kib < KWBD; _kib += 1) {
		// #pragma unroll
		for (int _nib = 0; _nib < NWBD / VWND; _nib += 1) {

			// Computes the indices for the global memory
			int ng = _nib + lb0 * (NWBD / VWND);
			int kg = _kib + lb1 * KWBD;
			int idn = (b_transpose) ? ng + kwg / VWND : ng + GetGroupID1()*(WGD / VWND);
			int idk = (b_transpose) ? kg + GetGroupID1()*WGD : kg + kwg;

			// Loads the data from global memory into the local memory
			const realND bvec = bgm[idk*(b_ld / VWND) + idn + (b_offset / VWND)];

			if ((bvec.x == ZERO) && (bvec.y == ZERO)) continue;
			isAllZero = 0;

#if VWND == 1
			blm[kg*(WGD + PADB) + ng] = bvec;
#elif VWND == 2
			blm[kg*(WGD + PADB) + ng * VWND + 0] = bvec.x;
			blm[kg*(WGD + PADB) + ng * VWND + 1] = bvec.y;
#elif VWND == 4
			blm[kg*(WGD + PADB) + ng * VWND + 0] = bvec.x;
			blm[kg*(WGD + PADB) + ng * VWND + 1] = bvec.y;
			blm[kg*(WGD + PADB) + ng * VWND + 2] = bvec.z;
			blm[kg*(WGD + PADB) + ng * VWND + 3] = bvec.w;
#elif VWND == 8
			blm[kg*(WGD + PADB) + ng * VWND + 0] = bvec.s0;
			blm[kg*(WGD + PADB) + ng * VWND + 1] = bvec.s1;
			blm[kg*(WGD + PADB) + ng * VWND + 2] = bvec.s2;
			blm[kg*(WGD + PADB) + ng * VWND + 3] = bvec.s3;
			blm[kg*(WGD + PADB) + ng * VWND + 4] = bvec.s4;
			blm[kg*(WGD + PADB) + ng * VWND + 5] = bvec.s5;
			blm[kg*(WGD + PADB) + ng * VWND + 6] = bvec.s6;
			blm[kg*(WGD + PADB) + ng * VWND + 7] = bvec.s7;
#elif VWND == 16
			blm[kg*(WGD + PADB) + ng * VWND + 0] = bvec.s0;
			blm[kg*(WGD + PADB) + ng * VWND + 1] = bvec.s1;
			blm[kg*(WGD + PADB) + ng * VWND + 2] = bvec.s2;
			blm[kg*(WGD + PADB) + ng * VWND + 3] = bvec.s3;
			blm[kg*(WGD + PADB) + ng * VWND + 4] = bvec.s4;
			blm[kg*(WGD + PADB) + ng * VWND + 5] = bvec.s5;
			blm[kg*(WGD + PADB) + ng * VWND + 6] = bvec.s6;
			blm[kg*(WGD + PADB) + ng * VWND + 7] = bvec.s7;
			blm[kg*(WGD + PADB) + ng * VWND + 8] = bvec.s8;
			blm[kg*(WGD + PADB) + ng * VWND + 9] = bvec.s9;
			blm[kg*(WGD + PADB) + ng * VWND + 10] = bvec.sA;
			blm[kg*(WGD + PADB) + ng * VWND + 11] = bvec.sB;
			blm[kg*(WGD + PADB) + ng * VWND + 12] = bvec.sC;
			blm[kg*(WGD + PADB) + ng * VWND + 13] = bvec.sD;
			blm[kg*(WGD + PADB) + ng * VWND + 14] = bvec.sE;
			blm[kg*(WGD + PADB) + ng * VWND + 15] = bvec.sF;
#endif
			if (b_conjugate) {
#pragma unroll
				for (int _vn = 0; _vn < VWND; _vn += 1) {
					COMPLEX_CONJUGATE(blm[kg*(WGD + PADB) + ng * VWND + _vn]);
				}
			}
		}
	}

	return isAllZero;
}

// =================================================================================================

// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix. In contrast to the functions above, this function performs doesn't
// use the vector data-types.
INLINE_FUNC int GlobalToLocalScalarA(const __global real* restrict agms, LOCAL_PTR real* alm,
	const int a_ld, const int a_offset, const int kwg,
	const int a_transpose, const int a_conjugate) {
#if MDIMCD == MDIMAD
	const int la0 = get_local_id(0);
	const int la1 = get_local_id(1);
#else
	const int tid = get_local_id(0) + MDIMCD * get_local_id(1);
	const int la0 = tid % MDIMAD;
	const int la1 = tid / MDIMAD;


#endif

	int allIsZero = 1;

	//#pragma unroll
	for (int _mia = 0; _mia < MWAD; _mia += 1) {
		//    #pragma unroll
		for (int _kia = 0; _kia < KWAD; _kia += 1) {

			// Computes the indices for the global memory
			int mg = _mia + la0 * MWAD;
			int kg = _kia + la1 * KWAD;
			int idm = (a_transpose) ? mg + kwg : mg + GetGroupID0()*WGD;
			int idk = (a_transpose) ? kg + GetGroupID0()*WGD : kg + kwg;

			// Loads the data from global memory into the local memory
			real result = agms[idk*a_ld + idm + a_offset];

			if (result == 0)
			{
				continue;
			}
			allIsZero = 0;
			if (a_conjugate) { COMPLEX_CONJUGATE(result); }
			alm[kg*(WGD + PADA) + mg] = result;
		}
	}

	return allIsZero;
}

// Same as above, but now for the B input matrix
INLINE_FUNC int GlobalToLocalScalarB(const __global real* restrict bgms, LOCAL_PTR real* blm,
	const int b_ld, const int b_offset, const int kwg,
	const int b_transpose, const int b_conjugate) {
#if MDIMCD == NDIMBD
	const int lb0 = get_local_id(0);
	const int lb1 = get_local_id(1);
#else
	const int tid = get_local_id(0) + MDIMCD * get_local_id(1);
	const int lb0 = tid % NDIMBD;
	const int lb1 = tid / NDIMBD;
#endif

	int allIsZero = 1;
	// #pragma unroll
	for (int _kib = 0; _kib < KWBD; _kib += 1) {
		//  #pragma unroll
		for (int _nib = 0; _nib < NWBD; _nib += 1) {

			// Computes the indices for the global memory
			int ng = _nib + lb0 * NWBD;
			int kg = _kib + lb1 * KWBD;
			int idn = (b_transpose) ? ng + kwg : ng + GetGroupID1()*WGD;
			int idk = (b_transpose) ? kg + GetGroupID1()*WGD : kg + kwg;

			// Loads the data from global memory into the local memory
			real result = bgms[idk*b_ld + idn + b_offset];

			if (result == 0)
			{
				continue;
			}
			allIsZero = 1;
			if (b_conjugate) { COMPLEX_CONJUGATE(result); }
			blm[kg*(WGD + PADB) + ng] = result;
		}
	}

	return allIsZero;
}

// =================================================================================================

// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix. In contrast to the functions above, this function performs bounds
// checks and doesn't use the vector data-types.
INLINE_FUNC void GlobalToLocalCheckedA(const __global real* restrict agms, LOCAL_PTR real* alm,
	const int a_ld, const int a_offset, const int kwg,
	const int a_transpose, const int a_conjugate,
	const int kSizeM, const int kSizeK) {
#if MDIMCD == MDIMAD
	const int la0 = get_local_id(0);
	const int la1 = get_local_id(1);
#else
	const int tid = get_local_id(0) + MDIMCD * get_local_id(1);
	const int la0 = tid % MDIMAD;
	const int la1 = tid / MDIMAD;
#endif
#pragma unroll
	for (int _mia = 0; _mia < MWAD; _mia += 1) {
#pragma unroll
		for (int _kia = 0; _kia < KWAD; _kia += 1) {

			// Computes the indices for the global memory
			int mg = _mia + la0 * MWAD;
			int kg = _kia + la1 * KWAD;
			int idm = (a_transpose) ? mg + kwg : mg + GetGroupID0()*WGD;
			int idk = (a_transpose) ? kg + GetGroupID0()*WGD : kg + kwg;

			// Loads the data from global memory into the local memory
			int condition = (a_transpose) ? (idm < kSizeK) && (idk < kSizeM) :
				(idm < kSizeM) && (idk < kSizeK);
			if (condition) {
				real result = agms[idk*a_ld + idm + a_offset];
				if (a_conjugate) { COMPLEX_CONJUGATE(result); }
				alm[kg*(WGD + PADA) + mg] = result;
			}
			else {
				SetToZero(alm[kg*(WGD + PADA) + mg]);
			}
		}
	}
}

// Same as above, but now for the B input matrix
INLINE_FUNC void GlobalToLocalCheckedB(const __global real* restrict bgms, LOCAL_PTR real* blm,
	const int b_ld, const int b_offset, const int kwg,
	const int b_transpose, const int b_conjugate,
	const int kSizeN, const int kSizeK) {
#if MDIMCD == NDIMBD
	const int lb0 = get_local_id(0);
	const int lb1 = get_local_id(1);
#else
	const int tid = get_local_id(0) + MDIMCD * get_local_id(1);
	const int lb0 = tid % NDIMBD;
	const int lb1 = tid / NDIMBD;
#endif
#pragma unroll
	for (int _kib = 0; _kib < KWBD; _kib += 1) {
#pragma unroll
		for (int _nib = 0; _nib < NWBD; _nib += 1) {

			// Computes the indices for the global memory
			int ng = _nib + lb0 * NWBD;
			int kg = _kib + lb1 * KWBD;
			int idn = (b_transpose) ? ng + kwg : ng + GetGroupID1()*WGD;
			int idk = (b_transpose) ? kg + GetGroupID1()*WGD : kg + kwg;

			// Loads the data from global memory into the local memory
			int condition = (b_transpose) ? (idn < kSizeK) && (idk < kSizeN) :
				(idn < kSizeN) && (idk < kSizeK);
			if (condition) {
				real result = bgms[idk*b_ld + idn + b_offset];
				if (b_conjugate) { COMPLEX_CONJUGATE(result); }
				blm[kg*(WGD + PADB) + ng] = result;
			}
			else {
				SetToZero(blm[kg*(WGD + PADB) + ng]);
			}
		}
	}
}

// =================================================================================================

// End of the C++11 raw string literal


// =================================================================================================

// Main body of the kernel. This is the direct version without pre/post processing and restrictions.
INLINE_FUNC void XgemmDirect(const int kSizeM, const int kSizeN, const int kSizeK,
	const real_arg arg_alpha,
	const real_arg arg_beta,
	const __global realMD* restrict agm, const int a_offset, const int a_ld,
	const __global realND* restrict bgm, const int b_offset, const int b_ld,
	__global real* cgm, const int c_offset, const int c_ld,
	LOCAL_PTR real* alm, LOCAL_PTR real* blm,
	const int a_transpose, const int b_transpose, const int c_transpose,
	const int a_conjugate, const int b_conjugate, const int checkZero) {
	const real alpha = GetRealArg(arg_alpha);
	const real beta = GetRealArg(arg_beta);

	// Extra pointers to scalar versions of global memory
	const __global real* restrict agms = (const __global real* restrict) agm;
	const __global real* restrict bgms = (const __global real* restrict) bgm;

	// Allocates workitem-private memory (registers)
#pragma promote_to_registers
	real apd[MWID];
#pragma promote_to_registers
	real bpd[NWID];
#pragma promote_to_registers
	real cpd[NWID * MWID];

	// Initializes the accumulation registers
#pragma unroll
	for (int _mi = 0; _mi < MWID; _mi += 1) {
#pragma unroll
		for (int _ni = 0; _ni < NWID; _ni += 1) {
			SetToZero(cpd[_ni * MWID + _mi]);
		}
	}

	// The faster version of GEMM is not allowed on the (incomplete) borders. Therefore, this section
	// processes only the main parts: output blocks of WGD by WGD.
	const int idm = get_local_id(0) * MWID + GetGroupID0() * WGD;
	const int idn = get_local_id(1) * NWID + GetGroupID1() * WGD;
	if ((idm < (kSizeM / WGD)*WGD) && (idn < (kSizeN / WGD)*WGD)) {

		// Loops over all complete workgroup tiles (K-dimension)
		int kwg = 0;

		int blockAIsZero = 0;
		int blockBIsZero = 0;

		for (; kwg < (kSizeK / WGD) * WGD; kwg += WGD) {

			// Loads data: off-chip --> local (matrix A and B)
			if (a_ld % VWMD == 0 && a_offset % VWMD == 0) {
				blockAIsZero = GlobalToLocalDirectA(agm, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate);
			}
			else {
				blockAIsZero = GlobalToLocalScalarA(agms, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate);
			}

			if (b_ld % VWND == 0 && b_offset % VWND == 0) {
				blockBIsZero = GlobalToLocalDirectB(bgm, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate);
			}
			else {
				blockAIsZero = GlobalToLocalScalarB(bgms, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate);
			}
			barrier(CLK_LOCAL_MEM_FENCE);

			if (!checkZero && blockBIsZero) continue;

			// Loops over all workitem tiles, unrolled by a factor KWID
			for (int pwi = 0; pwi < WGD; pwi += KWID)
			{
#pragma unroll
				for (int _pit = 0; _pit < KWID; _pit += 1) {
					int kg = pwi + _pit;

					// Loads data: local --> private (matrix A and B)
#pragma unroll
					for (int _mi = 0; _mi < MWID; _mi += 1) {
						apd[_mi] = LocalToPrivateDirectA(alm, _mi, kg, a_transpose);
					}
#pragma unroll
					for (int _ni = 0; _ni < NWID; _ni += 1) {
						bpd[_ni] = LocalToPrivateDirectB(blm, _ni, kg, b_transpose);
					}

					// Performs the accumulation (Cpmd += Apmd * Bpmd)
#pragma unroll
					for (int _ni = 0; _ni < NWID; _ni += 1) {
#pragma unroll
						for (int _mi = 0; _mi < MWID; _mi += 1) {
							MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
						}
					}
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE);

		}

		// Loop over the remaining part (incomplete tile in K-dimension)
		for (; kwg < kSizeK; ++kwg) {

			// Loads data: off-chip --> private (matrix A and B)
#pragma unroll
			for (int _mi = 0; _mi < MWID; _mi += 1) {
				apd[_mi] = GlobalToPrivateDirectA(agms, _mi, a_ld, a_offset, idm, kwg, a_transpose, a_conjugate);
			}
#pragma unroll
			for (int _ni = 0; _ni < NWID; _ni += 1) {
				bpd[_ni] = GlobalToPrivateDirectB(bgms, _ni, b_ld, b_offset, idn, kwg, b_transpose, b_conjugate);
			}

			// Performs the accumulation (Cpmd += Apmd * Bpmd)
#pragma unroll
			for (int _ni = 0; _ni < NWID; _ni += 1) {
#pragma unroll
				for (int _mi = 0; _mi < MWID; _mi += 1) {
					MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
				}
			}
		}

		// Stores a tile of results and performs the multiplication with alpha and beta
#pragma unroll
		for (int _ni = 0; _ni < NWID; _ni += 1) {
#pragma unroll
			for (int _mi = 0; _mi < MWID; _mi += 1) {
				StoreResultsDirect(cgm, cpd[_ni * MWID + _mi], _mi, _ni, idm, idn,
					alpha, beta, c_ld, c_offset, c_transpose);
			}
		}
	}

	// Simple but slower version for the parts on the edge (incomplete tiles in M and N-dimensions)
	else {

		// Loops over all complete workgroup tiles (K-dimension)
		int kwg = 0;
		for (; kwg < (kSizeK / WGD) * WGD; kwg += WGD) {

			// Loads data: off-chip --> local (matrix A and B)
			GlobalToLocalCheckedA(agms, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate, kSizeM, kSizeK);
			GlobalToLocalCheckedB(bgms, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate, kSizeN, kSizeK);
			barrier(CLK_LOCAL_MEM_FENCE);

			// Loops over all workitem tiles, unrolled by a factor KWID
			for (int pwi = 0; pwi < WGD; pwi += KWID) {
#pragma unroll
				for (int _pit = 0; _pit < KWID; _pit += 1) {
					int kg = pwi + _pit;

					// Loads data: local --> private (matrix A and B)
#pragma unroll
					for (int _mi = 0; _mi < MWID; _mi += 1) {
						apd[_mi] = LocalToPrivateDirectA(alm, _mi, kg, a_transpose);
					}
#pragma unroll
					for (int _ni = 0; _ni < NWID; _ni += 1) {
						bpd[_ni] = LocalToPrivateDirectB(blm, _ni, kg, b_transpose);
					}

					// Performs the accumulation (C += A * B)
#pragma unroll
					for (int _ni = 0; _ni < NWID; _ni += 1) {
#pragma unroll
						for (int _mi = 0; _mi < MWID; _mi += 1) {
							MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
						}
					}
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}

		// Loop over the remaining part (incomplete tile in K-dimension)
		for (; kwg < kSizeK; ++kwg) {

			// Loads data: off-chip --> private (matrix A and B)
#pragma unroll
			for (int _mi = 0; _mi < MWID; _mi += 1) {
				apd[_mi] = GlobalToPrivateCheckedA(agms, _mi, a_ld, a_offset, idm, kwg, a_transpose, a_conjugate, kSizeM);
			}
#pragma unroll
			for (int _ni = 0; _ni < NWID; _ni += 1) {
				bpd[_ni] = GlobalToPrivateCheckedB(bgms, _ni, b_ld, b_offset, idn, kwg, b_transpose, b_conjugate, kSizeN);
			}

			// Performs the accumulation (C += A * B)
#pragma unroll
			for (int _ni = 0; _ni < NWID; _ni += 1) {
#pragma unroll
				for (int _mi = 0; _mi < MWID; _mi += 1) {
					MultiplyAdd(cpd[_ni * MWID + _mi], apd[_mi], bpd[_ni]);
				}
			}
		}

		// Stores a tile of results and performs the multiplication with alpha and beta
#pragma unroll
		for (int _ni = 0; _ni < NWID; _ni += 1) {
#pragma unroll
			for (int _mi = 0; _mi < MWID; _mi += 1) {
				StoreResultsChecked(cgm, cpd[_ni * MWID + _mi], _mi, _ni, idm, idn, kSizeM, kSizeN,
					alpha, beta, c_ld, c_offset, c_transpose);
			}
		}
	}
}

// =================================================================================================

// Direct version of the GEMM kernel with [A, B] = [non-transposed, non-transposed]
__kernel void XgemmDirectNN(const int kSizeM, const int kSizeN, const int kSizeK,
	const real_arg arg_alpha, const real_arg arg_beta,
	const __global realMD* restrict agm, const int a_offset, const int a_ld,
	const __global realND* restrict bgm, const int b_offset, const int b_ld,
	__global real* cgm, const int c_offset, const int c_ld,
	const int c_transpose, const int a_conjugate, const int b_conjugate) {
	__local real alm[WGD * (WGD + PADA)];
	__local real blm[WGD * (WGD + PADB)];
	XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
		agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
		alm, blm, 0, 0, c_transpose, a_conjugate, b_conjugate, false);
}

// Direct version of the GEMM kernel with [A, B] = [non-transposed, transposed]
__kernel void XgemmDirectNT(const int kSizeM, const int kSizeN, const int kSizeK,
	const real_arg arg_alpha, const real_arg arg_beta,
	const __global realMD* restrict agm, const int a_offset, const int a_ld,
	const __global realND* restrict bgm, const int b_offset, const int b_ld,
	__global real* cgm, const int c_offset, const int c_ld,
	const int c_transpose, const int a_conjugate, const int b_conjugate) {
	__local real alm[WGD * (WGD + PADA)];
	__local real blm[WGD * (WGD + PADB)];
	XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
		agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
		alm, blm, 0, 1, c_transpose, a_conjugate, b_conjugate, false);
}

// Direct version of the GEMM kernel with [A, B] = [transposed, non-transposed]
__kernel void XgemmDirectTN(const int kSizeM, const int kSizeN, const int kSizeK,
	const real_arg arg_alpha, const real_arg arg_beta,
	const __global realMD* restrict agm, const int a_offset, const int a_ld,
	const __global realND* restrict bgm, const int b_offset, const int b_ld,
	__global real* cgm, const int c_offset, const int c_ld,
	const int c_transpose, const int a_conjugate, const int b_conjugate, const int checkZero) {
	__local real alm[WGD * (WGD + PADA)];
	__local real blm[WGD * (WGD + PADB)];
	XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
		agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
		alm, blm, 1, 0, c_transpose, a_conjugate, b_conjugate, checkZero);
}

// Direct version of the GEMM kernel with [A, B] = [transposed, transposed]
__kernel void XgemmDirectTT(const int kSizeM, const int kSizeN, const int kSizeK,
	const real_arg arg_alpha, const real_arg arg_beta,
	const __global realMD* restrict agm, const int a_offset, const int a_ld,
	const __global realND* restrict bgm, const int b_offset, const int b_ld,
	__global real* cgm, const int c_offset, const int c_ld,
	const int c_transpose, const int a_conjugate, const int b_conjugate) {
	__local real alm[WGD * (WGD + PADA)];
	__local real blm[WGD * (WGD + PADB)];
	XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
		agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
		alm, blm, 1, 1, c_transpose, a_conjugate, b_conjugate, false);
}

// =================================================================================================

// End of the C++11 raw string literal


// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.
#ifndef GEMMK
#define GEMMK 0    // Kernel to choose: 0 regular, 1 with 2D register tiling
#endif
#ifndef MWG
#define MWG 8      // Tile-size in dimension M (e.g. 64, 128)
#endif
#ifndef NWG
#define NWG 8      // Tile-size in dimension N (e.g. 64, 128)
#endif
#ifndef KWG
#define KWG 8      // Tile-size in dimension K (e.g. 8, 16)
#endif
#ifndef MDIMC
#define MDIMC 8    // Threads per workgroup in M-dimension (e.g. 8, 16, 32)
#endif
#ifndef NDIMC
#define NDIMC 8    // Threads per workgroup in N-dimension (e.g. 8, 16, 32)
#endif
#ifndef MDIMA
#define MDIMA 8    // Re-shaped tile dimension of matrix A: KDIMA * MDIMA (kernel 0 only)
#endif
#ifndef NDIMB
#define NDIMB 8    // Re-shaped tile dimension of matrix B: KDIMB * NDIMB (kernel 0 only)
#endif
#ifndef KWI
#define KWI 1      // Unroll factor of the KWG loop (smaller or equal than KWG)
#endif
#ifndef VWM
#define VWM 1      // Vector width of matrices A and C
#endif
#ifndef VWN
#define VWN 1      // Vector width of matrix B
#endif
#ifndef STRM
#define STRM 0     // Use strided access within a thread in the M-dimension (1) or not (0) (kernel 0 only)
#endif
#ifndef STRN
#define STRN 0     // Use strided access within a thread in the N-dimension (1) or not (0) (kernel 0 only)
#endif
#ifndef SA
#define SA 0       // Use local/shared memory to cache matrix A (1) or not (0) (kernel 0 only)
#endif
#ifndef SB
#define SB 0       // Use local/shared memory to cache matrix B (1) or not (0) (kernel 0 only)
#endif
#ifndef KREG
#define KREG 1     // Amount of register tiling in second dimension, multiple of VWN (kernel 1 only)
#endif

// Helper parameters based on the above tuning parameters
#define MWI (MWG/MDIMC)               // Work per work-item (M-dimension)
#define NWI (NWG/NDIMC)               // Work per work-item (N-dimension)
#define KDIMA ((MDIMC*NDIMC)/(MDIMA)) // Re-shaped tile dimension of matrix A: KDIMA * MDIMA
#define KDIMB ((MDIMC*NDIMC)/(NDIMB)) // Re-shaped tile dimension of matrix B: KDIMB * NDIMB
#define MWA (MWG/MDIMA)               // Amount of loads-per-thread for matrix A (M-dimension)
#define KWA (KWG/KDIMA)               // Amount of loads-per-thread for matrix A (K-dimension)
#define KWB (KWG/KDIMB)               // Amount of loads-per-thread for matrix B (K-dimension)
#define NWB (NWG/NDIMB)               // Amount of loads-per-thread for matrix B (N-dimension)

// Settings
#ifndef USE_VECTOR_MAD
#define USE_VECTOR_MAD 0      // Unroll (0) or don't (1) unroll the vector MAD manually
#endif
#ifndef GLOBAL_MEM_FENCE
#define GLOBAL_MEM_FENCE 0    // Global synchronisation barrier for potential better performance
#endif

#ifndef SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA
#define SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA 0
#endif
#ifndef SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA
#define SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA 0
#endif
#ifndef SUBGROUP_SHUFFLING_INTEL
#define SUBGROUP_SHUFFLING_INTEL 0
#endif
#ifndef USE_SUBGROUP_SHUFFLING
#define USE_SUBGROUP_SHUFFLING 0     // Optionally enables subgroup shuffling for Intel GPUs
#endif

// Intel subgroups (https://www.khronos.org/registry/OpenCL/extensions/intel/cl_intel_subgroups.html)
#if USE_SUBGROUP_SHUFFLING == 1 && SUBGROUP_SHUFFLING_INTEL == 1
#pragma OPENCL EXTENSION cl_intel_subgroups: enable
#define SUBGROUP_SIZE 8              // Assumes subgroup size is always 8 on Intel GPUs
#endif

// NVIDIA warps as subgroups using inline PTX (https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html)
#if USE_SUBGROUP_SHUFFLING == 1
#if SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA == 1 || SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA == 1
#define SUBGROUP_SIZE 32            // Assumes subgroup size is always 32 on NVIDIA GPUs
#endif
#endif

#if NWI != SUBGROUP_SIZE || MDIMC < SUBGROUP_SIZE
#undef USE_SUBGROUP_SHUFFLING
#define USE_SUBGROUP_SHUFFLING 0     // Disables subgroups in case the assumptions don't hold
#endif

// =================================================================================================

// Data-widths in dimension M
#if VWM == 1
typedef real realM;
#elif VWM == 2
typedef real2 realM;
#elif VWM == 4
typedef real4 realM;
#elif VWM == 8
typedef real8 realM;
#elif VWM == 16
typedef real16 realM;
#endif

// Data-widths in dimension N
#if VWN == 1
typedef real realN;
#elif VWN == 2
typedef real2 realN;
#elif VWN == 4
typedef real4 realN;
#elif VWN == 8
typedef real8 realN;
#elif VWN == 16
typedef real16 realN;
#endif

// =================================================================================================

// Initializes the accumulation registers to zero
INLINE_FUNC realM InitAccRegisters() {
	realM result;
#if VWM == 1
	SetToZero(result);
#elif VWM == 2
	SetToZero(result.x);
	SetToZero(result.y);
#elif VWM == 4
	SetToZero(result.x);
	SetToZero(result.y);
	SetToZero(result.z);
	SetToZero(result.w);
#elif VWM == 8
	SetToZero(result.s0);
	SetToZero(result.s1);
	SetToZero(result.s2);
	SetToZero(result.s3);
	SetToZero(result.s4);
	SetToZero(result.s5);
	SetToZero(result.s6);
	SetToZero(result.s7);
#elif VWM == 16
	SetToZero(result.s0);
	SetToZero(result.s1);
	SetToZero(result.s2);
	SetToZero(result.s3);
	SetToZero(result.s4);
	SetToZero(result.s5);
	SetToZero(result.s6);
	SetToZero(result.s7);
	SetToZero(result.s8);
	SetToZero(result.s9);
	SetToZero(result.sA);
	SetToZero(result.sB);
	SetToZero(result.sC);
	SetToZero(result.sD);
	SetToZero(result.sE);
	SetToZero(result.sF);
#endif
	return result;
}

// =================================================================================================

// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix.
#if SA == 1
INLINE_FUNC void GlobalToLocalA(const __global realM* restrict agm, LOCAL_PTR realM* alm,
	const int kSizeM, const int tid, const int kwg) {
	const int la0 = tid % MDIMA;
	const int la1 = tid / MDIMA;
#pragma unroll
	for (int _mia = 0; _mia < MWA / VWM; _mia += 1) {
#pragma unroll
		for (int _kia = 0; _kia < KWA; _kia += 1) {

			// Computes the indices based on strided/non-strided access
#if STRM == 0
			int mg = _mia + la0 * (MWA / VWM);
#elif STRM == 1
			int mg = la0 + _mia * MDIMA;
#endif

			// Computes the indices for the global memory
			int kg = _kia + la1 * KWA;
			int idm = mg + GetGroupID0() * (MWG / VWM);
			int idk = kg + kwg;

			// Loads the data from global memory (not transposed) into the local memory
			alm[kg*(MWG / VWM) + mg] = agm[idk*(kSizeM / VWM) + idm];
		}
	}
}
#endif

// Same as above, but now for the B input matrix
#if SB == 1
INLINE_FUNC void GlobalToLocalB(const __global realN* restrict bgm, LOCAL_PTR realN* blm,
	const int kSizeN, const int tid, const int kwg) {
	const int lb0 = tid % NDIMB;
	const int lb1 = tid / NDIMB;
#pragma unroll
	for (int _kib = 0; _kib < KWB; _kib += 1) {
#pragma unroll
		for (int _nib = 0; _nib < NWB / VWN; _nib += 1) {

			// Computes the indices based on strided/non-strided access
#if STRN == 0
			int ng = _nib + lb0 * (NWB / VWN);
#elif STRN == 1
			int ng = lb0 + _nib * NDIMB;
#endif

			// Computes the indices for the global memory
			int kg = _kib + lb1 * KWB;
			int idn = ng + GetGroupID1() * (NWG / VWN);
			int idk = kg + kwg;

			// Loads the data from global memory (transposed) into the local memory
			blm[kg*(NWG / VWN) + ng] = bgm[idk*(kSizeN / VWN) + idn];
		}
	}
}
#endif

// =================================================================================================

// Caches global off-chip memory directly into per-thread private memory (registers). This function
// is specific for caching the A input matrix.
#if SA == 0 && GEMMK == 0
INLINE_FUNC realM GlobalToPrivateA(const __global realM* restrict agm, const int _mi,
	const int kSizeM, const int idk, const int kwg) {
	// Computes the indices based on strided/non-strided access
#if STRM == 0
	int mg = _mi + get_local_id(0)*(MWI / VWM);
#elif STRM == 1
	int mg = get_local_id(0) + _mi * MDIMC;
#endif

	// Computes the indices for the global memory
	int idm = mg + GetGroupID0() * (MWG / VWM);

	// Loads the data from global memory (not transposed) and stores into registers
	return agm[idk*(kSizeM / VWM) + idm];
}
#endif

// Same as above, but now for the B input matrix
#if SB == 0 && GEMMK == 0
INLINE_FUNC realN GlobalToPrivateB(const __global realN* restrict bgm, const int _ni,
	const int kSizeN, const int idk) {
	// Computes the indices based on strided/non-strided access
#if STRN == 0
	int ng = _ni + get_local_id(1)*(NWI / VWN);
#elif STRN == 1
	int ng = get_local_id(1) + _ni * NDIMC;
#endif

	// Computes the indices for the global memory
	int idn = ng + GetGroupID1() * (NWG / VWN);

	// Loads the data from global memory (transposed) and stores into registers
	return bgm[idk*(kSizeN / VWN) + idn];
}
#endif

// =================================================================================================

// Caches on-chip local memory into per-thread private memory (registers). This function is specific
// for caching the A input matrix.
#if SA == 1
INLINE_FUNC realM LocalToPrivateA(LOCAL_PTR realM* alm, const int _mi, const int kg) {
#if STRM == 0
	int mg = _mi + get_local_id(0)*(MWI / VWM);
#elif STRM == 1
	int mg = get_local_id(0) + _mi * MDIMC;
#endif
	return alm[kg*(MWG / VWM) + mg];
}
#endif

// Same as above, but now for the B input matrix
#if SB == 1
INLINE_FUNC realN LocalToPrivateB(LOCAL_PTR realN* blm, const int _ni, const int kg) {
#if STRN == 0
	int ng = _ni + get_local_id(1)*(NWI / VWN);
#elif STRN == 1
	int ng = get_local_id(1) + _ni * NDIMC;
#endif
	return blm[kg*(NWG / VWN) + ng];
}
#endif



// The vectorised multiply-add function
INLINE_FUNC realM MultiplyAddVector(realM cvec, const realM avec, const real bval) {
#if USE_VECTOR_MAD == 1
	cvec += avec * bval;
#else
#if VWM == 1
	MultiplyAdd(cvec, avec, bval);
#elif VWM == 2
	MultiplyAdd(cvec.x, avec.x, bval);
	MultiplyAdd(cvec.y, avec.y, bval);
#elif VWM == 4
	MultiplyAdd(cvec.x, avec.x, bval);
	MultiplyAdd(cvec.y, avec.y, bval);
	MultiplyAdd(cvec.z, avec.z, bval);
	MultiplyAdd(cvec.w, avec.w, bval);
#elif VWM == 8
	MultiplyAdd(cvec.s0, avec.s0, bval);
	MultiplyAdd(cvec.s1, avec.s1, bval);
	MultiplyAdd(cvec.s2, avec.s2, bval);
	MultiplyAdd(cvec.s3, avec.s3, bval);
	MultiplyAdd(cvec.s4, avec.s4, bval);
	MultiplyAdd(cvec.s5, avec.s5, bval);
	MultiplyAdd(cvec.s6, avec.s6, bval);
	MultiplyAdd(cvec.s7, avec.s7, bval);
#elif VWM == 16
	MultiplyAdd(cvec.s0, avec.s0, bval);
	MultiplyAdd(cvec.s1, avec.s1, bval);
	MultiplyAdd(cvec.s2, avec.s2, bval);
	MultiplyAdd(cvec.s3, avec.s3, bval);
	MultiplyAdd(cvec.s4, avec.s4, bval);
	MultiplyAdd(cvec.s5, avec.s5, bval);
	MultiplyAdd(cvec.s6, avec.s6, bval);
	MultiplyAdd(cvec.s7, avec.s7, bval);
	MultiplyAdd(cvec.s8, avec.s8, bval);
	MultiplyAdd(cvec.s9, avec.s9, bval);
	MultiplyAdd(cvec.sA, avec.sA, bval);
	MultiplyAdd(cvec.sB, avec.sB, bval);
	MultiplyAdd(cvec.sC, avec.sC, bval);
	MultiplyAdd(cvec.sD, avec.sD, bval);
	MultiplyAdd(cvec.sE, avec.sE, bval);
	MultiplyAdd(cvec.sF, avec.sF, bval);
#endif
#endif
	return cvec;
}

// =================================================================================================

// Merges the results in Cpm with the global array in Cgm. This also performs the multiplication
// with the constants: Cgm = alpha*A*B + beta*Cgm = alpha*Cpm + beta*Cgm
INLINE_FUNC void StoreResults(__global realM* cgm, realM c_value, const int _mi, const int _ni,
	const int kSizeM, const real alpha, const real beta) {
#if STRM == 0
	int mg = _mi + get_local_id(0)*(MWI / VWM);
#elif STRM == 1
	int mg = get_local_id(0) + _mi * MDIMC;
#endif
#if STRN == 0
	int ng = _ni + get_local_id(1)*NWI;
#elif STRN == 1
	int ng = _ni % VWN + get_local_id(1)*VWN + (_ni / VWN)*VWN*NDIMC;
#endif
	int idm = mg + GetGroupID0() * (MWG / VWM);
	int idn = ng + GetGroupID1() * NWG;
	int index = idn * (kSizeM / VWM) + idm;

	realM result;
	realM xval = c_value;

	// The final multiplication with alpha (in case beta == 0)
	if (IsZero(beta)) {
#if VWM == 1
		Multiply(result, alpha, xval);
#elif VWM == 2
		Multiply(result.x, alpha, xval.x);
		Multiply(result.y, alpha, xval.y);
#elif VWM == 4
		Multiply(result.x, alpha, xval.x);
		Multiply(result.y, alpha, xval.y);
		Multiply(result.z, alpha, xval.z);
		Multiply(result.w, alpha, xval.w);
#elif VWM == 8
		Multiply(result.s0, alpha, xval.s0);
		Multiply(result.s1, alpha, xval.s1);
		Multiply(result.s2, alpha, xval.s2);
		Multiply(result.s3, alpha, xval.s3);
		Multiply(result.s4, alpha, xval.s4);
		Multiply(result.s5, alpha, xval.s5);
		Multiply(result.s6, alpha, xval.s6);
		Multiply(result.s7, alpha, xval.s7);
#elif VWM == 16
		Multiply(result.s0, alpha, xval.s0);
		Multiply(result.s1, alpha, xval.s1);
		Multiply(result.s2, alpha, xval.s2);
		Multiply(result.s3, alpha, xval.s3);
		Multiply(result.s4, alpha, xval.s4);
		Multiply(result.s5, alpha, xval.s5);
		Multiply(result.s6, alpha, xval.s6);
		Multiply(result.s7, alpha, xval.s7);
		Multiply(result.s8, alpha, xval.s8);
		Multiply(result.s9, alpha, xval.s9);
		Multiply(result.sA, alpha, xval.sA);
		Multiply(result.sB, alpha, xval.sB);
		Multiply(result.sC, alpha, xval.sC);
		Multiply(result.sD, alpha, xval.sD);
		Multiply(result.sE, alpha, xval.sE);
		Multiply(result.sF, alpha, xval.sF);
#endif
	}

	// The final multiplication with alpha and the addition with beta*C
	else {
		realM yval = cgm[index];
#if VWM == 1
		AXPBY(result, alpha, xval, beta, yval);
#elif VWM == 2
		AXPBY(result.x, alpha, xval.x, beta, yval.x);
		AXPBY(result.y, alpha, xval.y, beta, yval.y);
#elif VWM == 4
		AXPBY(result.x, alpha, xval.x, beta, yval.x);
		AXPBY(result.y, alpha, xval.y, beta, yval.y);
		AXPBY(result.z, alpha, xval.z, beta, yval.z);
		AXPBY(result.w, alpha, xval.w, beta, yval.w);
#elif VWM == 8
		AXPBY(result.s0, alpha, xval.s0, beta, yval.s0);
		AXPBY(result.s1, alpha, xval.s1, beta, yval.s1);
		AXPBY(result.s2, alpha, xval.s2, beta, yval.s2);
		AXPBY(result.s3, alpha, xval.s3, beta, yval.s3);
		AXPBY(result.s4, alpha, xval.s4, beta, yval.s4);
		AXPBY(result.s5, alpha, xval.s5, beta, yval.s5);
		AXPBY(result.s6, alpha, xval.s6, beta, yval.s6);
		AXPBY(result.s7, alpha, xval.s7, beta, yval.s7);
#elif VWM == 16
		AXPBY(result.s0, alpha, xval.s0, beta, yval.s0);
		AXPBY(result.s1, alpha, xval.s1, beta, yval.s1);
		AXPBY(result.s2, alpha, xval.s2, beta, yval.s2);
		AXPBY(result.s3, alpha, xval.s3, beta, yval.s3);
		AXPBY(result.s4, alpha, xval.s4, beta, yval.s4);
		AXPBY(result.s5, alpha, xval.s5, beta, yval.s5);
		AXPBY(result.s6, alpha, xval.s6, beta, yval.s6);
		AXPBY(result.s7, alpha, xval.s7, beta, yval.s7);
		AXPBY(result.s8, alpha, xval.s8, beta, yval.s8);
		AXPBY(result.s9, alpha, xval.s9, beta, yval.s9);
		AXPBY(result.sA, alpha, xval.sA, beta, yval.sA);
		AXPBY(result.sB, alpha, xval.sB, beta, yval.sB);
		AXPBY(result.sC, alpha, xval.sC, beta, yval.sC);
		AXPBY(result.sD, alpha, xval.sD, beta, yval.sD);
		AXPBY(result.sE, alpha, xval.sE, beta, yval.sE);
		AXPBY(result.sF, alpha, xval.sF, beta, yval.sF);
#endif
	}
	cgm[index] = result;
}


