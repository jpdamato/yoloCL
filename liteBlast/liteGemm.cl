// Select a kernel
#define KERNEL 10

// Constants for kernels 1 -- 5
#define TS 32                        // The square-root of the 2D tile-size (== work-group dims)

// Constants for kernels 3, 5
#define WPT 8                        // The amount of work-per-thread, i.e. the thread-coarsening factor
#define RTS (TS/WPT)                 // The reduced tile-size in one dimension

// Constants for kernels 4, 7 -- 10
#define WIDTH 4                      // The vector-width (in number of floats)

// Constants for kernel 5
#define TSDK 16                     // The tile-size in dimension K (for kernel 5 only)
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

#define BK TSK
#define BN TSN
#define BM TSM
#define TX RTSM
#define TY RTSN


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


// Data-widths
#if WIDTH == 1
typedef float floatX;
#elif WIDTH == 2
typedef float2 floatX;
#elif WIDTH == 4
typedef float4 floatX;
#elif WIDTH == 8
typedef float8 floatX;
#endif




// =================================================================================================

// Pad the P * Q matrix with zeroes to form a P_XL * Q_XL matrix
__kernel void paddingAddZeroes(const int P, const int Q,
	const __global float* input,
	const int P_XL, const int Q_XL,
	__global float* output) {

	// Thread identifiers
	const int tx = get_group_id(0)*PADDINGX + get_local_id(0); // 0..P_XL in blocks of PADDINGX
	const int ty = get_group_id(1)*PADDINGY + get_local_id(1); // 0..Q_XL in blocks of PADDINGY

	// Local memory to fit a tile of TS*TS elements of A and B
	__local float Asub[TS][TS];
	__local float Bsub[TS][TS];

	// Initialise the accumulation register
	float acc = 0.0f;

	
	// Check whether we are within bounds of the XL matrix
	if (tx < P_XL && ty < Q_XL) {

		// Copy the input or pad a zero
		float value;
		if (tx < P && ty < Q) {
			output[ty*P_XL + tx] = input[ty*P + tx];
		}
		else {
			output[ty*P_XL + tx] = 0.0f;
		}

	}
}

// =================================================================================================

// Remove padded values from a P_XL * Q_XL matrix to form a P * Q matrix
__kernel void paddingRemoveZeroes(const int P_XL, const int Q_XL,
	const __global float* input,
	const int P, const int Q,
	__global float* output) {

	// Thread identifiers
	const int tx = get_group_id(0)*PADDINGX + get_local_id(0); // 0..P in blocks of PADDINGX
	const int ty = get_group_id(1)*PADDINGY + get_local_id(1); // 0..Q in blocks of PADDINGY

	// Only store the result if within P * Q bounds
	if (tx < P && ty < Q) {
		output[ty*P + tx] = input[ty*P_XL + tx];
	}
}


// Simple transpose kernel for a P * Q matrix
__kernel void transpose(const int P, const int Q,
	const __global float* input,
	__global float* output, global int* outC)
{

	// Thread identifiers
	const int tx = get_local_id(0);
	const int ty = get_local_id(1);
	const int ID0 = get_group_id(0)*TRANSPOSEX + tx; // 0..P
	const int ID1 = get_group_id(1)*TRANSPOSEY + ty; // 0..Q

	// Set-up the local memory for shuffling
	__local float buffer[TRANSPOSEX][TRANSPOSEY];

	// Swap the x and y coordinates to perform the rotation (coalesced)
	if (ID0 < P && ID1 < Q) 
	{
		buffer[ty][tx] = input[ID1*P + ID0];
	}

	// Synchronise all threads
	barrier(CLK_LOCAL_MEM_FENCE);

	// We don't have to swap the x and y thread indices here,
	// because that's already done in the local memory
	const int newID0 = get_group_id(1)*TRANSPOSEY + tx;
	const int newID1 = get_group_id(0)*TRANSPOSEX + ty;

	// Store the transposed result (coalesced)
	if (newID0 < Q && newID1 < P)
	{
		output[newID1*Q + newID0] = input[ID1*P + ID0];
	}
}

// With support for incomplete tiles and arbitrary input/output matrix sizes
__kernel void myGEMM10(const int M, const int N, const int K,
	const __global floatX* A,
	const __global floatX* B,
	__global float* C) {

	// Thread identifiers
	const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM == RTSM)
	const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN == RTSN)
	const int gidm = get_group_id(0); // Work-group ID
	const int gidn = get_group_id(1); // Work-group ID
	const int tid = tidn * RTSM + tidm; // Global thread ID (max RTSM*RTSN)

	// Local memory to fit two tiles of A and B
	__local float Asub[2][TSK*TSM];
	__local float Bsub[2][TSK*TSN];

	// Allocate register space
	float Areg;
	float Breg[WPTN];
	float acc[WPTM][WPTN];

	// Initialise the accumulation registers
#pragma unroll
	for (int wm = 0; wm < WPTM; wm++) {
#pragma unroll
		for (int wn = 0; wn < WPTN; wn++) {
			acc[wm][wn] = 0.0f;
		}
	}

	// Tile A
#pragma unroll
	for (int la = 0; la < LPTA / WIDTH; la++) 
	{
		int id = la * RTSN*RTSM + tid;
		int row = MOD2(id, TSM / WIDTH);
		int col = DIV2(id, TSM / WIDTH);

		// Load the value (wide vector load)
		int tiledIndex = TSK * 0 + col;
		int indexA = tiledIndex * (M / WIDTH) + gidm * (TSM / WIDTH) + row;

		floatX vecA = A[indexA];

		// Store the loaded vector into local memory
		Asub[0][col*TSM + WIDTH * row + 0] = vecA.x;
		Asub[0][col*TSM + WIDTH * row + 1] = vecA.y;
		Asub[0][col*TSM + WIDTH * row + 2] = vecA.z;
		Asub[0][col*TSM + WIDTH * row + 3] = vecA.w;
	}

	// Tile B
#pragma unroll
	for (int lb = 0; lb < LPTB / WIDTH; lb++) 
	{
		int id = lb * RTSN*RTSM + tid;
		int row = MOD2(id, TSN / WIDTH);
		int col = DIV2(id, TSN / WIDTH);

		// Load the value (wide vector load)
		int tiledIndex = TSK * 0 + col;
		int indexB = tiledIndex * (N / WIDTH) + gidn * (TSN / WIDTH) + row;

		floatX vecB = B[indexB];

		// Store the loaded vector into local memory
		Bsub[0][col*TSN + WIDTH * row + 0] = vecB.x;
		Bsub[0][col*TSN + WIDTH * row + 1] = vecB.y;
		Bsub[0][col*TSN + WIDTH * row + 2] = vecB.z;
		Bsub[0][col*TSN + WIDTH * row + 3] = vecB.w;
	}

	// Loop over all tiles
	const int numTiles = K / TSK;
	int t = 0;
	do {

		// Synchronise
		barrier(CLK_LOCAL_MEM_FENCE);

		// Load the next tile of A and B into local memory
		int tt = t + 1;
		if (tt < numTiles) 
		{
			// Tile A
#pragma unroll
			for (int la = 0; la < LPTA / WIDTH; la++) 
			{
				int id = la * RTSN*RTSM + tid;
				int row = MOD2(id, TSM / WIDTH);
				int col = DIV2(id, TSM / WIDTH);

				// Load the value (wide vector load)
				int tiledIndex = TSK * tt + col;
				int indexA = tiledIndex * (M / WIDTH) + gidm * (TSM / WIDTH) + row;

				floatX vecA = A[indexA];

				// Store the loaded vector into local memory
				Asub[tt % 2][col*TSM + WIDTH * row + 0] = vecA.x;
				Asub[tt % 2][col*TSM + WIDTH * row + 1] = vecA.y;
				Asub[tt % 2][col*TSM + WIDTH * row + 2] = vecA.z;
				Asub[tt % 2][col*TSM + WIDTH * row + 3] = vecA.w;
			}

			// Tile B
#pragma unroll
			for (int lb = 0; lb < LPTB / WIDTH; lb++) 
			{
				int id = lb * RTSN*RTSM + tid;
				int row = MOD2(id, TSN / WIDTH);
				int col = DIV2(id, TSN / WIDTH);

				// Load the value (wide vector load)
				int tiledIndex = TSK * tt + col;
				int indexB = tiledIndex * (N / WIDTH) + gidn * (TSN / WIDTH) + row;

				floatX vecB = B[indexB];

				// Store the loaded vector into local memory
				Bsub[tt % 2][col*TSN + WIDTH * row + 0] = vecB.x;
				Bsub[tt % 2][col*TSN + WIDTH * row + 1] = vecB.y;
				Bsub[tt % 2][col*TSN + WIDTH * row + 2] = vecB.z;
				Bsub[tt % 2][col*TSN + WIDTH * row + 3] = vecB.w;
			}
		}

		// Loop over the values of a single tile
#pragma unroll
		for (int k = 0; k < TSK; k++) 
		{

			// Cache the values of Bsub in registers
#pragma unroll
			for (int wn = 0; wn < WPTN; wn++) 
			{
				int col = tidn + wn * RTSN;
				Breg[wn] = Bsub[t % 2][k*TSN + col];
			}

			// Perform the computation
#pragma unroll
			for (int wm = 0; wm < WPTM; wm++) 
			{
				int row = tidm + wm * RTSM;
				Areg = Asub[t % 2][k*TSM + row];
#pragma unroll
				for (int wn = 0; wn < WPTN; wn++) 
				{
					acc[wm][wn] += Areg * Breg[wn];
				}
			}
		}

		// Next tile
		t++;
	} while (t < numTiles);

	// Store the final results in C
	float maxF = -100000000.0f;
	for (int wm = 0; wm < WPTM; wm++)
	{
		int globalRow = gidm * TSM + tidm + wm * RTSM;
#pragma unroll
		for (int wn = 0; wn < WPTN; wn++)
		{
			int globalCol = gidn * TSN + tidn + wn * RTSN;
			C[globalCol*M + globalRow] = acc[wm][wn];
			maxF = max(maxF, acc[wm][wn]); 
		}
	}

}

