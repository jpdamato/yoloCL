
#define OPENCL_GPU 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#if defined OPENCL_GPU
typedef global int* Int_ptr;
typedef global char* Char_ptr;
typedef global uchar* Uchar_ptr;
typedef global short* Short_PTR;
typedef global unsigned int* Uint_ptr;
typedef global float* Float_ptr;

#else
typedef unsigned int* Uint_ptr;
typedef int* Int_ptr;
typedef char* Char_ptr;
typedef uchar* Uchar_ptr;
typedef short* Short_PTR;
typedef float* Float_ptr;
#endif 

#ifdef OPENCL_GPU
typedef struct {
	int *leaf;
	int n;
	int *parent;
	int *child;
	int *group;
	char **name;

	int groups;
	int *group_size;
	int *group_offset;
} tree;
tree *read_tree(char *filename);

typedef enum {
	LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU
} ACTIVATION;

typedef enum {
	PNG, BMP, TGA, JPG
} IMTYPE;

typedef enum {
	MULT, ADD, SUB, DIV
} BINARY_ACTIVATION;

typedef enum {
	CONVOLUTIONAL,
	DECONVOLUTIONAL,
	CONNECTED,
	MAXPOOL,
	SOFTMAX,
	DETECTION,
	DROPOUT,
	CROP,
	ROUTE,
	COST,
	NORMALIZATION,
	AVGPOOL,
	LOCAL,
	SHORTCUT,
	ACTIVE,
	RNN,
	GRU,
	LSTM,
	CRNN,
	BATCHNORM,
	NETWORK,
	XNOR,
	REGION,
	YOLO,
	ISEG,
	REORG,
	UPSAMPLE,
	LOGXENT,
	L2NORM,
	BLANK
} LAYER_TYPE;

typedef enum {
	SSE, MASKED, L1, SEG, SMOOTH, WGAN
} COST_TYPE;

typedef struct {
	int batch;
	float learning_rate;
	float momentum;
	float decay;
	int adam;
	float B1;
	float B2;
	float eps;
	int t;
} update_args;

struct network;
typedef struct network network;

struct layer;
typedef struct layer layer;

struct layer
{
	LAYER_TYPE type;
	ACTIVATION activation;
	COST_TYPE cost_type;
	int batch_normalize;
	int shortcut;
	int batch;
	int forced;
	int flipped;
	int inputs;
	int outputs;
	int nweights;
	int nbiases;
	int extra;
	int truths;
	int h, w, c;
	int out_h, out_w, out_c;
	int n;
	int max_boxes;
	int groups;
	int size;
	int side;
	int stride;
	int reverse;
	int flatten;
	int spatial;
	int pad;
	int sqrt;
	int flip;
	int index;
	int binary;
	int xnor;
	int steps;
	int hidden;
	int truth;
	float smooth;
	float dot;
	float angle;
	float jitter;
	float saturation;
	float exposure;
	float shift;
	float ratio;
	float learning_rate_scale;
	float clip;
	int noloss;
	int softmax;
	int classes;
	int coords;
	int background;
	int rescore;
	int objectness;
	int joint;
	int noadjust;
	int reorg;
	int log;
	int tanh;
	int *mask;
	int total;

	float alpha;
	float beta;
	float kappa;

	float coord_scale;
	float object_scale;
	float noobject_scale;
	float mask_scale;
	float class_scale;
	int bias_match;
	int random;
	float ignore_thresh;
	float truth_thresh;
	float thresh;
	float focus;
	int classfix;
	int absolute;

	int onlyforward;
	int stopbackward;
	int dontload;
	int dontsave;
	int dontloadscales;
	int numload;

	float temperature;
	float probability;
	float scale;

	size_t workspace_size;
};

#endif
#define BLOCK 512




#ifdef OPENCL_GPU 
kernel
#endif
void fill_kernel(int N, float ALPHA, Float_ptr X, int INCX)
{
	int i = get_global_id(0); //(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i < N) X[i*INCX] = ALPHA;
}

#ifdef OPENCL_GPU 
kernel
#endif
void copy_kernel(int N, Float_ptr X, int OFFX, int INCX, Float_ptr Y, int OFFY, int INCY)
{
	int i = get_global_id(0); //(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i < N) Y[i*INCY + OFFY] = X[i*INCX + OFFX];
}

#ifdef OPENCL_GPU 
kernel
#endif
void mul_kernel(int N, Float_ptr X, int INCX, Float_ptr Y, int INCY)
{
	int i = get_global_id(0); //(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i < N) Y[i*INCY] *= X[i*INCX];
}


#ifdef OPENCL_GPU 
kernel
#endif
void  fast_variance_kernel(Float_ptr x, Float_ptr mean, int batch, int filters, int spatial, Float_ptr variance)
{
	const int threads = BLOCK;
#ifdef OPENCL_GPU
	local float _local[BLOCK];
#else
	float _local[BLOCK];
#endif

	int id = get_local_id(0);// threadIdx.x;
	_local[id] = 0;

	int filter = get_group_id(0);//blockIdx.x;

	int i, j;
	for (j = 0; j < batch; ++j) {
		for (i = 0; i < spatial; i += threads) {
			int index = j*spatial*filters + filter*spatial + i + id;

			_local[id] += (i + id < spatial) ? pow((x[index] - mean[filter]), 2.0f) : 0;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (id == 0) {
		variance[filter] = 0;
		for (i = 0; i < threads; ++i) {
			variance[filter] += _local[i];
		}
		variance[filter] /= (spatial * batch - 1);
	}
}


#ifdef OPENCL_GPU 
kernel
#endif
void shortcut_kernel(int size, int minw, int minh, int minc, int stride, int sample, int batch, int w1, int h1, int c1, Float_ptr add, int w2, int h2, int c2, float s1, float s2, Float_ptr out)
{
	int id = get_global_id(0);// (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (id >= size) return;
	int i = id % minw;
	id /= minw;
	int j = id % minh;
	id /= minh;
	int k = id % minc;
	id /= minc;
	int b = id % batch;

	int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
	int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
	out[out_index] = s1*out[out_index] + s2*add[add_index];
	//out[out_index] += add[add_index];
}


#ifdef OPENCL_GPU 
kernel
#endif
void upsample_kernel(uint N, Float_ptr x, int w, int h, int c, int batch, int stride, int forward, float scale, Float_ptr out)
{
	uint i = get_global_id(0);//(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i >= N) return;
	int out_index = i;
	int out_w = i % (w*stride);
	i = i / (w*stride);
	int out_h = i % (h*stride);
	i = i / (h*stride);
	int out_c = i%c;
	i = i / c;
	int b = i%batch;

	int in_w = out_w / stride;
	int in_h = out_h / stride;
	int in_c = out_c;

	int in_index = b*w*h*c + in_c*w*h + in_h*w + in_w;

	out[out_index] += scale * x[in_index];
	//BUG
	//if (forward) out[out_index] += scale * x[in_index];
	//else atomic_add(x + in_index, scale * out[out_index]);
}



#ifdef OPENCL_GPU 
kernel
#endif
void binarize_kernel(Float_ptr x, int n, Float_ptr binary)
{
	int i = get_global_id(0); // (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i >= n) return;
	binary[i] = (x[i] >= 0) ? 1 : -1;
}


#ifdef OPENCL_GPU 
kernel
#endif
void binarize_input_kernel(Float_ptr input, int n, int size, Float_ptr binary)
{
	int s = get_global_id(0); //(blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (s >= size) return;
	int i = 0;
	float mean = 0;
	for (i = 0; i < n; ++i) {
		mean += fabs(input[i*size + s]);
	}
	mean = mean / n;
	for (i = 0; i < n; ++i) {
		binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
	}
}



#ifdef OPENCL_GPU 
kernel
#endif 
void binarize_weights_kernel(Float_ptr weights, int n, int size, Float_ptr binary)
{
	int f = get_global_id(0); // (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (f >= n) return;
	int i = 0;
	float mean = 0;
	for (i = 0; i < size; ++i) {
		mean += fabs(weights[f*size + i]);
	}
	mean = mean / size;
	for (i = 0; i < size; ++i) {
		binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
		//binary[f*size + i] = weights[f*size + i];
	}
}



#ifdef OPENCL_GPU 
kernel
#endif  
void smooth_kernel(Float_ptr x, int n, int w, int h, int c, int size, float rate, Float_ptr delta)
{
	int id = get_global_id(0); // (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (id >= n) return;

	int j = id % w;
	id /= w;
	int i = id % h;
	id /= h;
	int k = id % c;
	id /= c;
	int b = id;

	int w_offset = -(size / 2.f);
	int h_offset = -(size / 2.f);

	int out_index = j + w*(i + h*(k + c*b));
	int l, m;
	for (l = 0; l < size; ++l) {
		for (m = 0; m < size; ++m) {
			int cur_h = h_offset + i + l;
			int cur_w = w_offset + j + m;
			int index = cur_w + w*(cur_h + h*(k + b*c));
			int valid = (cur_h >= 0 && cur_h < h &&
				cur_w >= 0 && cur_w < w);
			delta[out_index] += valid ? rate*(x[index] - x[out_index]) : 0;
		}
	}
}




int float_abs_compare(const void * a, const void * b)
{
	float fa = *(const float*)a;
	if (fa < 0) fa = -fa;
	float fb = *(const float*)b;
	if (fb < 0) fb = -fb;
	return (fa > fb) - (fa < fb);
}


// HAVE TO IMPLEMENT

#ifdef OPENCL_GPU 
kernel
#endif
void kernel_add_bias(Float_ptr output, Float_ptr biases, int batch, int n, int size)
{
	int ind = get_global_id(0);

	if (ind >= batch * n * size) return;

	int b = ind / (n * size);
	int i = (ind % (n * size)) / (size);
	int j = ind % size;

	output[(b*n + i)*size + j] += biases[i];
}


#ifdef OPENCL_GPU 
kernel
#endif
void kernel_scale_bias(Float_ptr output, Float_ptr scales, int batch, int n, int size)
{
	int ind = get_global_id(0);

	if (ind >= batch * n * size) return;

	int b = ind / (n * size);
	int i = (ind % (n * size)) / (size);
	int j = ind % size;

	output[(b*n + i)*size + j] *= scales[i];
}

#ifdef OPENCL_GPU 
kernel
#endif
void kernel_fill(int N, Float_ptr X, int INCX, float ALPHA)
{
	int i = get_global_id(0);
	if (i < N) X[i*INCX] = ALPHA;
}

#ifdef OPENCL_GPU 
kernel
#endif
void kernel_copy_offset(int N, Float_ptr X,  int INCX, Float_ptr Y, int INCY, int offset )
{
	int i = get_global_id(0); 
	if (i < N) Y[i*INCY+offset] = X[i*INCX];
}

#ifdef OPENCL_GPU 
kernel
#endif
void kernel_copy(int N, Float_ptr X,  int INCX, Float_ptr Y, int INCY )
{
	int i = get_global_id(0); 
	if (i < N) Y[i*INCY] = X[i*INCX];
}



float linear_activate(float x) { return x; }
float logistic_activate(float x) { return 1. / (1. + exp(-x)); }
float loggy_activate(float x) { return 2. / (1. + exp(-x)) - 1; }
float relu_activate(float x) { return x*(x>0); }
float elu_activate(float x) { return (x >= 0)*x + (x < 0)*(exp(x) - 1); }
float selu_activate(float x) { return (x >= 0)*1.0507*x + (x < 0)*1.0507*1.6732*(exp(x) - 1); }
float relie_activate(float x) { return (x>0) ? x : .01*x; }
float ramp_activate(float x) { return x*(x>0) + .1*x; }
float leaky_activate(float x) { return (x>0) ? x : .1*x; }
float tanh_activate(float x) { return (exp(2 * x) - 1) / (exp(2 * x) + 1); }

float plse_activate(float x)
{
	if (x < -4) return .01 * (x + 4);
	if (x > 4)  return .01 * (x - 4) + 1;
	return .125*x + .5;
}

float lhtan_activate(float x)
{
	if (x < 0) return .001*x;
	if (x > 1) return .001*(x - 1) + 1;
	return x;
}
float lhtan_gradient(float x)
{
	if (x > 0 && x < 1) return 1;
	return .001;
}

float hardtan_gradient(float x)
{
	if (x > -1 && x < 1) return 1;
	return 0;
}


float stair_activate(float x)
{
	int n = floor(x);
	if (n % 2 == 0) return floor(x / 2.);
	else return (x - n) + floor(x / 2.);
}
float hardtan_activate(float x)
{
	if (x < -1) return -1;
	if (x > 1) return 1;
	return x;
}



float activate(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate(x);
        case LOGISTIC:
            return logistic_activate(x);
        case LOGGY:
            return loggy_activate(x);
        case RELU:
            return relu_activate(x);
        case ELU:
            return elu_activate(x);
        case SELU:
            return selu_activate(x);
        case RELIE:
            return relie_activate(x);
        case RAMP:
            return ramp_activate(x);
        case LEAKY:
            return leaky_activate(x);
        case TANH:
            return tanh_activate(x);
        case PLSE:
            return plse_activate(x);
        case STAIR:
            return stair_activate(x);
        case HARDTAN:
            return hardtan_activate(x);
        case LHTAN:
            return lhtan_activate(x);
    }
    return 0;
}

#ifdef OPENCL_GPU 
kernel
#endif
void kernel_activate(Float_ptr x, ACTIVATION a, int n)
{
	int i = get_global_id(0);
	if (i >= n) return;
    x[i] = activate(x[i],a);
}


#ifdef OPENCL_GPU 
kernel
#endif
void kernel_normalize(Float_ptr x, Float_ptr mean, Float_ptr variance, int batch, int filters, int spatial)
{
	int ind = get_global_id(0);

	if (ind >= batch * filters * spatial) return;

	int b = ind / (filters * spatial);
	int f = (ind % (filters * spatial)) / (spatial);
	int i = ind % spatial;

	int index = b*filters*spatial + f*spatial + i;
	x[index] = (x[index] - mean[f]) / (sqrt(variance[f]) + .000001f);
}

float _im2col_get_pixel(Float_ptr im, int height, int width, int channels,
		int row, int col, int channel, int pad)
	{
		if ((row - pad) < 0 || (col - pad) < 0 ||
			(row - pad) >= height || (col - pad) >= width) return 0;
		else
			return im[(col - pad) + width*((row - pad) + height*channel)];
	}

	#ifdef OPENCL_GPU 
	kernel
#endif
void kernel_avg_pool(Float_ptr input, Float_ptr output, int h, int w, int c, int batch)
	{
		int id = get_global_id(0);

		if (id > batch * c ) return;

		int k = id % c;
		id /= c;
		int b = id;

		int out_index = k + b*c;
		output[out_index] = 0;
		for (int i = 0; i < h*w; ++i) 
			{
					int in_index = i + h*w*(k + b*c);
					output[out_index] += input[in_index];
			}
		output[out_index] /= h*w;
		
}

#ifdef OPENCL_GPU 
	kernel
#endif
		void kernel_im2col( Float_ptr data_im, Float_ptr data_col, int height, int width, int pad, int ksize, int stride, int channels)
	{
	int i = get_global_id(0);

		int height_col = (height + 2 * pad - ksize) / stride + 1;
		int width_col = (width + 2 * pad - ksize) / stride + 1;
		int channels_col = channels * ksize * ksize;

		if (i >= height_col * width_col * channels_col) return;

		int c = i / (height_col * width_col);
		int h = (i % (height_col * width_col)) / height_col;
		int w = i % width_col;

		
		if (c >= channels_col) return;
		if (w >= width_col) return;
		if (h >= height_col) return;

		
		int w_offset = c % ksize;
		int h_offset = (c / ksize) % ksize;
		int c_im = c / ksize / ksize;

		int im_row = h_offset + h * stride;
		int im_col = w_offset + w * stride;
		int col_index = (c * height_col + h) * width_col + w;
		if (col_index < height_col * channels_col * width_col)
		{
			data_col[col_index] = _im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
		}
	}
	
	
#ifdef OPENCL_GPU 
	kernel
#endif
	void kernel_maxpool(int l_out_h, int l_out_w, int l_c, int l_h, int l_w, int l_stride, int l_pad, int l_size,Float_ptr l_output, Int_ptr l_indexes,  Float_ptr net_input)
		{
			int b, i, j, k, m, n;
			int w_offset = -l_pad / 2;
			int h_offset = -l_pad / 2;

			int h = l_out_h;
			int w = l_out_w;
			int c = l_c;
			int index = get_global_id(0);

			j = index % w;
			index /= w;
			i = index % h;
			index /= h;
			k = index % c;
			index /= c;
			b = index;

			
			int out_index = j + w*(i + h*(k + c*b));
			float max = -FLT_MAX;
			int max_i = -1;
			for (n = 0; n < l_size; ++n)
			{
				for (m = 0; m < l_size; ++m)
				{
					int cur_h = h_offset + i*l_stride + n;
					int cur_w = w_offset + j*l_stride + m;
					int index = cur_w + l_w*(cur_h + l_h*(k + b*l_c));
					int valid = (cur_h >= 0 && cur_h < l_h &&
						cur_w >= 0 && cur_w < l_w);
					float val = (valid != 0) ? net_input[index] : -FLT_MAX;
					max_i = (val > max) ? index : max_i;
					max = (val > max) ? val : max;
				}
			}
			l_output[out_index] = max;
			l_indexes[out_index] = max_i;

		}
	
#ifdef OPENCL_GPU 
kernel
#endif
void kernel_upsample(Float_ptr in, int w, int h, int c, int batch, int stride, int forward, float scale, Float_ptr out)
{
	int ind = get_global_id(0);

	if (ind >= batch * c * h * stride * w * stride) return;
	int i, j, k, b;
	b = ind / (c * h * stride * w * stride);
	k = (ind % (c * h * stride * w * stride)) / (h * stride * w * stride);
	j = (ind % (h * stride * w * stride)) / (w * stride);
	i = ind % (w * stride);


	int in_index = b*w*h*c + k*w*h + (j / stride)*w + i / stride;
	int out_index = b*w*h*c*stride*stride + k*w*h*stride*stride + j*w*stride + i;
	if (forward) out[out_index] = scale*in[in_index];
	else in[in_index] += scale*out[out_index];

}

#ifdef OPENCL_GPU 
kernel
#endif
	void kernel_shortcut(int batch, int w1, int h1, int c1, Float_ptr add, int w2, int h2, int c2, float s1, float s2, Float_ptr out)
	{
		int id = get_global_id(0);

		int stride = w1 / w2;
		int sample = w2 / w1;
	
		if (stride < 1) stride = 1;
		if (sample < 1) sample = 1;
		int minw = (w1 < w2) ? w1 : w2;
		int minh = (h1 < h2) ? h1 : h2;
		int minc = (c1 < c2) ? c1 : c2;

		if (id > batch * minc * minh * minw) return;

		int i = id % minw;
		id /= minw;
		int j = id % minh;
		id /= minh;
		int k = id % minc;
		id /= minc;
		int b = id;

		int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
		int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
		out[out_index] = s1*out[out_index] + s2*add[add_index];
	}
	
	
#ifdef OPENCL_GPU 
kernel
#endif
void kernel_yolo_layerAcc(Float_ptr l_output, Int_ptr indexes, Int_ptr lengths, int lnlb)
{
	int index = get_global_id(0);

	for (int i = 0; i < lnlb; i++)
	{
		if ((index >= indexes[i]) && (index < indexes[i] + lengths[i]))
		{
			l_output[index] = logistic_activate(l_output[index]);
			return;
		}
	}

}
