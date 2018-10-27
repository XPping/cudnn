
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cfloat>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include <cudnn.h>

#include "common.h"
#include "config.cuh"
#include "layer.cuh"
#include "kernel.cuh"



///////////////////////////////////////////////////////////////////////////////////////////
// CUDNN/CUBLAS training context

struct TrainingContext
{
	cudnnHandle_t cudnnHandle;
	cublasHandle_t cublasHandle;

	cudnnTensorDescriptor_t inputTensorDesc, conv1OutputDesc, pool1OutputDesc,
		conv2OutputDesc, pool2OutputDesc, fc1OutputDesc, fc2OutputDesc;
	cudnnTensorDescriptor_t conv1BiasDesc, conv2BiasDesc;
	cudnnFilterDescriptor_t conv1FilterDesc, conv2FilterDesc;
	
	cudnnConvolutionDescriptor_t conv1OpDesc, conv2OpDesc;
	cudnnPoolingDescriptor_t pool1OpDesc, pool2OpDesc;
	cudnnConvolutionFwdAlgo_t conv1Algo, conv2Algo;   // the specifig algorithm of conv
	cudnnConvolutionBwdFilterAlgo_t conv1BwdFilterAlgo, conv2BwdFilterAlgo;
	cudnnConvolutionBwdDataAlgo_t conv2BwdAlgo;
	cudnnActivationDescriptor_t conv1Act, conv2Act, fc1Act;
	

	int m_gpuid;
	int m_batchSize;
	size_t m_workspaceSize;

	FullyConnectedLayer& ref_fc1, &ref_fc2;

	// Disable copying
	TrainingContext& operator=(const TrainingContext&) = delete;
	TrainingContext(const TrainingContext&) = delete;

	TrainingContext(int gpuid, int batch_size,
		ConvBiasLayer& conv1, MaxPoolLayer& pool1, ConvBiasLayer& conv2, MaxPoolLayer& pool2,
		FullyConnectedLayer& fc1, FullyConnectedLayer& fc2) : ref_fc1(fc1), ref_fc2(fc2), m_gpuid(gpuid)
	{
		m_batchSize = batch_size;

		// Create CUBLAS and CUDNN handles
		checkCudaErrors(cudaSetDevice(gpuid));
		checkCudaErrors(cublasCreate(&cublasHandle));
		checkCUDNN(cudnnCreate(&cudnnHandle));

		// Create tensor descriptors                                    ///// 实例化对象
		checkCUDNN(cudnnCreateTensorDescriptor(&inputTensorDesc));			// input data
		checkCUDNN(cudnnCreateTensorDescriptor(&conv1OutputDesc));          // conv1 data
		checkCUDNN(cudnnCreateTensorDescriptor(&conv1BiasDesc));      // conv1 bias data
		checkCUDNN(cudnnCreateTensorDescriptor(&pool1OutputDesc));          // pool1 data
		checkCUDNN(cudnnCreateTensorDescriptor(&conv2OutputDesc));          // conv2 data
		checkCUDNN(cudnnCreateTensorDescriptor(&conv2BiasDesc));      // conv2 bias data
		checkCUDNN(cudnnCreateTensorDescriptor(&pool2OutputDesc));          // pool2 data
		checkCUDNN(cudnnCreateTensorDescriptor(&fc1OutputDesc));            // fc1 data
		checkCUDNN(cudnnCreateTensorDescriptor(&fc2OutputDesc));            // fc2 data

		checkCUDNN(cudnnCreateActivationDescriptor(&conv1Act));  // conv1 ReLU
		checkCUDNN(cudnnCreateActivationDescriptor(&conv2Act));  // conv2 ReLU
		checkCUDNN(cudnnCreateActivationDescriptor(&fc1Act));    // fc1 ReLU

		checkCUDNN(cudnnCreateFilterDescriptor(&conv1FilterDesc));      // conv1 filter
		checkCUDNN(cudnnCreateFilterDescriptor(&conv2FilterDesc));      // conv2 filter

		checkCUDNN(cudnnCreateConvolutionDescriptor(&conv1OpDesc));       // conv1 op
		checkCUDNN(cudnnCreateConvolutionDescriptor(&conv2OpDesc));       // conv2 op

		checkCUDNN(cudnnCreatePoolingDescriptor(&pool1OpDesc));           // pool1 op
		checkCUDNN(cudnnCreatePoolingDescriptor(&pool2OpDesc));           // pool2 op


																		// Set tensor descriptor sizes                                  ///// below is: 初始化对象
		checkCUDNN(cudnnSetTensor4dDescriptor(conv1BiasDesc,          // conv1 bias
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			1, conv1.out_channels,
			1, 1));
		checkCUDNN(cudnnSetTensor4dDescriptor(conv2BiasDesc,          // conv2 bias
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			1, conv2.out_channels,
			1, 1));
		checkCUDNN(cudnnSetPooling2dDescriptor(pool1OpDesc,                // pool1  op
			CUDNN_POOLING_MAX,
			CUDNN_PROPAGATE_NAN,
			pool1.size, pool1.size,
			0, 0,
			pool1.stride, pool1.stride));
		checkCUDNN(cudnnSetPooling2dDescriptor(pool2OpDesc,                // pool2 op
			CUDNN_POOLING_MAX,
			CUDNN_PROPAGATE_NAN,
			pool1.size, pool1.size,
			0, 0,
			pool1.stride, pool1.stride));
		checkCUDNN(cudnnSetTensor4dDescriptor(pool2OutputDesc,              // pool2 data        
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			batch_size, conv2.out_channels,
			conv2.out_height / pool2.stride,
			conv2.out_width / pool2.stride));

		checkCUDNN(cudnnSetTensor4dDescriptor(fc1OutputDesc,               // fc1 data
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			batch_size, fc1.outputs, 1, 1));

		checkCUDNN(cudnnSetTensor4dDescriptor(fc2OutputDesc,               // fc2 data
			CUDNN_TENSOR_NCHW,                                         // 好奇没有初始化 input/pool1/conv1/conv2/conv-filter data and conv1/conv2 op
			CUDNN_DATA_FLOAT,                                          // 在下面 的SetFwdConvolutionTensors函数中做了
			batch_size, fc2.outputs, 1, 1));



		checkCUDNN(cudnnSetActivationDescriptor(conv1Act, CUDNN_ACTIVATION_RELU,   // 初始化为ReLU激活函数
			CUDNN_PROPAGATE_NAN, 0.01));
		checkCUDNN(cudnnSetActivationDescriptor(conv2Act, CUDNN_ACTIVATION_RELU,
			CUDNN_PROPAGATE_NAN, 0.01));
		checkCUDNN(cudnnSetActivationDescriptor(fc1Act, CUDNN_ACTIVATION_RELU,
			CUDNN_PROPAGATE_NAN, 0.01));


		// Set convolution tensor sizes and compute workspace size
		size_t workspace = 0;
		workspace = std::max(workspace,
			SetFwdConvolutionTensors(conv1, inputTensorDesc, conv1OutputDesc, conv1FilterDesc, conv1OpDesc, conv1Algo));
		workspace = std::max(workspace,
			SetBwdConvolutionTensors(inputTensorDesc, conv1OutputDesc, conv1FilterDesc, conv1OpDesc, &conv1BwdFilterAlgo, nullptr));

		workspace = std::max(workspace,
			SetFwdConvolutionTensors(conv2, pool1OutputDesc, conv2OutputDesc, conv2FilterDesc, conv2OpDesc, conv2Algo));
		workspace = std::max(workspace,
			SetBwdConvolutionTensors(pool1OutputDesc, conv2OutputDesc, conv2FilterDesc, conv2OpDesc, &conv2BwdFilterAlgo, &conv2BwdAlgo));

		// The workspace is allocated later (if necessary)
		m_workspaceSize = workspace;
	}

	~TrainingContext()
	{
		checkCudaErrors(cudaSetDevice(m_gpuid));

		checkCudaErrors(cublasDestroy(cublasHandle));
		checkCUDNN(cudnnDestroy(cudnnHandle));
		checkCUDNN(cudnnDestroyTensorDescriptor(inputTensorDesc));
		checkCUDNN(cudnnDestroyTensorDescriptor(conv1OutputDesc));
		checkCUDNN(cudnnDestroyTensorDescriptor(conv1BiasDesc));
		checkCUDNN(cudnnDestroyActivationDescriptor(conv1Act));
		checkCUDNN(cudnnDestroyTensorDescriptor(pool1OutputDesc));
		checkCUDNN(cudnnDestroyTensorDescriptor(conv2OutputDesc));
		checkCUDNN(cudnnDestroyTensorDescriptor(conv2BiasDesc));
		checkCUDNN(cudnnDestroyActivationDescriptor(conv2Act));
		checkCUDNN(cudnnDestroyTensorDescriptor(pool2OutputDesc));
		checkCUDNN(cudnnDestroyTensorDescriptor(fc1OutputDesc));
		checkCUDNN(cudnnDestroyTensorDescriptor(fc2OutputDesc));
		checkCUDNN(cudnnDestroyActivationDescriptor(fc1Act));
		checkCUDNN(cudnnDestroyFilterDescriptor(conv1FilterDesc));
		checkCUDNN(cudnnDestroyFilterDescriptor(conv2FilterDesc));
		checkCUDNN(cudnnDestroyConvolutionDescriptor(conv1OpDesc));
		checkCUDNN(cudnnDestroyConvolutionDescriptor(conv2OpDesc));
		checkCUDNN(cudnnDestroyPoolingDescriptor(pool1OpDesc));
		checkCUDNN(cudnnDestroyPoolingDescriptor(pool2OpDesc));
	}

	size_t SetFwdConvolutionTensors(ConvBiasLayer& conv, cudnnTensorDescriptor_t& srcTensorDesc, cudnnTensorDescriptor_t& dstTensorDesc,
		cudnnFilterDescriptor_t& filterDesc, cudnnConvolutionDescriptor_t& convDesc,
		cudnnConvolutionFwdAlgo_t& algo)
	{
		size_t sizeInBytes = 0;

		int n = m_batchSize;
		int c = conv.in_channels;
		int h = conv.in_height;
		int w = conv.in_width;

		checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,          // 实例化卷积的输入data
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			n, c,
			h, w));

		checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc,             // 实例化卷积的filter data
			CUDNN_DATA_FLOAT,
			CUDNN_TENSOR_NCHW,
			conv.out_channels,
			conv.in_channels,
			conv.kernel_size,
			conv.kernel_size));

#if CUDNN_MAJOR > 5
		checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,          // 实例化卷积 op
			0, 0,
			1, 1,
			1, 1,
			CUDNN_CROSS_CORRELATION,
			CUDNN_DATA_FLOAT));
#else
		checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
			0, 0,
			1, 1,
			1, 1,
			CUDNN_CROSS_CORRELATION));
#endif

		// Find dimension of convolution output
		checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc,
			srcTensorDesc,
			filterDesc,
			&n, &c, &h, &w));

		checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc,                  // 实例化卷积的输出data
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			n, c,
			h, w));
		checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,          // 根据CUDNN_CONVOLUTION_FWD_PREFER_FASTEST选择一个
			srcTensorDesc,                                                   // 合适的卷积算法，保存到algo中
			filterDesc,
			convDesc,
			dstTensorDesc,
			CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
			0,
			&algo));

		checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,     // 获得执行当前卷积的GPU memory 
			srcTensorDesc,
			filterDesc,
			convDesc,
			dstTensorDesc,
			algo,
			&sizeInBytes));

		return sizeInBytes;
	}
	// *_data is each layer's output data, others are weight of each layer
	void ForwardPropagation(float *input_data, float *conv1_data, float *conv1relu_data, float *pool1_data, float *conv2_data, 
		float *conv2relu_data, float *pool2_data, float *fc1_data, float *fc1relu_data,
		float *fc2_data, float *softmax_data,
		float *conv1_filter, float *conv1_bias,
		float *conv2_filter, float *conv2_bias,
		float *fc1_weight, float *fc1_bias,
		float *fc2_weight, float *fc2_bias, void *workspace, float *onevec)
	{
		float alpha = 1.0f, beta = 0.0f;
		checkCudaErrors(cudaSetDevice(m_gpuid));

		// Conv1 layer
		checkCUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, inputTensorDesc,
			input_data, conv1FilterDesc, conv1_filter, conv1OpDesc,
			conv1Algo, workspace, m_workspaceSize, &beta,
			conv1OutputDesc, conv1_data));
		checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, conv1BiasDesc,
			conv1_bias, &alpha, conv1OutputDesc, conv1_data));
		// ReLU activation
		checkCUDNN(cudnnActivationForward(cudnnHandle, conv1Act, &alpha,
			conv1OutputDesc, conv1_data, &beta, conv1OutputDesc, conv1relu_data));
		// Pool1 layer
		checkCUDNN(cudnnPoolingForward(cudnnHandle, pool1OpDesc, &alpha, conv1OutputDesc,
			conv1relu_data, &beta, pool1OutputDesc, pool1_data));

		// Conv2 layer
		checkCUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, pool1OutputDesc,
			pool1_data, conv2FilterDesc, conv2_filter, conv2OpDesc,
			conv2Algo, workspace, m_workspaceSize, &beta,
			conv2OutputDesc, conv2_data));
		checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, conv2BiasDesc,
			conv1_bias, &alpha, conv2OutputDesc, conv2_data));
		// ReLU activation
		checkCUDNN(cudnnActivationForward(cudnnHandle, conv2Act, &alpha,
			conv2OutputDesc, conv2_data, &beta, conv2OutputDesc, conv2relu_data));

		// Pool2 layer
		checkCUDNN(cudnnPoolingForward(cudnnHandle, pool2OpDesc, &alpha, conv2OutputDesc,
			conv2relu_data, &beta, pool2OutputDesc, pool2_data));

		// FC1 layer
		// Forward propagate neurons using weights (fc1 = pfc1'*pool2)
		checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
			ref_fc1.outputs, m_batchSize, ref_fc1.inputs,
			&alpha,
			fc1_weight, ref_fc1.inputs,
			pool2_data, ref_fc1.inputs,
			&beta,
			fc1_data, ref_fc1.outputs));
		// Add bias using GEMM's "beta" (fc1 += pfc1bias*1_vec')
		checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			ref_fc1.outputs, m_batchSize, 1,
			&alpha,
			fc1_bias, ref_fc1.outputs,
			onevec, 1,
			&alpha,
			fc1_data, ref_fc1.outputs));

		// ReLU activation
		checkCUDNN(cudnnActivationForward(cudnnHandle, fc1Act, &alpha,
			fc1OutputDesc, fc1_data, &beta, fc1OutputDesc, fc1relu_data));

		// FC2 layer
		// Forward propagate neurons using weights (fc2 = pfc2'*fc1relu)
		checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
			ref_fc2.outputs, m_batchSize, ref_fc2.inputs,
			&alpha,
			fc2_weight, ref_fc2.inputs,
			fc1relu_data, ref_fc2.inputs,
			&beta,
			fc2_data, ref_fc2.outputs));
		// Add bias using GEMM's "beta" (fc2 += pfc2bias*1_vec')
		checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			ref_fc2.outputs, m_batchSize, 1,
			&alpha,
			fc2_bias, ref_fc2.outputs,
			onevec, 1,
			&alpha,
			fc2_data, ref_fc2.outputs));

		// Softmax loss
		checkCUDNN(cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
			&alpha, fc2OutputDesc, fc2_data, &beta, fc2OutputDesc, softmax_data));
	}

	size_t SetBwdConvolutionTensors(cudnnTensorDescriptor_t& srcTensorDesc, cudnnTensorDescriptor_t& dstTensorDesc,
		cudnnFilterDescriptor_t& filterDesc, cudnnConvolutionDescriptor_t& convDesc,
		cudnnConvolutionBwdFilterAlgo_t *falgo, cudnnConvolutionBwdDataAlgo_t *dalgo)
	{
		size_t sizeInBytes = 0, tmpsize = 0;

		// If backprop filter algorithm was requested
		if (falgo)
		{
			checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
				cudnnHandle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
				CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, falgo));

			checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
				cudnnHandle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
				*falgo, &tmpsize));

			sizeInBytes = std::max(sizeInBytes, tmpsize);
		}

		// If backprop data algorithm was requested
		if (dalgo)
		{
			checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
				cudnnHandle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc,
				CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, dalgo));

			checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
				cudnnHandle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc,
				*dalgo, &tmpsize));

			sizeInBytes = std::max(sizeInBytes, tmpsize);
		}

		return sizeInBytes;
	}

	void Backpropagation(ConvBiasLayer& layer_conv1, MaxPoolLayer& layer_pool1, ConvBiasLayer& layer_conv2, MaxPoolLayer& layer_pool2,
		float *input_data, float *input_labels, float *conv1_data, float *conv1relu_data, float *pool1_data, 
		float *conv2_data, float *conv2relu_data, float *pool2_data, float *fc1_data, float *fc1relu_data,
		float *fc2_data, float *softmax_data, float *loss_diff,
		float *conv1_kernel, float *conv1_bias,
		float *conv2_kernel, float *conv2_bias,
		float *fc1_weight, float *fc1_bias,
		float *fc2_weight, float *fc2_bias,
		float *conv1_kernel_diff, float *conv1_bias_diff, float *conv1relu_diff, float *pooll_diff,
		float *conv2_kernel_diff, float *conv2_bias_dif, float *conv2_diff, float *conv2relu_diff, float *pool2_diff,
		float *fc1_weight_diff, float *fc1_bias_diff, float *fc1_diff, float *fc1relu_diff,
		float *fc2_weight_diff, float *fc2_bias_diff, float *fc2_diff,
		void *workspace, float *onevec)
	{
		float alpha = 1.0f, beta = 0.0f;

		float scalVal = 1.0f / static_cast<float>(m_batchSize);

		checkCudaErrors(cudaSetDevice(m_gpuid));

		// Initialization (using the training error function)
		checkCudaErrors(cudaMemcpyAsync(loss_diff, softmax_data, sizeof(float) * m_batchSize * ref_fc2.outputs, cudaMemcpyDeviceToDevice));

		// Softmax layer
		SoftmaxLossBackprop << <RoundUp(m_batchSize, BW), BW >> >(input_labels, ref_fc2.outputs, m_batchSize, loss_diff);

		// Accounting for batch size in SGD
		checkCudaErrors(cublasSscal(cublasHandle, ref_fc2.outputs * m_batchSize, &scalVal, loss_diff, 1));

		// FC2 layer
		// Compute derivative with respect to weights: gfc2 = (fc1relu * dfc2smax')
		checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ref_fc2.inputs, ref_fc2.outputs, m_batchSize,
			&alpha, fc1relu_data, ref_fc2.inputs, loss_diff, ref_fc2.outputs, &beta, fc2_weight_diff, ref_fc2.inputs));
		// Compute derivative with respect to bias: gfc2bias = dfc2smax * 1_vec
		checkCudaErrors(cublasSgemv(cublasHandle, CUBLAS_OP_N, ref_fc2.outputs, m_batchSize,
			&alpha, loss_diff, ref_fc2.outputs, onevec, 1, &beta, fc2_bias_diff, 1));
		// Compute derivative with respect to data (for previous layer): pfc2*dfc2smax (500x10*10xN)
		checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ref_fc2.inputs, m_batchSize, ref_fc2.outputs,
			&alpha, fc2_weight, ref_fc2.inputs, loss_diff, ref_fc2.outputs, &beta, fc2_diff, ref_fc2.inputs));

		// ReLU activation
		checkCUDNN(cudnnActivationBackward(cudnnHandle, fc1Act, &alpha,
			fc1OutputDesc, fc1relu_data, fc1OutputDesc, fc2_diff,
			fc1OutputDesc, fc1_data, &beta, fc1OutputDesc, fc1relu_diff));

		// FC1 layer
		// Compute derivative with respect to weights: gfc1 = (pool2 * dfc1relu')
		checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ref_fc1.inputs, ref_fc1.outputs, m_batchSize,
			&alpha, pool2_data, ref_fc1.inputs, fc1relu_diff, ref_fc1.outputs, &beta, fc1_weight_diff, ref_fc1.inputs));
		// Compute derivative with respect to bias: gfc1bias = dfc1relu * 1_vec
		checkCudaErrors(cublasSgemv(cublasHandle, CUBLAS_OP_N, ref_fc1.outputs, m_batchSize,
			&alpha, fc1relu_diff, ref_fc1.outputs, onevec, 1, &beta, fc1_bias_diff, 1));
		// Compute derivative with respect to data (for previous layer): pfc1*dfc1relu (800x500*500xN)
		checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ref_fc1.inputs, m_batchSize, ref_fc1.outputs,
			&alpha, fc1_weight, ref_fc1.inputs, fc1relu_diff, ref_fc1.outputs, &beta, fc1_diff, ref_fc1.inputs));


		// Pool2 layer
		checkCUDNN(cudnnPoolingBackward(cudnnHandle, pool2OpDesc, &alpha,
			pool2OutputDesc, pool2_data, pool2OutputDesc, fc1_diff,
			conv2OutputDesc, conv2relu_data, &beta, conv2OutputDesc, pool2_diff));

		// Relu activation
		checkCUDNN(cudnnActivationBackward(cudnnHandle, conv2Act, &alpha,
			conv2OutputDesc, conv2relu_data, conv2OutputDesc, pool2_diff,
			conv2OutputDesc, conv2_data, &beta, conv2OutputDesc, conv2relu_diff));

		// Conv2 layer
		checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle, &alpha, conv2OutputDesc,
			conv2relu_diff, &beta, conv2BiasDesc, conv2_bias_dif));


		checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, pool1OutputDesc,
			pool1_data, conv2OutputDesc, conv2relu_diff, conv2OpDesc,
			conv2BwdFilterAlgo, workspace, m_workspaceSize,
			&beta, conv2FilterDesc, conv2_kernel_diff));

		checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle, &alpha, conv2FilterDesc,
			conv2_kernel, conv2OutputDesc, conv2relu_diff, conv2OpDesc,
			conv2BwdAlgo, workspace, m_workspaceSize,
			&beta, pool1OutputDesc, conv2_diff));

		// Pool1 layer
		checkCUDNN(cudnnPoolingBackward(cudnnHandle, pool1OpDesc, &alpha,
			pool1OutputDesc, pool1_data, pool1OutputDesc, conv2_diff,
			conv1OutputDesc, conv1relu_data, &beta, conv1OutputDesc, pooll_diff));

		// Relu activation
		checkCUDNN(cudnnActivationBackward(cudnnHandle, conv1Act, &alpha,
			conv1OutputDesc, conv1relu_data, conv1OutputDesc, pooll_diff,
			conv1OutputDesc, conv1_data, &beta, conv1OutputDesc, conv1relu_diff));

		// Conv1 layer
		checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle, &alpha, conv1OutputDesc,
			conv1relu_diff, &beta, conv1BiasDesc, conv1_bias_diff));

		checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, inputTensorDesc,
			input_data, conv1OutputDesc, conv1relu_diff, conv1OpDesc,
			conv1BwdFilterAlgo, workspace, m_workspaceSize,
			&beta, conv1FilterDesc, conv1_kernel_diff));

		// No need for convBackwardData because there are no more layers below
	}

	void UpdateWeights(float learning_rate,
		ConvBiasLayer& conv1, ConvBiasLayer& conv2,
		float *conv1_kernel, float *conv1_bias,
		float *conv2_kernel, float *conv2_bias,
		float *fc1_weight, float *fc1_bias,
		float *fc2_weight, float *fc2_bias,
		float *conv1_kernel_diff, float *conv1_bias_diff,
		float *conv2_kernel_diff, float *conv2_bias_diff,
		float *fc1_weight_diff, float *fc1_bias_diff,
		float *fc2_weight_diff, float *fc2_bias_diff)
	{
		float alpha = -learning_rate;

		checkCudaErrors(cudaSetDevice(m_gpuid));

		// Conv1
		checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv1.kernel.size()),
			&alpha, conv1_kernel_diff, 1, conv1_kernel, 1));
		checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv1.bias.size()),
			&alpha, conv1_bias_diff, 1, conv1_bias, 1));

		// Conv2
		checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv2.kernel.size()),
			&alpha, conv2_kernel_diff, 1, conv2_kernel, 1));
		checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv2.bias.size()),
			&alpha, conv2_bias_diff, 1, conv2_bias, 1));

		// Fully connected 1
		checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.weight.size()),
			&alpha, fc1_weight_diff, 1, fc1_weight, 1));
		checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.bias.size()),
			&alpha, fc1_bias_diff, 1, fc1_bias, 1));

		// Fully connected 2
		checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc2.weight.size()),
			&alpha, fc2_weight_diff, 1, fc2_weight, 1));
		checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc2.bias.size()),
			&alpha, fc2_bias_diff, 1, fc2_bias, 1));
	}
};

