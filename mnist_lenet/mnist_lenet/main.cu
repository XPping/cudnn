
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

#include "read_mnist.h"
#include "common.h"
#include "config.cuh"
#include "layer.cuh"
#include "kernel.cuh"
#include "lenet.cuh"

///////////////////////////////////////////////////////////////////////////////////////////
// Command-line flags

// Application parameters
DEFINE_int32(gpu, 0, "The GPU ID to use");
DEFINE_int32(iterations, 1000, "Number of iterations for training");
DEFINE_int32(random_seed, -1, "Override random seed (default uses std::random_device)");
DEFINE_int32(classify, -1, "Number of images to classify to compute error rate (default uses entire test set)");

// Batch parameters
DEFINE_uint64(batch_size, 64, "Batch size for training");

// Filenames
DEFINE_bool(pretrained, false, "Use the pretrained CUDNN model as input");
DEFINE_bool(save_data, false, "Save pretrained weights to file");
DEFINE_string(train_images, "mnist dataset/train-images-idx3-ubyte", "Training images filename");
DEFINE_string(train_labels, "mnist dataset/train-labels-idx1-ubyte", "Training labels filename");
DEFINE_string(test_images, "mnist dataset/t10k-images-idx3-ubyte", "Test images filename");
DEFINE_string(test_labels, "mnist dataset/t10k-labels-idx1-ubyte", "Test labels filename");

// Solver parameters
DEFINE_double(learning_rate, 0.01, "Base learning rate");
DEFINE_double(lr_gamma, 0.0001, "Learning rate policy gamma");
DEFINE_double(lr_power, 0.75, "Learning rate policy power");



int main(int argc, char **argv)
{
#ifdef USE_GFLAGS
	gflags::ParseCommandLineFlags(&argc, &argv, true);
#endif

	size_t width, height, channels = 1;

	// Open input data
	printf("Reading input data\n");

	// Read dataset sizes
	size_t train_size = readUbyteMnist(FLAGS_train_images.c_str(), FLAGS_train_labels.c_str(), nullptr, nullptr, width, height);
	size_t test_size = readUbyteMnist(FLAGS_test_images.c_str(), FLAGS_test_labels.c_str(), nullptr, nullptr, width, height);
	if (train_size == 0)
		return 1;

	std::vector<uint8_t> train_images(train_size * width * height * channels), train_labels(train_size);
	std::vector<uint8_t> test_images(test_size * width * height * channels), test_labels(test_size);

	// Read data from datasets
	if (readUbyteMnist(FLAGS_train_images.c_str(), FLAGS_train_labels.c_str(), &train_images[0], &train_labels[0], width, height) != train_size)
		return 2;
	if (readUbyteMnist(FLAGS_test_images.c_str(), FLAGS_test_labels.c_str(), &test_images[0], &test_labels[0], width, height) != test_size)
		return 3;

	printf("Done. Training dataset size: %d, Test dataset size: %d\n", (int)train_size, (int)test_size);
	printf("Batch size: %lld, iterations: %d\n", FLAGS_batch_size, FLAGS_iterations);
	
	/*
	// This code snippet saves a random image and its label
	printf("%d, %d, %d\n", width, height, channels);
	std::random_device rd_image;
	int random_image = rd_image() % train_size;
	std::stringstream ss; ss << "image-" << (int)train_labels[random_image] << ".pgm";
	SavePGMFile(&train_images[0]+random_image*width+height, width, height, ss.str().c_str());
	*/
	// Choose GPU
	int num_gpus;
	checkCudaErrors(cudaGetDeviceCount(&num_gpus));
	if (FLAGS_gpu < 0 || FLAGS_gpu >= num_gpus)
	{
		printf("ERROR: Invalid GPU ID %d (There are %d GPUs on this machine)\n",
			FLAGS_gpu, num_gpus);
		return 4;
	}

	// Create the LeNet network architecture
	ConvBiasLayer conv1((int)channels, 20, 5, (int)width, (int)height);
	MaxPoolLayer pool1(2, 2);
	ConvBiasLayer conv2(conv1.out_channels, 50, 5, conv1.out_width / pool1.stride, conv1.out_height / pool1.stride);
	MaxPoolLayer pool2(2, 2);
	FullyConnectedLayer fc1((conv2.out_channels*conv2.out_width*conv2.out_height) / (pool2.stride * pool2.stride),
		500);
	FullyConnectedLayer fc2(fc1.outputs, 10);

	// Initialize CUDNN/CUBLAS training context
	TrainingContext context(FLAGS_gpu, FLAGS_batch_size, conv1, pool1, conv2, pool2, fc1, fc2);

	// Determine initial network structure
	bool bRet = true;
	if (FLAGS_pretrained)
	{
		bRet = conv1.FromFile("conv1");
		bRet &= conv2.FromFile("conv2");
		bRet &= fc1.FromFile("ip1");
		bRet &= fc2.FromFile("ip2");
	}
	if (!bRet || !FLAGS_pretrained)
	{
		// Create random network
		std::random_device rd;
		std::mt19937 gen(FLAGS_random_seed < 0 ? rd() : static_cast<unsigned int>(FLAGS_random_seed)); // random generate algorithm

		// Xavier weight filling
		float wconv1 = sqrt(3.0f / (conv1.kernel_size * conv1.kernel_size * conv1.in_channels));
		std::uniform_real_distribution<> dconv1(-wconv1, wconv1);
		float wconv2 = sqrt(3.0f / (conv2.kernel_size * conv2.kernel_size * conv2.in_channels));
		std::uniform_real_distribution<> dconv2(-wconv2, wconv2);
		float wfc1 = sqrt(3.0f / (fc1.inputs * fc1.outputs));
		std::uniform_real_distribution<> dfc1(-wfc1, wfc1);
		float wfc2 = sqrt(3.0f / (fc2.inputs * fc2.outputs));
		std::uniform_real_distribution<> dfc2(-wfc2, wfc2);

		// Randomize network
		for (auto&& iter : conv1.kernel)
			iter = static_cast<float>(dconv1(gen));
		for (auto&& iter : conv1.bias)
			iter = static_cast<float>(dconv1(gen));
		for (auto&& iter : conv2.kernel)
			iter = static_cast<float>(dconv2(gen));
		for (auto&& iter : conv2.bias)
			iter = static_cast<float>(dconv2(gen));
		for (auto&& iter : fc1.weight)
			iter = static_cast<float>(dfc1(gen));
		for (auto&& iter : fc1.bias)
			iter = static_cast<float>(dfc1(gen));
		for (auto&& iter : fc2.weight)
			iter = static_cast<float>(dfc2(gen));
		for (auto&& iter : fc2.bias)
			iter = static_cast<float>(dfc2(gen));
	}

	/////////////////////////////////////////////////////////////////////////////
	// Create GPU data structures    

	// Forward propagation data
	//float *d_data, *d_labels, *d_conv1, *d_pool1, *d_conv2, *d_pool2, *d_fc1, *d_fc1relu, *d_fc2, *d_fc2smax;
	float *input_data, *input_labels, *conv1_data, *conv1relu_data, 
		*pool1_data, *conv2_data, *conv2relu_data, *pool2_data, *fc1_data, *fc1relu_data, *fc2_data, *softmax_data;
	//                         Buffer    | Element       | N                   | C                  | H                                 | W
	//-----------------------------------------------------------------------------------------------------------------------------------------
	checkCudaErrors(cudaMalloc(&input_data, sizeof(float) * context.m_batchSize * channels           * height                            * width));
	checkCudaErrors(cudaMalloc(&input_labels, sizeof(float) * context.m_batchSize * 1 * 1 * 1));
	checkCudaErrors(cudaMalloc(&conv1_data, sizeof(float) * context.m_batchSize * conv1.out_channels * conv1.out_height                  * conv1.out_width));
	checkCudaErrors(cudaMalloc(&conv1relu_data, sizeof(float) * context.m_batchSize * conv1.out_channels * conv1.out_height                  * conv1.out_width));
	checkCudaErrors(cudaMalloc(&pool1_data, sizeof(float) * context.m_batchSize * conv1.out_channels * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride)));
	checkCudaErrors(cudaMalloc(&conv2_data, sizeof(float) * context.m_batchSize * conv2.out_channels * conv2.out_height                  * conv2.out_width));
	checkCudaErrors(cudaMalloc(&conv2relu_data, sizeof(float) * context.m_batchSize * conv2.out_channels * conv2.out_height                  * conv2.out_width));
	checkCudaErrors(cudaMalloc(&pool2_data, sizeof(float) * context.m_batchSize * conv2.out_channels * (conv2.out_height / pool2.stride) * (conv2.out_width / pool2.stride)));
	checkCudaErrors(cudaMalloc(&fc1_data, sizeof(float) * context.m_batchSize * fc1.outputs));
	checkCudaErrors(cudaMalloc(&fc1relu_data, sizeof(float) * context.m_batchSize * fc1.outputs));
	checkCudaErrors(cudaMalloc(&fc2_data, sizeof(float) * context.m_batchSize * fc2.outputs));
	checkCudaErrors(cudaMalloc(&softmax_data, sizeof(float) * context.m_batchSize * fc2.outputs));

	// Network parameters
	//float *d_pconv1, *d_pconv1bias, *d_pconv2, *d_pconv2bias;
	//float *d_pfc1, *d_pfc1bias, *d_pfc2, *d_pfc2bias;
	float *conv1_kernel, *conv1_bias, *conv2_kernel, *conv2_bias;
	float *fc1_weight, *fc1_bias, *fc2_weight, *fc2_bias;

	checkCudaErrors(cudaMalloc(&conv1_kernel, sizeof(float) * conv1.kernel.size()));
	checkCudaErrors(cudaMalloc(&conv1_bias, sizeof(float) * conv1.bias.size()));
	checkCudaErrors(cudaMalloc(&conv2_kernel, sizeof(float) * conv2.kernel.size()));
	checkCudaErrors(cudaMalloc(&conv2_bias, sizeof(float) * conv2.bias.size()));
	checkCudaErrors(cudaMalloc(&fc1_weight, sizeof(float) * fc1.weight.size()));
	checkCudaErrors(cudaMalloc(&fc1_bias, sizeof(float) * fc1.bias.size()));
	checkCudaErrors(cudaMalloc(&fc2_weight, sizeof(float) * fc2.weight.size()));
	checkCudaErrors(cudaMalloc(&fc2_bias, sizeof(float) * fc2.bias.size()));

	// Network parameter gradients
	//float *d_gconv1, *d_gconv1bias, *d_gconv2, *d_gconv2bias;
	//float *d_gfc1, *d_gfc1bias, *d_gfc2, *d_gfc2bias;
	float *conv1_kernel_diff, *conv1_bias_diff, *conv2_kernel_diff, *conv2_bias_diff;
	float *fc1_weight_diff, *fc1_bias_diff, *fc2_weight_diff, *fc2_bias_diff;

	checkCudaErrors(cudaMalloc(&conv1_kernel_diff, sizeof(float) * conv1.kernel.size()));
	checkCudaErrors(cudaMalloc(&conv1_bias_diff, sizeof(float) * conv1.bias.size()));
	checkCudaErrors(cudaMalloc(&conv2_kernel_diff, sizeof(float) * conv2.kernel.size()));
	checkCudaErrors(cudaMalloc(&conv2_bias_diff, sizeof(float) * conv2.bias.size()));
	checkCudaErrors(cudaMalloc(&fc1_weight_diff, sizeof(float) * fc1.weight.size()));
	checkCudaErrors(cudaMalloc(&fc1_bias_diff, sizeof(float) * fc1.bias.size()));
	checkCudaErrors(cudaMalloc(&fc2_weight_diff, sizeof(float) * fc2.weight.size()));
	checkCudaErrors(cudaMalloc(&fc2_bias_diff, sizeof(float) * fc2.bias.size()));

	// Differentials w.r.t. data
	//float *d_dpool1, *d_dpool2, *d_dconv2, *d_dfc1, *d_dfc1relu, *d_dfc2, *d_dfc2smax, *d_dlossdata;
	float *conv1relu_diff, *pool1_diff, *conv2_diff, *conv2relu_diff, *pool2_diff, *fc1_diff, *fc1relu_diff, *fc2_diff, *softmax_diff, *loss;
	//                         Buffer     | Element       | N                   | C                  | H                                 | W
	//-----------------------------------------------------------------------------------------------------------------------------------------
	checkCudaErrors(cudaMalloc(&conv1relu_diff, sizeof(float) * context.m_batchSize * conv1.out_channels * conv1.out_height                  * conv1.out_width));
	checkCudaErrors(cudaMalloc(&pool1_diff, sizeof(float) * context.m_batchSize * conv1.out_channels * conv1.out_height                  * conv1.out_width));
	checkCudaErrors(cudaMalloc(&conv2_diff, sizeof(float) * context.m_batchSize * conv1.out_channels * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride)));
	checkCudaErrors(cudaMalloc(&conv2relu_diff, sizeof(float) * context.m_batchSize * conv2.out_channels * conv2.out_height                  * conv2.out_width));
	checkCudaErrors(cudaMalloc(&pool2_diff, sizeof(float) * context.m_batchSize * conv2.out_channels * conv2.out_height                  * conv2.out_width));
	checkCudaErrors(cudaMalloc(&fc1_diff, sizeof(float) * context.m_batchSize * fc1.inputs));
	checkCudaErrors(cudaMalloc(&fc1relu_diff, sizeof(float) * context.m_batchSize * fc1.outputs));
	checkCudaErrors(cudaMalloc(&fc2_diff, sizeof(float) * context.m_batchSize * fc2.inputs));
	checkCudaErrors(cudaMalloc(&softmax_diff, sizeof(float) * context.m_batchSize * fc2.outputs));
	checkCudaErrors(cudaMalloc(&loss, sizeof(float) * context.m_batchSize * fc2.outputs));

	// Temporary buffers and workspaces
	float *d_onevec;
	void *d_cudnn_workspace = nullptr;
	checkCudaErrors(cudaMalloc(&d_onevec, sizeof(float)* context.m_batchSize));
	if (context.m_workspaceSize > 0)
		checkCudaErrors(cudaMalloc(&d_cudnn_workspace, context.m_workspaceSize));

	/////////////////////////////////////////////////////////////////////////////

	// Copy initial network to device
	checkCudaErrors(cudaMemcpyAsync(conv1_kernel, &conv1.kernel[0], sizeof(float) * conv1.kernel.size(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(conv1_bias, &conv1.bias[0], sizeof(float) * conv1.bias.size(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(conv2_kernel, &conv2.kernel[0], sizeof(float) * conv2.kernel.size(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(conv2_bias, &conv2.bias[0], sizeof(float) * conv2.bias.size(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(fc1_weight, &fc1.weight[0], sizeof(float) * fc1.weight.size(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(fc1_bias, &fc1.bias[0], sizeof(float) * fc1.bias.size(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(fc2_weight, &fc2.weight[0], sizeof(float) * fc2.weight.size(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(fc2_bias, &fc2.bias[0], sizeof(float) * fc2.bias.size(), cudaMemcpyHostToDevice));

	// Fill one-vector with ones
	FillOnes << <RoundUp(context.m_batchSize, BW), BW >> >(d_onevec, context.m_batchSize);

	printf("Preparing dataset\n");

	// Normalize training set to be in [0,1]
	std::vector<float> train_images_float(train_images.size()), train_labels_float(train_size);
	for (size_t i = 0; i < train_size * channels * width * height; ++i)
		train_images_float[i] = (float)train_images[i] / 255.0f;

	for (size_t i = 0; i < train_size; ++i)
		train_labels_float[i] = (float)train_labels[i];

	printf("Training...\n");

	// Use SGD to train the network
	checkCudaErrors(cudaDeviceSynchronize());
	auto t1 = std::chrono::high_resolution_clock::now();
	for (int iter = 0; iter < FLAGS_iterations; ++iter)
	{
		// Train
		int imageid = iter % (train_size / context.m_batchSize);

		// Prepare current batch on device
		checkCudaErrors(cudaMemcpyAsync(input_data, &train_images_float[imageid * context.m_batchSize * width*height*channels],
			sizeof(float) * context.m_batchSize * channels * width * height, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpyAsync(input_labels, &train_labels_float[imageid * context.m_batchSize],
			sizeof(float) * context.m_batchSize, cudaMemcpyHostToDevice));

		// Forward propagation
		context.ForwardPropagation(input_data, conv1_data, conv1relu_data, pool1_data, conv2_data, conv2relu_data, pool2_data, 
			fc1_data, fc1relu_data, fc2_data, softmax_data,
			conv1_kernel, conv1_bias, conv2_kernel, conv2_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias,
			d_cudnn_workspace, d_onevec);

		// Backward propagation
		context.Backpropagation(conv1, pool1, conv2, pool2,
			input_data, input_labels, conv1_data, conv1relu_data, pool1_data, conv2_data, conv2relu_data, pool2_data, 
			fc1_data, fc1relu_data, fc2_data, softmax_data, loss,
			conv1_kernel, conv1_bias, conv2_kernel, conv2_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias,
			conv1_kernel_diff, conv1_bias_diff, conv1relu_diff, pool1_diff, conv2_kernel_diff, conv2_bias_diff, conv2_diff, 
			conv2relu_diff, pool2_diff, fc1_weight_diff, fc1_bias_diff,
			fc1_diff, fc1relu_diff, fc2_weight_diff, fc2_bias_diff, fc2_diff, d_cudnn_workspace, d_onevec);
		
		// Printf train loss
		std::vector<float> softmax_vec(context.m_batchSize * fc2.outputs);
		// Copy back loss
		checkCudaErrors(cudaMemcpy(&softmax_vec[0], softmax_data, sizeof(float) * context.m_batchSize * fc2.outputs, cudaMemcpyDeviceToHost));
		const float* _label = &train_labels_float[imageid * context.m_batchSize];
		float num_errors = 0.0;
		for (int _i = 0; _i < context.m_batchSize; _i++) {
			const float* _softmax = &softmax_vec[0] + _i * fc2.outputs;
			int chosen = 0;
			for (int id = 1; id < 10; ++id) {
				if (_softmax[chosen] < _softmax[id]) chosen = id;
			}
			if (chosen != _label[_i]) ++num_errors;
		}
		printf("%d iter, train error: %f\n", iter, num_errors / context.m_batchSize);
		
		// Compute learning rate
		float learningRate = static_cast<float>(FLAGS_learning_rate * pow((1.0 + FLAGS_lr_gamma * iter), (-FLAGS_lr_power)));

		
		// Update weights
		context.UpdateWeights(learningRate, conv1, conv2,
			conv1_kernel, conv1_bias, conv2_kernel, conv2_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias,
			conv1_kernel_diff, conv1_bias_diff, conv2_kernel_diff, conv2_bias_diff, 
			fc1_weight_diff, fc1_bias_diff, fc2_weight_diff, fc2_bias_diff);
	}
	checkCudaErrors(cudaDeviceSynchronize());
	auto t2 = std::chrono::high_resolution_clock::now();

	printf("Iteration time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / FLAGS_iterations);


	// Save model
	if (FLAGS_save_data)
	{
		// Copy trained weights from GPU to CPU
		checkCudaErrors(cudaMemcpy(&conv1.kernel[0], conv1_kernel, sizeof(float) * conv1.kernel.size(), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(&conv1.bias[0], conv1_bias, sizeof(float) * conv1.bias.size(), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(&conv2.kernel[0], conv2_kernel, sizeof(float) * conv2.kernel.size(), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(&conv2.bias[0], conv2_bias, sizeof(float) * conv2.bias.size(), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(&fc1.weight[0], fc1_weight, sizeof(float) * fc1.weight.size(), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(&fc1.bias[0], fc1_bias, sizeof(float) * fc1.bias.size(), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(&fc2.weight[0], fc2_weight, sizeof(float) * fc2.weight.size(), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(&fc2.bias[0], fc2_bias, sizeof(float) * fc2.bias.size(), cudaMemcpyDeviceToHost));

		// Now save data
		printf("Saving data to file\n");
		conv1.ToFile("conv1");
		conv2.ToFile("conv2");
		fc1.ToFile("ip1");
		fc2.ToFile("ip2");
	}


	float classification_error = 1.0f;

	int classifications = FLAGS_classify;
	if (classifications < 0)
		classifications = (int)test_size;

	// Test the resulting neural network's classification
	if (classifications > 0)
	{
		// Initialize a TrainingContext structure for testing (different batch size)
		TrainingContext test_context(FLAGS_gpu, 1, conv1, pool1, conv2, pool2, fc1, fc2);

		// Ensure correct workspaceSize is allocated for testing
		if (context.m_workspaceSize < test_context.m_workspaceSize)
		{
			checkCudaErrors(cudaFree(d_cudnn_workspace));
			checkCudaErrors(cudaMalloc(&d_cudnn_workspace, test_context.m_workspaceSize));
		}

		int num_errors = 0;
		for (int i = 0; i < classifications; ++i)
		{
			std::vector<float> data(width * height);
			// Normalize image to be in [0,1]
			for (int j = 0; j < width * height; ++j)
				data[j] = (float)test_images[i * width*height*channels + j] / 255.0f;

			checkCudaErrors(cudaMemcpyAsync(input_data, &data[0], sizeof(float) * width * height, cudaMemcpyHostToDevice));

			// Forward propagate test image
			test_context.ForwardPropagation(input_data, conv1_data, conv1relu_data, pool1_data, conv2_data, conv2relu_data, pool2_data, fc1_data, 
				fc1relu_data, fc2_data, softmax_data,
				conv1_kernel, conv1_bias, conv2_kernel, conv2_bias, fc1_weight, fc1_bias,
				fc2_weight, fc2_bias, d_cudnn_workspace, d_onevec);

			// Perform classification
			std::vector<float> class_vec(10);

			// Copy back result
			checkCudaErrors(cudaMemcpy(&class_vec[0], softmax_data, sizeof(float) * 10, cudaMemcpyDeviceToHost));

			// Determine classification according to maximal response
			int chosen = 0;
			for (int id = 1; id < 10; ++id)
			{
				if (class_vec[chosen] < class_vec[id]) chosen = id;
			}

			if (chosen != test_labels[i])
				++num_errors;
		}
		classification_error = (float)num_errors / (float)classifications;

		printf("Classification result: %.2f%% error (used %d images)\n", classification_error * 100.0f, (int)classifications);
	}

	// Free data structures
	checkCudaErrors(cudaFree(input_data));
	checkCudaErrors(cudaFree(conv1_data));
	checkCudaErrors(cudaFree(conv1relu_data));
	checkCudaErrors(cudaFree(pool1_data));
	checkCudaErrors(cudaFree(conv2_data));
	checkCudaErrors(cudaFree(conv2relu_data));
	checkCudaErrors(cudaFree(pool2_data));
	checkCudaErrors(cudaFree(fc1_data));
	checkCudaErrors(cudaFree(fc2_data));
	checkCudaErrors(cudaFree(conv1_kernel));
	checkCudaErrors(cudaFree(conv1_bias));
	checkCudaErrors(cudaFree(conv2_kernel));
	checkCudaErrors(cudaFree(conv2_bias));
	checkCudaErrors(cudaFree(fc1_weight));
	checkCudaErrors(cudaFree(fc1_bias));
	checkCudaErrors(cudaFree(fc2_weight));
	checkCudaErrors(cudaFree(fc2_bias));
	checkCudaErrors(cudaFree(conv1_kernel_diff));
	checkCudaErrors(cudaFree(conv1_bias_diff));
	checkCudaErrors(cudaFree(conv2_kernel_diff));
	checkCudaErrors(cudaFree(conv2_bias_diff));
	checkCudaErrors(cudaFree(fc1_weight_diff));
	checkCudaErrors(cudaFree(fc1_bias_diff));
	checkCudaErrors(cudaFree(fc2_weight_diff));
	checkCudaErrors(cudaFree(fc2_bias_diff));
	checkCudaErrors(cudaFree(conv1relu_diff));
	checkCudaErrors(cudaFree(pool1_diff));
	checkCudaErrors(cudaFree(conv2_diff));
	checkCudaErrors(cudaFree(conv2relu_diff));
	checkCudaErrors(cudaFree(pool2_diff));
	checkCudaErrors(cudaFree(fc1_diff));
	checkCudaErrors(cudaFree(fc2_diff));
	checkCudaErrors(cudaFree(input_labels));
	checkCudaErrors(cudaFree(loss));
	checkCudaErrors(cudaFree(d_onevec));
	if (d_cudnn_workspace != nullptr)
		checkCudaErrors(cudaFree(d_cudnn_workspace));

	return 0;
}
