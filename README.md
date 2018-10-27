# cudnn
lenet

# Environment
vs2015, cuda8.0, cudnn7.5

# Realize Lenet to classfier mnist, the network is as follow
Convolution(kernel_size=5, stride=1, padding='VALID');
relu();
Pool1(kernel_size=2, stride=1);
Convolution(kernel_size=5, stride=1, padding='VALID');
relu();
Pool2(kernel_size=2, stride=1);
FullyConnect(weight_size=500);
relu();
FullyConnect(weight_size=10);
Softmax();

# Result screenshot
![image](https://github.com/XPping/cudnn/raw/master/mnist_lenet/result screenshot/result.png)


# Reference
https://github.com/tbennun/cudnn-training.git
#Rename parameters to better understand the cod;
#Add two relu for convolution(include ForwardPropagation and BackPropagation);
#Echo train iter error when training.
