#include "LayerNormPlugin.h"

using namespace nvinfer1;

PluginFieldCollection    LayerNormPluginCreator::fc_ {};
std::vector<PluginField> LayerNormPluginCreator::attr_;

template<typename T, int X,int STEP>
__global__ void layerNormKernelV1(T *pInput, T *pOutput)
{
    const int threadId = threadIdx.x;
    const int M = blockDim.x; //M=256
    const int idx = threadId + M*blockIdx.x;
    __shared__ T mean_shared, var_shared;
    typedef cub::BlockReduce<T, X, cub::BLOCK_REDUCE_WARP_REDUCTIONS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    T tmp1 = pInput[idx]/(T)(M);
    T& ref0 = tmp1;
    T sum  = BlockReduce(temp_storage).Sum(ref0);
    if(threadId == 0)
    {
        mean_shared =(T)sum / (T)(M);
        //mean_shared =(T)sum;
        //printf("M=%d,mean_shared=%f ",M,mean_shared);
    }
    __syncthreads();
    const T moment = (T)(pInput[idx] - mean_shared);
    T moment2 = moment * moment;
    //T tmp2 = moment2/(T)(M);
    T &ref1 = moment2;
    T  var  = BlockReduce(temp_storage).Sum(ref1);
    //T  var  = BlockReduce(temp_storage).Sum(ref1);
    if (threadId == 0)
    {
        var_shared =(T)var / (T)(M);
        //var_shared =(T)var;
        //printf("var_shared=%.5f ",var_shared);
    }
    __syncthreads();
    //pOutput[idx] = (pInput[idx] - mean_shared) / ((T)sqrtf(var_shared + (T)1e-7)*(T)16);
    pOutput[idx] = moment * (T)rsqrtf(var_shared + (T)1e-7);
}

template<typename T, int X, int STEP>
__global__ void layerNormKernelV2(T *pInput, T *pOutput, float epsilon)
{
    // 先找到当前线程位于线程格中的哪一个线程块blockId
    //const int blockId = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
    const int blockId = blockIdx.x;
    // 找到当前线程位于线程块中的哪一个线程threadId
    //const int threadId = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    const int threadId = threadIdx.x;
    // 计算一个线程块中一共有多少个线程M
    //const int M = blockDim.x*blockDim.y*blockDim.z;
    const int M = blockDim.x;
    // 求得当前的线程序列号idx
    const int idx = threadId + M*blockId;
    __shared__ T mean_shared, var_shared;
    typedef cub::BlockReduce<T, X> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    //printf("idx=%d,v=%f ",idx,pInput[idx]);
    T _x_thread_sum = 0;
    for(int i = 0;i<STEP;i++)
    {
        //printf("idx=%d,k=%f ",idx,pInput[idx*STEP+i]);
        _x_thread_sum +=pInput[idx*STEP+i];
    }
    //printf("idx=%d,_x_thread_sum=%f ",idx,_x_thread_sum);
    //T& ref0 = _x_thread_sum;
    //T sum  = BlockReduce(temp_storage).Sum(ref0);
    //T sum  = BlockReduce(temp_storage).Sum(_x_thread_sum);
    T sum  = BlockReduce(temp_storage).Sum(_x_thread_sum / (T)(STEP*M));
    //__syncthreads();
    if(threadId == 0)
    {
        //printf("sum=%f ",sum);
        //printf("blockDim.y=%d ",blockDim.y);
        //printf("idx=%d,v=%f ",idx,pInput[idx]);
        mean_shared =(T)sum;
        //printf("M=%d,mean_shared=%f ",M,mean_shared);
    }
    __syncthreads();
    T moment_x = 0;
    for(int i = 0;i<STEP;i++)
    {
        T moment = pInput[idx*STEP+i] - mean_shared;
        T moment2 = moment * moment;
        moment_x += moment2;
    }
    //moment_x = moment_x / (T)(STEP*M);
    //T &ref1 = moment_x;
    //T  var  = BlockReduce(temp_storage).Sum(ref1);
    T  var  = BlockReduce(temp_storage).Sum(moment_x / (T)(STEP*M));
    if (threadId == 0)
    {
        //printf("var=%f ",var);
        var_shared =(T)var;
        //printf("var_shared=%.5f ",var_shared);
    }
    __syncthreads();
    for(int i = 0;i<STEP;i++)
    {
        T moment = pInput[idx*STEP+i] - mean_shared;
        pOutput[idx*STEP+i] = moment * (T)rsqrtf(var_shared + (T)epsilon);
    }
}

template<typename T, int X,int STEP>
__global__ void layerNormKernelV3(T *pInput, T *pOutput)
{
    const int threadId = threadIdx.x;
    const int M = blockDim.x; //M=256
    const int idx = threadId + M*blockIdx.x;
    __shared__ T mean_shared, var_shared;
    typedef cub::BlockReduce<T, X, cub::BLOCK_REDUCE_WARP_REDUCTIONS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    T tmp1 = pInput[idx]/(T)(M);
    T sum  = BlockReduce(temp_storage).Sum(tmp1);
    if(threadId == 0)
    {
        //mean_shared =(T)sum / (T)(M);
        mean_shared =(T)sum;
        //printf("M=%d,mean_shared=%f ",M,mean_shared);
    }
    __syncthreads();
    const T moment = (pInput[idx] - mean_shared)/(T)256;
    T moment2 = moment * moment;
    T  var  = BlockReduce(temp_storage).Sum(moment2);
    //T  var  = BlockReduce(temp_storage).Sum(ref1);
    if (threadId == 0)
    {
        //var_shared =(T)var / (T)(M);
        var_shared =(T)rsqrtf(var + (T)1e-7f);
        //printf("var_shared=%.5f ",var_shared);
    }
    __syncthreads();
    pOutput[idx] = moment * var_shared * (T)16;
}


int32_t LayerNormPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    int nGrid = inputDesc[0].dims.d[0]*inputDesc[0].dims.d[1],nBlock = 1;
    for (int i = 2; i < inputDesc[0].dims.nbDims; ++i)
    {
        nBlock *= inputDesc[0].dims.d[i];
        //printf("inputDesc[0].dims.d[%d] =[%d]",i , inputDesc[0].dims.d[i]);
    }
    //printf("nGrid:[%d],nBlock:[%d]\n", nGrid, nBlock);
    //nElement = nGrid * nBlock;
    /*
    const int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
    if (inputDesc[0].type == DataType::kFLOAT)
    {
        //outputDesc[0].type = DataType::kFLOAT;
        (layerNormKernelV3<float>) <<<nBlock, 128, 0, stream>>>((float *)inputs[0], (float *)outputs[0],epsilon_);
    }
    else
    {
        //outputDesc[0].type = DataType::kHALF;
        (layerNormKernelV3<half>) <<<nBlock, 128, 0, stream>>>((half *)inputs[0], (half *)outputs[0],epsilon_);
    }*/
    //dim3 block_size_v2(CEIL_DIVIDE(nBlock, 4), 4, 1);
    //dim3 block_size_v3(CEIL_DIVIDE(nBlock, 16), 4, 4);
    /*
    switch (inputDesc[0].type)
    {
    case DataType::kFLOAT:
        (layerNormKernelV1<float, 256, 1>)<<<nGrid, nBlock, 0, stream>>>((float *)inputs[0], (float *)outputs[0]);
        break;
    default:
        (layerNormKernelV1<half, 256, 1>)<<<nGrid, nBlock, 0, stream>>>((half *)inputs[0], (half *)outputs[0]);
        break;
    }
    */
    if (inputDesc[0].type == DataType::kFLOAT)
    {
        switch (nBlock)
        {
        case 256:
            (layerNormKernelV1<float, 256, 1>)<<<nGrid, nBlock, 0, stream>>>((float *)inputs[0], (float *)outputs[0]);
            break;
        case 384:
            (layerNormKernelV1<float, 384, 1>)<<<nGrid, nBlock, 0, stream>>>((float *)inputs[0], (float *)outputs[0]);
            break;
        default: // shoulf NOT be here
            printf("[LayerNormPlugin::enqueue] nGrid = %d is not supported\n", nGrid);
            break;
        }
    }
    else
    {
        switch (nBlock)
        {
        case 256:
            (layerNormKernelV1<half, 256, 1>)<<<nGrid, nBlock, 0, stream>>>((half *)inputs[0], (half *)outputs[0]);
            break;
        case 384:
            (layerNormKernelV1<half, 384, 1>)<<<nGrid, nBlock, 0, stream>>>((half *)inputs[0], (half *)outputs[0]);
            break;
        default: // shoulf NOT be here
            printf("[LayerNormPlugin::enqueue] nGrid = %d is not supported\n", nGrid);
            break;
        }
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);
