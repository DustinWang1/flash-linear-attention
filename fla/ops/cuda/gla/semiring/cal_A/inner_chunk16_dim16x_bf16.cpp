#include <torch/extension.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;

void run_fwd_inner_chunk16_dim16x(int batchSize, int M, int N_K, 
                                bf16 *Q, bf16 *K,  
                                bf16 *gK, 
                                bf16 *QK
                                );
void run_bwd_inner_chunk16_dim16x(int batchSize, int M, int N_K, 
                                bf16 *Q, bf16 *K, bf16 *G, bf16 *DQK,
                                bf16 *DQ, bf16 *DK, bf16 *DG
                            );            

torch::Tensor fwd(torch::Tensor Q,
torch::Tensor K, torch::Tensor g_K
) {   
    auto QK = torch::empty({Q.size(0), Q.size(1), Q.size(2), Q.size(3), Q.size(3)}, Q.options());    

    int B_size = Q.size(0); // This is the batch size dimension.
    int H_size = Q.size(1); // This is the head dimension
    int num_chunk = Q.size(2); // This is the chunk dimension.    
    int M = Q.size(-2); // this is the chunk size
    int N_K = Q.size(-1); // this is the head_K dim
    
    run_fwd_inner_chunk16_dim16x(B_size * H_size * num_chunk, M, N_K,  
      Q.data_ptr<bf16>(), K.data_ptr<bf16>(),   
      g_K.data_ptr<bf16>(),  QK.data_ptr<bf16>());
      
    return QK;
}

std::vector<torch::Tensor> bwd(torch::Tensor Q,
torch::Tensor K,  
torch::Tensor g_K, torch::Tensor DQK
      ) {
    
    auto DQ = torch::empty_like(Q);
    auto DK = torch::empty_like(K);
    auto Dg_K = torch::empty_like(g_K);
    
    int B_size = Q.size(0); // This is the batch size dimension.
    int H_size = Q.size(1); // This is the head dimension
    int num_chunk = Q.size(2); // This is the chunk dimension.    
    int M = Q.size(-2);
    int N_K = Q.size(-1);

    run_bwd_inner_chunk16_dim16x(B_size * H_size * num_chunk, M, N_K, 
                        Q.data_ptr<bf16>(),
                        K.data_ptr<bf16>(), 
                        g_K.data_ptr<bf16>(),
                        DQK.data_ptr<bf16>(),
                        DQ.data_ptr<bf16>(), 
                        DK.data_ptr<bf16>(),
                        Dg_K.data_ptr<bf16>()
                        );                
    return {DQ, DK, Dg_K};
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fwd, "Batched matrix multiplication with shared memory (CUDA)");
    m.def("backward", &bwd, "Batched matrix multiplication with shared memory (CUDA)");
}

