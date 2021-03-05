#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"

int main(int argc,  char** argv){
    cl::sycl::queue matmul_queue(cl::sycl::default_selector{});

    cl::sycl::buffer<float, 2> A(cl::sycl::range(32, 32));
    cl::sycl::buffer<float, 2> B(cl::sycl::range(32, 32));

    
    auto matmul_task=[&](cl::sycl::handler& h) 
    { 
    
    
    
    };
    matmul_queue.submit(matmul_task);
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    return 0;

}
