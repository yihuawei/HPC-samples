#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"

const int row_A=32;
const int col_A=128;
const int row_B=128;
const int col_B=64;






int main(int argc,  char** argv){
    
    std::array<char,12> host_A;
    std::array<char,12> host_B;
    std::array<char,12> host_C;
	
	
	
	
    cl::sycl::queue matmul_queue(cl::sycl::default_selector{});

    cl::sycl::buffer<float, 2> A(cl::sycl::range<2>(32, 32));
    cl::sycl::buffer<float, 2> B(cl::sycl::range<2>(32, 32));

    
    auto matmul_task=[&](cl::sycl::handler& h) 
    { 
    
    
    
    };
    matmul_queue.submit(matmul_task);
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    return 0;

}
