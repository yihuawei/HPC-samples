#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"

const int row_A=32;
const int col_A=128;
const int row_B=128;
const int col_B=64;
const int row_C=row_A;
const int col_C=col_B;

namespace sycl = cl::sycl;

int main(int argc,  char** argv){
    
    std::array<float, row_A*col_A> host_A;
    std::array<float, row_B*col_B> host_B;
    std::array<float, row_C*col_C> host_C;
	
	
    sycl::buffer<float, 2> buffer_A(host_A.data(), sycl::range(row_A, col_A));
    sycl::buffer<float, 2> buffer_B(host_B.data(), sycl::range(row_B, col_B));
    sycl::buffer<float, 2> buffer_B(host_C.data(), sycl::range(row_C, col_C));
    
    auto matmul_task=[&](cl::sycl::handler& h) 
    { 
         accessor& device_A = buffer_A.get_access<sycl::access::mode::read>(h);
         accessor& device_B = buffer_B.get_access<sycl::access::mode::read>(h);
         accessor& device_C = buffer_C.get_access<sycl::access::mode::discard_write>(h);
         
	 int width_A = device_A.get_range()[1];
	 h.parallel_for(sycl::range(row_C, col_C), [=](auto index) {
	     int row = index[0];
             int col = index[1];
             float sum = 0;

             for (int i = 0; i < width_a; i++) {
                 sum += device_A[row][i] * device_B[i][col];
             }

             device_C[index] = sum;
	 });
    
    };

    sycl::queue matmul_queue(cl::sycl::default_selector{});
    matmul_queue.submit(matmul_task);
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    return 0;

}
