#include <random>
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
    
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    std::generate(host_A.begin(), host_A.end(), [&]()->float{ return dis(gen); });
    std::generate(host_B.begin(), host_B.end(), [&]()->float{ return dis(gen); });

    std::ostream_iterator<float> out_iter(std::cout, " ");
    copy(host_A.cbegin(), host_A.cend(), out_iter);
    copy(host_B.cbegin(), host_B.cend(), out_iter);

    sycl::buffer<float, 2> buffer_A(host_A.data(), sycl::range(row_A, col_A));
    sycl::buffer<float, 2> buffer_B(host_B.data(), sycl::range(row_B, col_B));
    sycl::buffer<float, 2> buffer_C(host_C.data(), sycl::range(row_C, col_C));
    
    auto matmul_task=[&](cl::sycl::handler& h) 
    { 
         sycl::accessor device_A = buffer_A.get_access<sycl::access::mode::read>(h);
         sycl::accessor device_B = buffer_B.get_access<sycl::access::mode::read>(h);
         sycl::accessor device_C = buffer_C.get_access<sycl::access::mode::discard_write>(h);
         
	 int width_A = device_A.get_range()[1];
	 h.parallel_for(sycl::range(row_C, col_C), [=](sycl::item<2> index) {
	     int row = index[0];
             int col = index[1];
             float sum = 0;

             for (int i = 0; i < width_A; i++) {
                 sum += device_A[row][i] * device_B[i][col];
             }

             device_C[index] = sum;
	 });
    
    };

    sycl::queue matmul_queue(cl::sycl::default_selector{});
    matmul_queue.submit(matmul_task);
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    return 0;

}
