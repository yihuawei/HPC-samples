#include <random>
#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"

const int m=6;
const int k=8;
const int n=4;

namespace sycl = cl::sycl;

int main(int argc,  char** argv){
    
    std::array<float, m*k> host_A;
    std::array<float, k*n> host_B;
    std::array<float, m*n> host_C;
    
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    std::generate(host_A.begin(), host_A.end(), [&]()->float{ return dis(gen); });
    std::generate(host_B.begin(), host_B.end(), [&]()->float{ return dis(gen); });

    std::ostream_iterator<float> out_iter(std::cout, " ");
    //copy(host_A.cbegin(), host_A.cend(), out_iter);
    //copy(host_B.cbegin(), host_B.cend(), out_iter);

    sycl::buffer<float, 2> buffer_A(host_A.data(), sycl::range(m, k));
    sycl::buffer<float, 2> buffer_B(host_B.data(), sycl::range(k, n));
    sycl::buffer<float, 2> buffer_C(host_C.data(), sycl::range(m, n));
    auto matmul_task=[&](cl::sycl::handler& h) 
    { 
         sycl::accessor device_A = buffer_A.get_access<sycl::access::mode::read>(h);
         sycl::accessor device_B = buffer_B.get_access<sycl::access::mode::read>(h);
         sycl::accessor device_C = buffer_C.get_access<sycl::access::mode::discard_write>(h);
         
	 int width_A = buffer_A.get_range()[1];;
	 h.parallel_for(sycl::range(m, n), [=](sycl::item<2> index) {
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
    

    std::ostream_iterator<float> out_iter(std::cout, " ");
    
    struct cout_class_func{
        const char* str;
	int index=0;
	int width=0;
        cout_class_func(const char* s, int width):str(s), width(width){
	    std::cout<<std::string(s)<<end::endl;
	}
        void operator () (int i){
	    if(index%width==0) std::cout<<endl;
            std::cout<<i<<" ";
            index++;
        }
	~cout_class_func(){
	    std::cout<<std::string(s)<<end::endl;
	}
    };
    
    
    for_each(host_A.cbegin(),host_A.cend(), cout_class_func("Matrix A", k));
    for_each(host_B.cbegin(),host_B.cend(), cout_class_func("Matrix B", n));
    for_each(host_C.cbegin(),host_C.cend(), cout_class_func("Matrix C", n))

    return 0;
    double alpha = 1.0;
    double beta = 0.0;
    auto transA = oneapi::mkl::transpose::nontrans;
    auto transB = oneapi::mkl::transpose::nontrans;
    int lda = k;
    int ldb = n;
    int ldc = n;
    auto gemm_buffer_A = buffer_A.reinterpret<float, 1>(sycl::range(m*k));
    auto gemm_buffer_B = buffer_B.reinterpret<float, 1>(sycl::range(k*n));
    auto gemm_buffer_C = buffer_C.reinterpret<float, 1>(sycl::range(m*n));
    //oneapi::mkl::blas::row_major::gemm(matmul_queue, transA, transB, m, n, k,
    //                                       alpha, gemm_buffer_A, lda, gemm_buffer_B, ldb, beta, gemm_buffer_C, ldc);  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    return 0;

}
