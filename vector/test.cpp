#include <iostream>
#include <chrono>

#include "parvec.h"

template<typename Functor>
void bench(std::string benchName, Functor func) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10000; i++) {
        func();
    }

    auto finish = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::micro> elapsed = finish - start;
    std::cout << benchName << " - Elapsed Time: " << elapsed.count() / 10000 << " us" << std::endl;
}

void test_sum()
{
    std::size_t vec_log_size = 1 << 16;
    std::cout << "[sum] Working with vector of size: " << vec_log_size << std::endl; 
    parvec a(vec_log_size);
    parvec b(vec_log_size);

    // std::cout << a << std::endl;
    // std::cout << b << std::endl;

    a.random_init(100, true);
    b.random_init(100, true);
    bench("AVX", [&a, &b] {
        a.add<AVX2>(b);
    });
    
    a.random_init(100, true);
    b.random_init(100, true);
    bench("Normal", [&a, &b] {
        a.add<Normal>(b);
    });

    a.random_init(100, true);
    b.random_init(100, true);
    bench("AVX Optimized", [&a, &b] {
        a.add<AVX2_Optimized>(b);
    });
    
    // std::cout << a << std::endl;
    // std::cout << b << std::endl;
}

void test_len()
{
    std::size_t vec_log_size = 1 << 16;
    std::cout << "[Vector length] Working with vector of size: " << vec_log_size << std::endl; 
    parvec a(vec_log_size);

    a.random_init(100, true);

    bench("Normal", [&a] {
        a.vec_len<Normal>();
    });

    bench("AVX", [&a] {
        a.vec_len<AVX2>();
    });
}

void test_norm()
{
    std::size_t vec_log_size = 1 << 16;
    std::cout << "[Normalization] Working with vector of size: " << vec_log_size << std::endl; 
    parvec a(vec_log_size);

    a.random_init(100, true);

    bench("Normal", [&a] {
        a.norm<Normal>();
    });

    bench("AVX", [&a] {
        a.norm<AVX2>();
    });
}

void test_pos_sum()
{
    std::size_t vec_log_size = 1 << 16;
    std::cout << "[Sum of positives] Working with vector of size: " << vec_log_size << std::endl; 
    parvec a(vec_log_size);

    a.random_init(100, true);

    bench("Normal", [&a] {
        a.pos_sum<Normal>();
    });

    bench("AVX", [&a] {
        a.pos_sum<AVX2>();
    });
}

void test_find()
{
    std::size_t vec_log_size = 1 << 16;
    std::cout << "[Search] Working with vector of size: " << vec_log_size << std::endl; 
    parvec a(vec_log_size);

    a.random_init(100, true);

    bench("Normal", [&a, &vec_log_size] {
        a.find<Normal>(a.my_vec[vec_log_size-1]);
    });

    bench("AVX", [&a, &vec_log_size] {
        a.find<AVX2>(a.my_vec[vec_log_size-1]);
    });

    bench("AVX Optimized", [&a, &vec_log_size] {
        a.find<AVX2_Optimized>(a.my_vec[vec_log_size-1]);
    });

    // std::cout << a << std::endl;
    // std::cout << a.find<Normal>(a.my_vec[vec_log_size-1]) << std::endl;
    // std::cout << a.find<AVX2>(a.my_vec[vec_log_size-1]) << std::endl;
    // std::cout << a.find<AVX2_Optimized>(a.my_vec[vec_log_size-1]) << std::endl;
}

void test_count_occurrence()
{
    std::size_t vec_log_size = 1 << 16;
    std::cout << "[Count Occurrence - Random] Working with vector of size: " << vec_log_size << std::endl; 
    parvec a(vec_log_size);

    // In random case SIMD is 10x faster than Normal, this doesn't happen when the array is initialized with same_init. I think branch predictor is getting fucked in the former case.  
    a.random_init(2, false);

    bench("Normal", [&a, &vec_log_size] {
        a.count_occurence<Normal>(a.my_vec[vec_log_size-1]);
    });

    bench("AVX", [&a, &vec_log_size] {
        a.count_occurence<AVX2>(a.my_vec[vec_log_size-1]);
    });

    bench("AVX Optimized", [&a, &vec_log_size] {
        a.count_occurence<AVX2_Optimized>(a.my_vec[vec_log_size-1]);
    });
    // std::cout << a.count_occurence<AVX2_Optimized>(a.my_vec[vec_log_size-1]) << std::endl;
    // std::cout << a.count_occurence<Normal>(a.my_vec[vec_log_size-1]) << std::endl;
    // 
    
    std::cout << "[Count Occurrence - Same] Working with vector of size: " << vec_log_size << std::endl; 
    a.same_init();

    bench("Normal", [&a, &vec_log_size] {
        a.count_occurence<Normal>(a.my_vec[vec_log_size-1]);
    });

    bench("AVX", [&a, &vec_log_size] {
        a.count_occurence<AVX2>(a.my_vec[vec_log_size-1]);
    });

    bench("AVX Optimized", [&a, &vec_log_size] {
        a.count_occurence<AVX2_Optimized>(a.my_vec[vec_log_size-1]);
    });
}

void test_min()
{
    std::size_t vec_log_size = 1 << 16;
    std::cout << "[Min] Working with vector of size: " << vec_log_size << std::endl; 
    parvec a(vec_log_size);

    a.random_init(100, true);

    bench("Normal", [&a, &vec_log_size] {
        a.min<Normal>();
    });

    bench("AVX", [&a, &vec_log_size] {
        a.min<AVX2>();
    });

    bench("AVX Optimized", [&a, &vec_log_size] {
        a.min<AVX2_Optimized>();
    });

    // std::cout << a << std::endl;
    // std::cout << a.my_vec[a.min<Normal>()] << std::endl;
    // std::cout << a.my_vec[a.min<AVX2_Optimized>()] << std::endl;
}

int main()
{
    test_sum();
    test_len();
    test_norm();
    test_pos_sum();
    test_find();
    test_count_occurrence(); 
    test_min();

    return 0;
}
