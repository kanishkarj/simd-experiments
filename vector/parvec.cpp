#include <iostream>
#include <x86intrin.h>
#include <cstdlib>
#include <cmath>
#include <malloc.h>

#include "parvec.h"

parvec::parvec(std::size_t size) : len(size), my_vec((double*)memalign(32, sizeof(double) * size))
{
}

parvec::~parvec()
{
}

void parvec::random_init(int max_val, bool include_neg) {
    for (int i = 0; i < len; i ++ ) {
        my_vec[i] = rand() %  max_val - (include_neg? max_val/2:0);
    }
}

void parvec::same_init() {
    int val = rand() % 100;
    for (int i = 0; i < len; i ++ ) {
        my_vec[i] = val;
    }
}

std::ostream &operator<<(std::ostream &os, const parvec &vec)
{
    os << "[";
    for (int i=0;i<vec.len;i++)
    {
        os << vec.my_vec[i] << ",";
    }
    os << "]";
    return os;
}

template <>
void parvec::add<Normal>(parvec &other)
{
    for (int i = 0; i < len; i++)
    {
        my_vec[i] += other.my_vec[i];
    }
}

template <>
void parvec::add<AVX2>(parvec &other)
{
    int stride = 4;

    for (int i = 0; i < len; i+=stride)
    {
        __m256d x = _mm256_load_pd(&my_vec[i]);
        __m256d y = _mm256_load_pd(&other.my_vec[i]);

        _mm256_store_pd(&my_vec[i], _mm256_add_pd(x, y));
    }
}

template <>
void parvec::add<AVX2_Optimized>(parvec &other)
{
    int stride = 8;

    for (int i = 0; i < len; i+=stride)
    {
        // TODO: My experiments with prefetching have generally
        // caused worse performance. need to explore this further.
        
        __m256d x1 = _mm256_load_pd(&my_vec[i]);
        __m256d y1 = _mm256_load_pd(&other.my_vec[i]);

        __m256d x2 = _mm256_load_pd(&my_vec[i+4]);
        __m256d y2 = _mm256_load_pd(&other.my_vec[i+4]);

        _mm256_store_pd(&my_vec[i], _mm256_add_pd(x1, y1));
        _mm256_store_pd(&my_vec[i+4], _mm256_add_pd(x2, y2));
    }
}


template <>
double parvec::vec_len<Normal>()
{
    double sum = 0;
    for (int i = 0; i < len; i++)
    {
        sum += my_vec[i] * my_vec[i]; 
    }
    return sqrt(sum); 
}

template <>
double parvec::vec_len<AVX2>()
{
    int stride = 4;
    double sum = 0;

    __m256d acc = _mm256_setzero_pd(); 

    for (int i = 0; i < len / stride; i++)
    {
        int p = i * stride;
        __m256d x = _mm256_loadu_pd(&my_vec[p]);

        acc = _mm256_fmadd_pd(x, x, acc);
    }
    
    acc = _mm256_hadd_pd(acc, acc);
    sum = ((double*)&acc)[0] + ((double*)&acc)[2];
    
    return sqrt(sum);
}


template <>
void parvec::norm<Normal>()
{
    double vec_len = this->vec_len<Normal>();
    for (int i = 0; i < len; i++)
    {
        my_vec[i] /= vec_len;
    }
}

template <>
void parvec::norm<AVX2>()
{
    double vec_len = this->vec_len<AVX2>();
    int stride = 4;

    __m256d acc = _mm256_set1_pd(vec_len); 

    for (int i = 0; i < len / stride; i++)
    {
        int p = i * stride;
        __m256d x = _mm256_loadu_pd(&my_vec[p]);
        _mm256_store_pd(&my_vec[p], _mm256_div_pd(x, acc));
    }
}

template <>
double parvec::pos_sum<AVX2>()
{
    __m256d zero_reg = _mm256_setzero_pd(); 
    __m256d acc = _mm256_setzero_pd(); 
    int stride = 4;

    for (int i = 0; i < len / stride; i++)
    {
        int p = i * stride;
        __m256d x = _mm256_loadu_pd(&my_vec[p]);

        __m256d mask = _mm256_cmp_pd(x, zero_reg, 13);
        x = _mm256_blendv_pd(zero_reg, x, mask); 

        acc = _mm256_add_pd(x, acc);
    }

    acc = _mm256_hadd_pd(acc, acc);
    return ((double*)&acc)[0] + ((double*)&acc)[2];
}

template <>
double parvec::pos_sum<Normal>()
{
    double sum = 0; 
    for (int i = 0; i < len;i ++ ) {
        if (my_vec[i] >= 0) {
            sum += my_vec[i]; 
        }
    }
    return sum;
}

template <>
int parvec::find<Normal>(double needle)
{
    for (int i = 0; i < len;i ++ ) {
        if (my_vec[i] == needle) {
            return i;
        }
    }
    return -1;
}

template <>
int parvec::find<AVX2>(double needle)
{
    int stride = 4;
    __m256d needle_reg = _mm256_set1_pd(needle);

    for (int i = 0; i < len; i += stride)
    {
        __m256d x    = _mm256_loadu_pd(&my_vec[i]);
        __m256d mask = _mm256_cmp_pd(x, needle_reg, 0);
        int imask = _mm256_movemask_pd(mask);

        if (imask != 0) {
            return i + __builtin_ctz(imask);
        }
    }
    return -1;
}

template <>
int parvec::find<AVX2_Optimized>(double needle)
{
    int stride = 4;
    __m256d needle_reg = _mm256_set1_pd(needle);

    for (int i = 0; i < len; i += stride*2)
    {
        __m256d x1 = _mm256_loadu_pd(&my_vec[i]);
        __m256d x2 = _mm256_loadu_pd(&my_vec[i+4]);

        __m256d mask1 = _mm256_cmp_pd(x1, needle_reg, 0);
        __m256d mask2 =_mm256_cmp_pd(x2, needle_reg, 0);

        __m256d mask = _mm256_or_pd(mask1, mask2);

        int imask1 = _mm256_movemask_pd(mask1);
        int imask2 = _mm256_movemask_pd(mask2);

        if (imask2 | imask2 != 0) {
            return i + __builtin_ctz(imask2 << 4 + imask1);
        }
    }
    return -1;
}

template <>
int parvec::count_occurence<Normal>(double needle)
{
    int cnt = 0;
    for (int i = 0; i < len;i ++ ) {
        if (my_vec[i] == needle) {
            cnt ++;
        }
    }
    return cnt;
}

template <>
int parvec::count_occurence<AVX2>(double needle)
{
    int stride = 4;
    int cnt = 0;
    __m256d needle_reg = _mm256_set1_pd(needle);

    for (int i = 0; i < len / stride; i++)
    {
        int p = i * stride;

        __m256d x    = _mm256_loadu_pd(&my_vec[p]);
        __m256d mask = _mm256_cmp_pd(x, needle_reg, 0);
        int imask    = _mm256_movemask_pd(mask);
        
        if (imask != 0) {
            cnt += __popcntd(imask);
        }
    }
    return cnt;
}

template <>
int parvec::count_occurence<AVX2_Optimized>(double needle)
{
    int stride = 8;

    __m256d needle_reg = _mm256_set1_pd(needle);

    // Using multiple accumulators to increase Instruction level parallelism
    __m256d acc1 = _mm256_setzero_pd();
    __m256d acc2 = _mm256_setzero_pd();

    const __m256d ones = _mm256_set1_pd(1);

    for (int i = 0; i < len; i+=stride)
    {
        __m256d x1    = _mm256_loadu_pd(&my_vec[i]);
        __m256d mask1 = _mm256_cmp_pd(x1, needle_reg, 0);
                mask1 = _mm256_and_pd(mask1, ones);

        __m256d x2    = _mm256_loadu_pd(&my_vec[i+4]);
        __m256d mask2 = _mm256_cmp_pd(x2, needle_reg, 0);
                mask2 = _mm256_and_pd(mask2, ones);

        // Doing the accumulation at the end makes it faster.
        acc1  = _mm256_add_pd(acc1, mask1);
        acc2  = _mm256_add_pd(acc2, mask2);
    }
    
    acc1 = _mm256_add_pd(acc1, acc2);

    acc1 = _mm256_hadd_pd(acc1, acc1);
    return (int)((double*)&acc1)[0] + ((double*)&acc1)[2];
}

template <>
int parvec::min<Normal>()
{
    int ind = 0;
    
    for (int i = 0; i < len; i++) {
        if (my_vec[i] < my_vec[ind]) {
            ind = i;
        }
    }
    
    return ind;
}

template <>
int parvec::min<AVX2>()
{
    __m256d min_val = _mm256_set1_pd(INT64_MAX);
    __m256d min_indices = _mm256_set1_pd(0);
    __m256d curr_indices = _mm256_setr_pd(0, 1, 2, 3);
    
    const __m256d reg4 = _mm256_set1_pd(4);

    int stride = 4;
    for (int i = 0; i < len; i+=stride) {
        __m256d x    = _mm256_loadu_pd(&my_vec[i]);

        __m256d mask = _mm256_cmp_pd(x, min_val, 1);

        min_val = _mm256_min_pd(min_val, x);
        
        min_indices = _mm256_blendv_pd(min_indices, curr_indices, mask);

        curr_indices = _mm256_add_pd(curr_indices, reg4);
    }

    double min_val_arr[4]; double min_idx_arr[4];

    _mm256_storeu_pd(&min_val_arr[0], min_val);
    _mm256_storeu_pd(&min_idx_arr[0], min_indices);
    
    int arr_ind = (int)min_idx_arr[0], ind = 0;
    for (int i = 1; i < 4; i++) {
        if(min_val_arr[i] < min_val_arr[ind]) {
            ind = i;
            arr_ind = (int)min_idx_arr[i];
        }
    }
    
    return arr_ind;
}

template <>
int parvec::min<AVX2_Optimized>()
{
    __m256d min_val = _mm256_set1_pd(INT64_MAX);
    int min_index = 0;

    int stride = 8;
    for (int i = 0; i < len; i+=stride) {
        __m256d x1   = _mm256_loadu_pd(&my_vec[i]);
        __m256d x2   = _mm256_loadu_pd(&my_vec[i+4]);

        __m256d m = _mm256_min_pd(x1, x2);

        __m256d mask = _mm256_cmp_pd(m, min_val, 1);
        __m128i maski = _mm256_cvttpd_epi32(mask);
        
        if (! _mm_testz_si128(maski, maski)) { [[unlikely]]
            for (int j = i; j < i+4; j++) {
                if (my_vec[min_index] > my_vec[j]) {
                    min_index = j;
                }
            }
            min_val = _mm256_min_pd(min_val, m);
        }
    }

    return min_index;
}