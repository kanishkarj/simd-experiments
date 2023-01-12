#include <iostream>
#include <algorithm>
#include "string_ops.h"
#include <cstdlib>
#include <cstring>
#include <string>
#include <chrono>

using namespace std;

template<typename Functor>
void bench(string benchName, Functor func, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        func();
    }

    auto finish = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = finish - start;
    std::cout << benchName << " - Elapsed Time: " << elapsed.count() / iterations << " ms" << std::endl;
}

// Unique pointer
string get_random_str(int len) {
    string text = string(len, '*');
    
    for (int i = 0; i < len; i++ ) {
        text[i] = (char)(rand() % (122-65) + 65);
    }
    
    return text;
}

void print_vec(vector<int> v) {
    cout << endl;
    for (auto x: v) {
        cout << x << ", ";
    }
    cout << endl;
}

void test_to_lower()
{
    int tlen = 1 << 24;

    string text = get_random_str(tlen);
    
    bench("[To Lower] AVX", [&] {
        to_lower<algos::AVX2>((char*)text.c_str());
    }, 1);

    text = get_random_str(tlen);
 
    bench("[To Lower] Lib", [&] {
        to_lower<algos::Lib>((char*)text.c_str());
    }, 1);
}

void test_equals()
{
    int tlen = 1 << 24;

    string text = get_random_str(tlen);
    
    bench("[Equals] AVX", [&] {
        is_equal<algos::AVX2>((char*)text.c_str(), (char*)text.c_str(), tlen);
    }, 1000);

    bench("[Equals] Normal", [&] {
        is_equal<algos::Naive>((char*)text.c_str(), (char*)text.c_str(), tlen);
    }, 1000);

    bench("[Equals] Lib", [&] {
        is_equal<algos::Lib>((char*)text.c_str(), (char*)text.c_str(), tlen);
    }, 1000);
}

void test_strlen()
{
    int tlen = 1 << 24;

    string text = get_random_str(tlen);
    
    bench("[String length] AVX", [&] {
        string_len<algos::AVX2>((char*)text.c_str());
    }, 1000);

    bench("[String length] AVX Optimized", [&] {
        string_len<algos::AVX2_Optimized>((char*)text.c_str());
    }, 1000);

    bench("[String length] Normal", [&] {
        string_len<algos::Naive>((char*)text.c_str());
    }, 1000);

    bench("[String length] Lib", [&] {
        string_len<algos::Lib>((char*)text.c_str());
    }, 1000);
}

void test_search()
{
    int tlen = 1 << 24;
    int nlen = 1 << 8;
    int needle_start_pos = 1 << 16;

    string text = get_random_str(tlen);
    char needle[nlen]; 
    strncpy(needle, text.c_str() + needle_start_pos, nlen);
    
    bench("[String search] AVX", [&] {
        search<algos::AVX2>(needle, (char*)text.c_str());
    }, 1000);

    bench("[String search] Normal", [&] {
        search<algos::RabinKarp>(needle, (char*)text.c_str());
    }, 1000);

    bench("[String search] Lib", [&] {
        search<algos::Lib>(needle, (char*)text.c_str());
    }, 1000);
}

void test_small_search()
{
    int tlen = 1 << 24;
    int nlen = 1 << 4;
    int needle_start_pos = 1 << 16;

    string text = get_random_str(tlen);
    char needle[nlen]; 
    strncpy(needle, text.c_str() + needle_start_pos, nlen);
    
    bench("[Small String search] AVX", [&] {
        search<algos::AVX2>(needle, (char*)text.c_str());
    }, 1000);

    bench("[Small String search] Normal", [&] {
        search<algos::RabinKarp>(needle, (char*)text.c_str());
    }, 1000);

    bench("[Small String search] Lib", [&] {
        search<algos::Lib>(needle, (char*)text.c_str());
    }, 1000);
}
int main()
{
    test_to_lower();
    // test_equals();
    // test_strlen();
    // test_search();
    // test_small_search();
    return 0;
}