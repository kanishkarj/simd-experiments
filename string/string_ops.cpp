#include "string_ops.h"
#include <bits/stdc++.h>
#include <x86intrin.h>
#include <algorithm>
#include <cstdlib>

using namespace std;

template<>
vector<int> search<algos::Lib>(char* needle, char* text) {
    vector<int> ans(0,0);
    ans.reserve(8);
    
    string s(text);
    string substr(needle);

    int index = 0;
    while ((index = s.find(substr, index)) != string::npos) {
        ans.push_back(index);
        index += substr.length();
    }

    return ans;
}

template<>
vector<int> search<algos::RabinKarp>(char* needle, char* text) {
    vector<int> pos(0,0);
    pos.reserve(8); 

    int nlen = strlen(needle);
    int tlen = strlen(text);
    
    long h = powl(CHAR_SPACE_BASE, nlen - 1);
    
    long nhash = 0;
    long thash = 0;
    
    int i,j;
    
    // The hash of the pattern and the first window of size nlen in the text.
    for (i=0; i < nlen; i++) {
        nhash = (nhash * CHAR_SPACE_BASE % PRIME + needle[i]) % PRIME;
        thash = (thash * CHAR_SPACE_BASE % PRIME + text[i])   % PRIME;
    }
    
    // Slide the pattern over text one by one
    for (i = 0; i <= tlen - nlen; i++) {
        // i points to the starting of the window
        
        // Check if the current window matches
        if (nhash == thash) {
            for (j = 0; j < nlen; j++) {
                if (needle[j] != text[i+j]) {
                    break;
                }
            }
            
            if (j == nlen) {
                // Implies we reached the end of loop, implying the needle matched the string perfectly.
                pos.push_back(i);
            }
        }
        
        // Move to next block
        if (i < tlen - nlen) {
            thash = ((thash - text[i]*h)*CHAR_SPACE_BASE + text[i+nlen]) % PRIME;
            thash = thash < 0 ? thash + PRIME : thash;
        }
    }
    
    return pos;
}

template<>
bool is_equal<algos::AVX2>(char* str1, char* str2, int len) {
    int stride = 32;

    for (int i=0 ; i < len ; i+=stride) {
        __m256i block1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*> (str1 + i)); 
        __m256i block2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*> (str2 + i)); 

        __m256i match = _mm256_cmpeq_epi8(block1, block2);
        uint32_t match_mask = _mm256_movemask_epi8(match); 
        
        if (match_mask != UINT32_MAX) {
            return false;
        }
    }
    
    return true;
}

template<>
bool is_equal<algos::Naive> (char* str1, char* str2, int len) {
    int j;
    for (j = 0; j < len; j++) {
        if (str1[j] != str2[j]) {
            break;
        }
    }
    return j == len;
}

template<>
bool is_equal<algos::Lib> (char* str1, char* str2, int len) {
    return strcmp(str1, str2) == 0;
}

template<>
int string_len<algos::Lib>(char* str)
{
    return strlen(str);
}

template<>
int string_len<algos::Naive>(char* str)
{
    int i=0;
    while (true) {
        if (str[i++] == '\0') {
            return i;
        }
    }
    return -1;
}

template<>
int string_len<algos::AVX2>(char* str)
{
    int stride = 32;
    __m256i compare_with = _mm256_setzero_si256(); 
    int i=0;
    while (true) {
        __m256i block = _mm256_loadu_si256(reinterpret_cast<const __m256i*> (str + i)); 
        __m256i flag =_mm256_cmpeq_epi8(block, compare_with);
        
        if (!_mm256_testc_si256(compare_with, flag)) {
            uint32_t match_mask = _mm256_movemask_epi8(flag); 
            return i + __builtin_clz(match_mask);
        }

        i += stride;
    }
    
    return i;
}

void print(__m128i v) {
    auto t = (char*) &v;
    for (int i = 0; i < 16; i++)
        std::cout << (t[i]) << " ";
    std::cout << std::endl;
}

template<>
int string_len<algos::AVX2_Optimized>(char* str)
{
    int stride = 16;

    __m128i zeroes = _mm_setzero_si128(); 

    int i=0;
    while (true) {
        __m128i block = _mm_loadu_si128(reinterpret_cast<const __m128i*> (str + i)); 

        // Most SIG because we want the left most one
        int flags = _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_EACH | _SIDD_LEAST_SIGNIFICANT;
        
        if (_mm_cmpistrc(block, zeroes, flags)) {
            return i + _mm_cmpistri(block, zeroes, flags);
        }

        i += stride;
    }
    
    return i;
}

template<>
vector<int> search<algos::AVX2>(char* needle, char* text) {
    vector<int> pos(0,0);
    pos.reserve(8); 

    int nlen = strlen(needle);
    int tlen = strlen(text);
    
    __m256i first = _mm256_set1_epi8(needle[0]);
    __m256i last = _mm256_set1_epi8(needle[nlen-1]);
    
    int stride = 32;
    for (int i=0 ; i < tlen - nlen ; i+=stride) {
        __m256i block_start = _mm256_loadu_si256(reinterpret_cast<const __m256i*> (text + i)); 
        __m256i block_end = _mm256_loadu_si256(reinterpret_cast<const __m256i*> (text + i + nlen - 1)); 
        
        __m256i start_match = _mm256_cmpeq_epi8(block_start, first);
        __m256i end_match   = _mm256_cmpeq_epi8(block_end, last);
        
        __m256i match = _mm256_and_si256(start_match, end_match);

        uint32_t match_mask = _mm256_movemask_epi8(match); 
        
        while (match_mask != 0) {
            int maski = __builtin_ctz(match_mask);
            int match_pos = maski + i;

            if (is_equal<algos::AVX2>(text + match_pos, needle, nlen)) {
                pos.push_back(match_pos);
            }

            match_mask &= !(1 << maski);
        }
    }
        
    return pos;
}

vector<int> search_small(char* needle, char* text) {
    vector<int> pos(0,0);
    pos.reserve(8); 

    int tlen = strlen(text);
    int nlen = strlen(needle);
    
    uint32_t flag = 0;
    
    if (nlen == 32) {
        flag = UINT32_MAX;
    }
    else {
        flag = ~(~0u << nlen);
    }
    
    __m256i nreg = _mm256_loadu_si256(reinterpret_cast<const __m256i*> (needle));
    
    for (int i=0 ; i < tlen; i+=1) {
        __m256i block = _mm256_loadu_si256(reinterpret_cast<const __m256i*> (text + i)); 
        __m256i match = _mm256_cmpeq_epi8(block, nreg);

        // Movemask seems to be fucking with endianness
        uint32_t match_mask = _mm256_movemask_epi8(match); 
        
        if (match_mask == flag) {
            pos.push_back(i);
        }
    }
        
    return pos;
}

template<>
void to_lower<algos::AVX2>(char* text)
{
    int len = strlen(text);
    __m256i block_a = _mm256_set1_epi8('A'-1);
    __m256i block_z = _mm256_set1_epi8('Z'+1);
    __m256i addend = _mm256_set1_epi8('a' - 'A');
    __m256i zeroes = _mm256_setzero_si256();
    
    int stride = 32;
    for (int i=0 ; i < len ; i+=stride) {   
        __m256i block = _mm256_loadu_si256(reinterpret_cast<const __m256i*> (text + i));
        __m256i mask_a = _mm256_cmpgt_epi8(block, block_a);
        __m256i mask_z = _mm256_cmpgt_epi8(block_z, block);
        
        __m256i match_mask = _mm256_and_si256(mask_a, mask_z); 
        __m256i to_add = _mm256_blendv_epi8(zeroes, addend, match_mask);

        block = _mm256_add_epi8(block, to_add);

        _mm256_storeu_si256(reinterpret_cast<__m256i_u*>(text + i), block);
    }
}

template<>
void to_lower<algos::Lib>(char* text)
{
    int len = strlen(text);
    transform(text, text+len, text+len, ::tolower);
}

