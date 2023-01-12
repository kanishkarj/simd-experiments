#include <vector>

const long PRIME          = __INT64_MAX__;
const int CHAR_SPACE_BASE = 256;

namespace algos {
    struct RabinKarp;
    struct AVX2;
    struct Naive;
    struct Lib;
    struct AVX2_Optimized;
}

/**
 * Where the search phrase is greater than 32 characters long. 
*/
template<typename T>
std::vector<int> search(char* needle, char* text);

/**
 * Where the search phrase length is lesser than or equals to 32. 
*/
std::vector<int> search_small(char* needle, char* text);

template<typename T>
int string_len(char* str);

template<typename T>
void to_lower(char* txt);

template<typename T>
bool is_equal(char* str1, char* str2, int len);