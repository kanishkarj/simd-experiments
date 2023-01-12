#ifndef TREENODE_H
#define TREENODE_H

#include <vector>
#include <ostream>

struct Normal;
struct AVX2;
struct AVX2_Optimized;

class parvec
{

private:
    // std::vector<double> my_vec; 
    
public:
    double* my_vec; 
    std::size_t len;

    parvec(std::size_t size);
    ~parvec();
    
    template<typename T>
    void add(parvec &other);

    /**
     * Sum of all positive elements.
    */
    template<typename T>
    double pos_sum();

    template<typename T>
    void norm();

    /**
     * The mathematical length of the vector.
    */
    template<typename T>
    double vec_len();

    template<typename T>
    int find(double needle);

    template<typename T>
    int count_occurence(double needle);

    template<typename T>
    int min();

    void random_init(int max_val, bool include_neg); 
    void same_init(); 
    
    friend std::ostream& operator<<(std::ostream& os, const parvec& vec);
};
#endif