/*
 * Created on Fri Jul 05 2024
 *
 * Author: Chaoji Zuo
 * Email: chaoji.zuo@rutgers.edu
 *
 * Copyright (c) 2024 Rutgers University - Database Group
 */

#pragma once

#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <vector>
// #include <omp.h>
#include <assert.h>
#include <bitset.h>
#include <sys/time.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>

#ifdef __linux__
#include "sys/sysinfo.h"
#include "sys/types.h"
#elif __APPLE__
#include <mach/mach_host.h>
#include <mach/mach_init.h>
#include <mach/mach_types.h>
#include <mach/vm_statistics.h>
#endif

using std::cout;
using std::endl;
using std::getline;
using std::ifstream;
using std::ios;
using std::string;
using std::vector;

float
EuclideanDistance(const vector<float>& lhs, const vector<float>& rhs, const int& startDim, int lensDim);

float
EuclideanDistance(const vector<float>& lhs, const vector<float>& rhs);

float
EuclideanDistanceSquare(const vector<float>& lhs, const vector<float>& rhs);

void
AccumulateTime(timeval& t1, timeval& t2, double& val_time);
void
CountTime(timeval& t1, timeval& t2, double& val_time);
double
CountTime(timeval& t1, timeval& t2);

// the same to sort_indexes
template <typename T>
std::vector<std::size_t>
sort_permutation(const std::vector<T>& vec) {
    std::vector<std::size_t> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(), [&](std::size_t i, std::size_t j) { return vec[i] < vec[j]; });
    return p;
}

// apply permutation
template <typename T>
void
apply_permutation_in_place(std::vector<T>& vec, const std::vector<std::size_t>& p) {
    std::vector<bool> done(vec.size());
    for (std::size_t i = 0; i < vec.size(); ++i) {
        if (done[i]) {
            continue;
        }
        done[i] = true;
        std::size_t prev_j = i;
        std::size_t j = p[i];
        while (i != j) {
            std::swap(vec[prev_j], vec[j]);
            done[j] = true;
            prev_j = j;
            j = p[j];
        }
    }
}

template <typename T>
vector<T>
sort_indexes(const vector<T>& v) {
    // initialize original index locations
    vector<T> idx(v.size());
    iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values
    stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

    return idx;
}

template <typename T>
vector<T>
sort_indexes(const vector<T>& v, const int begin_bias, const int end_bias) {
    // initialize original index locations
    vector<T> idx(end_bias - begin_bias);
    iota(idx.begin() + begin_bias, idx.begin() + end_bias, 0);

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values
    stable_sort(idx.begin() + begin_bias, idx.begin() + end_bias, [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

    return idx;
}

template <typename T>
void
print_set(const vector<T>& v) {
    if (v.size() == 0) {
        cout << "ERROR: EMPTY VECTOR!" << endl;
        return;
    }
    cout << "vertex in set: {";
    for (size_t i = 0; i < v.size() - 1; i++) {
        cout << v[i] << ", ";
    }
    cout << v.back() << "}" << endl;
}

void
logTime(timeval& begin, timeval& end, const string& log);

template <typename T>
double
countPrecision(const vector<T>& truth, const vector<T>& pred) {
    // 使用 unordered_set 进行快速查找
    std::unordered_set<T> pred_set(pred.begin(), pred.end());

    double num_right = 0;
    for (const auto& one : truth) {
        if (pred_set.find(one) != pred_set.end()) {
            num_right += 1;
        }
    }

    return num_right / truth.size();
}

// double
// countPrecision(const vector<uint32_t>& truth, const vector<uint32_t>& pred);

// double
// countPrecision(const vector<size_t>& truth, const vector<size_t>& pred);

double
countApproximationRatio(const vector<vector<float>>& raw_data, const vector<int>& truth, const vector<int>& pred,
                        const vector<float>& query);

void
print_memory();
void
record_memory(long long&);
#define _INT_MAX 2147483640

vector<uint32_t>
greedyNearest(const vector<vector<float>>& dpts, const vector<float> query, const int k_smallest);

vector<uint32_t>
greedyNearest(const vector<vector<float>>& dpts, const vector<float> query, const int k_smallest,
              const BitsetView bitset);


vector<int> greedyNearest(const vector<vector<float>> &dpts,
                          const vector<float> query, const int l_bound,
                          const int r_bound, const int k_smallest);

// void evaluateKNNG(const vector<vector<int>> &gt,
//                   const vector<vector<int>> &knng, const int K, double
//                   &recall, double &precision);

// Return a n-bits bitset data with random t bits set to true
inline std::vector<uint8_t>
GenRandomBitset(size_t n, size_t t) {
    assert(t >= 0 && t <= n);
    std::vector<bool> bits_shuffle(n, false);
    for (size_t i = 0; i < t; ++i) bits_shuffle[i] = true;
    std::mt19937 g(42);
    std::shuffle(bits_shuffle.begin(), bits_shuffle.end(), g);
    std::vector<uint8_t> data((n + 8 - 1) / 8, 0);
    for (size_t i = 0; i < n; ++i) {
        if (bits_shuffle[i]) {
            data[i >> 3] |= (0x1 << (i & 0x7));
        }
    }
    return data;
}

void
SaveToCSVRow(const string& path, const int idx, const int search_range, const int K_neighbor, const double& search_time,
             const vector<uint32_t>& gt, const vector<float>& pts);

void
SaveToCSVRow(const string& path, const string& method, const int idx, const int search_range, const int m, const int search_ef,
             const double& search_time, const double& recall, const int comps);

void
SaveVecUintToFile(const std::vector<std::vector<uint8_t>>& vec, const std::string& filename);

void
LoadVecUintFromFile(std::vector<std::vector<uint8_t>>& vec, const std::string& filename);
