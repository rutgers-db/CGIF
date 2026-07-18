/*
 * Created on Sun Jul 07 2024
 *
 * Base index class for control index information and search information
 *
 * Author: Chaoji Zuo
 * Email: chaoji.zuo@rutgers.edu
 *
 * Copyright (c) 2024 Rutgers University - Database Group
 */

#pragma once

#include <fstream>
#include <iostream>

#include "bitset.h"
#include "data_utils.h"
#include "ivf.h"

namespace baseindex {

static unsigned const default_K = 16;
static unsigned const default_ef_construction = 400;

struct SearchInfo {
    SearchInfo() {
        reset();
    };

    // SearchInfo(const DataWrapper* data, const IndexParams* index_params, const Searchparams* search_params,
    //            const string& meth, const string& ver) {
    //     data_wrapper = data;
    //     index = index_params;
    //     search = search_params;
    //     version = ver;
    //     method = meth;
    //     // save_path = "../exp/search/" + version + "-" + method + "-" +
    //     //             data_wrapper->dataset + "-" +
    //     //             std::to_string(data_wrapper->data_size) + ".csv";
    //     Path(ver);
    // };

    void
    reset() {
        graph_cal_dist_time = 0;
        graph_other_time = 0;
        cluster_cal_dist_time = 0;
        cluster_other_time = 0;
        graph_visit_time = 0;

        num_satisfied_visited = 0;
        num_skipped = 0;
        num_unsatisfied_visited = 0;
        num_hop = 0;
        num_visited_clusters = 0;
        num_replaced_by_cluster_neighbor = 0;
        num_cluster_satisfied_visited = 0;

        distance_computations_graph_base = 0;
        distance_computations_cluster = 0;
        distance_computations_graph_expand = 0;
    }

    void
    swap(SearchInfo* other) {
        graph_cal_dist_time = other->graph_cal_dist_time;
        graph_other_time = other->graph_other_time;
        cluster_cal_dist_time = other->cluster_cal_dist_time;
        cluster_other_time = other->cluster_other_time;
        graph_visit_time = other->graph_visit_time;

        num_satisfied_visited = other->num_satisfied_visited;
        num_skipped = other->num_skipped;
        num_unsatisfied_visited = other->num_unsatisfied_visited;
        num_hop = other->num_hop;
        num_visited_clusters = other->num_visited_clusters;
        num_replaced_by_cluster_neighbor = other->num_replaced_by_cluster_neighbor;
        num_cluster_satisfied_visited = other->num_cluster_satisfied_visited;
    }

    // const IndexParams* index;
    // const Searchparams* search;

    string method;
    string save_path;
    string version;

    // double time;
    // double precision;
    // int query_id;
    // unsigned break_counter;
    // double internal_search_time;
    // size_t visited_num;
    // double fetch_nns_time;
    // double cal_dist_time;
    // double other_process_time;
    // size_t total_comparison;
    // size_t visited_num;
    // size_t path_counter;

    int idx;
    int query_id;
    int query_range;
    int query_k;
    int search_ef;
    mutable int step;
    mutable int member_step;
    mutable int true_positive;

    size_t distance_computations_graph_base;
    size_t distance_computations_cluster;
    size_t distance_computations_graph_expand;

    size_t num_satisfied_visited;
    size_t num_unsatisfied_visited;
    size_t num_skipped;
    size_t num_replaced_by_cluster_neighbor;
    size_t num_cluster_satisfied_visited;

    size_t num_hop;
    size_t num_visited_clusters;

    double graph_cal_dist_time;
    double graph_other_time;
    double graph_visit_time;
    double cluster_cal_dist_time;
    double cluster_other_time;

    vector<size_t> sorted_visited_cluster;

    vector<uint32_t> groundtruth_nns;

    bool is_investigate = false;

    void
    path(const string& ver, const string& dataset, const size_t data_size) {
        version = ver;
        if (is_investigate) {
            save_path = "../exp/log/path/" + version + "-" + dataset + "-" + std::to_string(data_size) + ".csv";
            std::cout << "Save result to :" << save_path << std::endl;
        }
    };

    void
    savePath(const int pid, const float dist, const bool is_target, const bool is_member, const size_t num_dist_comp) {
        std::ofstream file;
        file.open(save_path, std::ios_base::app);
        if (file) {
            file << step++ << "," << pid << "," << dist << "," << idx << "," << query_id << "," << query_range << ","
                 << search_ef << "," << is_target << "," << (is_member ? member_step++ : -1) << "," << true_positive
                 << "," << num_dist_comp << std::endl;
        }
        file.close();
    }

    // void
    // SavePathInvestigate(const float v1, const float v2, const float v3, const float v4, bool is_new_row = false)
    // {
    //     string investigate_path = "../exp/search/" + version + "-" + method + "-" + data_wrapper->dataset + "-" +
    //                               std::to_string(data_wrapper->data_size) + "-invetigate-path.csv";
    //     std::ofstream file;
    //     file.open(investigate_path, std::ios_base::app);

    //     if (is_new_row) {
    //         file << "\n";
    //     } else if (file) {
    //         file << v1 << "," << v2 << "," << v3 << "," << v4;
    //         file << "\n";
    //     }
    //     file.close();
    // };
};

class BaseIndex {
 public:
    BaseIndex() {
    }
    // virtual void buildIndex() = 0;
    virtual ~BaseIndex() {
    }

    mutable SearchInfo search_info;

    double sort_time;
    double nn_build_time;
    int num_search_comparison;

    // Indexing parameters
    struct IndexParams {
        // original params in hnsw
        unsigned K;
        unsigned ef_construction;
        unsigned random_seed;

        IndexParams() : K(default_K), ef_construction(default_ef_construction), random_seed(2023) {
        }
    };

    struct Searchparams {
        // vector<float> weights;
        // vector<std::pair<float, float>> weights;

        unsigned internal_search_K;
    };

    struct IndexInfo {};
};

}  // namespace baseindex
