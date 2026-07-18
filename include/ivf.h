/*
 * Created on Tue Jul 09 2024
 *
 * Manage inverted file index and interact with bitset
 *
 * Author: Chaoji Zuo
 * Email: chaoji.zuo@rutgers.edu
 *
 * Copyright (c) 2024 Rutgers University - Database Group
 */

#ifndef IVF_H
#define IVF_H

#include <memory>
#include <vector>

#include "bitset.h"

using std::vector;

class IVFIndex {
 public:
    IVFIndex() {
    }

    IVFIndex(const vector<vector<uint32_t>>& data, const size_t n) {
        clusters = data;
        labels = std::make_unique<vector<int>>(n, -1);  // Initialize with -1

        // Fill labels based on clusters
        for (size_t cluster_id = 0; cluster_id < clusters.size(); cluster_id++) {
            for (auto& point_id : clusters[cluster_id]) {
                labels->at(point_id) = cluster_id;
            }
        }

        // // initialize bitsets
        // for (auto& clus : clusters) {
        //     vector<uint8_t> vec_uint((clus.size() + 8 - 1) / 8, 0);
        //     for (size_t i = 0; i < clus.size(); i++) {
        //         if (all_bitset.test(clus[i])) {
        //             // set 1 if corresponding position is 1
        //             vec_uint[i >> 3] != (0x1 << (i & 0x7));
        //         }
        //     }
        //     BitsetView one_bitset(vec_uint.data(), clus.size());
        // }
    }

    IVFIndex(std::unique_ptr<vector<int>> data_labels, const size_t num_clusters) {
        labels = std::move(data_labels);
        clusters.resize(num_clusters);

        // NOTE: Don't need ivf_bitset_data for now, and it would be a waste of memory for large number of clusters
        // ivf_bitset_data.resize(num_clusters);

        // // Initialize bitset data
        // for (size_t i = 0; i < num_clusters; i++) {
        //     ivf_bitset_data[i].resize((labels->size() + 8 - 1) / 8, 1);
        // }

        // Build clusters and bitsets in one pass
        for (uint32_t i = 0; i < labels->size(); i++) {
            int cluster_id = labels->at(i);
            clusters[cluster_id].push_back(i);
            // ivf_bitset_data[cluster_id][i >> 3] &= ~(0x1 << (i & 0x7));
        }

        // // Create bitset views
        // for (auto& cluster_data : ivf_bitset_data) {
        //     ivf_bitset.emplace_back(cluster_data.data(), labels->size());
        // }
    }

    // initialize with ivf centroids
    IVFIndex(std::unique_ptr<vector<int>> data_labels, std::unique_ptr<vector<vector<float>>> data_centroids,
             const size_t num_clusters) {
        labels = std::move(data_labels);
        ivf_centroids = std::move(data_centroids);
        clusters.resize(num_clusters);
        for (uint32_t i = 0; i < labels->size(); i++) {
            int cluster_id = labels->at(i);
            clusters[cluster_id].push_back(i);
        }
    }

    // IVFIndex(const IVFIndex& obj) {
    //     clusters = obj.clusters;
    //     labels = std::make_unique<vector<int>>(*obj.labels);
    // }

    IVFIndex(const IVFIndex* obj) {
        clusters = obj->get_all_clusters();
        labels = std::make_unique<vector<int>>(obj->get_all_labels());
    }

    ~IVFIndex() {
    }

    virtual int
    get_cluster(uint32_t id) const {
        return labels->at(id);
    }

    virtual size_t
    get_num_clusters() const {
        return clusters.size();
    }

    virtual float
    get_avg_cluster_size() {
        if (average_cluster_size != 0) {
            return average_cluster_size;
        }

        float avg_size = 0;
        for (auto& clus : clusters) {
            avg_size += clus.size();
        }
        average_cluster_size = avg_size / clusters.size();
        return average_cluster_size;
    }

    const vector<uint32_t>&
    get_neighbors(uint32_t id) const {
        return clusters[get_cluster(id)];
    }

    // 优化1: 直接返回原始指针和大小，避免vector迭代器开销
    void
    get_cluster_neighbors_raw(size_t id, const uint32_t*& data_ptr, size_t& size) const {
        if (is_cache_accepted_elements) {
            const auto& neighbors = predicate_accepted_elements[id];
            data_ptr = neighbors.data();
            size = neighbors.size();
        } else {
            const auto& neighbors = clusters[id];
            data_ptr = neighbors.data();
            size = neighbors.size();
        }
    }

    // 优化2: 提供回调函数接口，允许内联优化
    template <typename Func>
    void
    for_each_neighbor(size_t cluster_id, Func callback) const {
        const auto& neighbors = clusters[cluster_id];
        const uint32_t* start = neighbors.data();
        const uint32_t* end = start + neighbors.size();

        // 手动展开循环可进一步优化性能
        for (const uint32_t* ptr = start; ptr < end; ++ptr) {
            callback(*ptr);
        }
    }

    const vector<uint32_t>&
    get_cluster_neighbors(size_t id) const {
        return clusters[id];
    }

    int*
    get_labels_ptr() const {
        return labels->data();
    }

    vector<float>
    get_cluster_centroid(size_t id) const {
        return ivf_centroids->at(id);
    }

    vector<vector<uint32_t>>
    get_all_clusters_copy() const {
        return clusters;
    }

    virtual void
    visited_append(vector<size_t>& cluster_visited_num, const int id) {
        cluster_visited_num[labels->at(id)]++;
    }

    const vector<vector<uint32_t>>&
    get_all_clusters() const {
        return clusters;
    }

    const vector<int>&
    get_all_labels() const {
        return *labels;
    }

    void
    set_predicate_accepted_elements(const BitsetView& bitset) {
        if (predicate_accepted_elements.size() != get_num_clusters()) {
            predicate_accepted_elements.resize(get_num_clusters());
            for (size_t i = 0; i < get_num_clusters(); i++) {
                predicate_accepted_elements[i].reserve(clusters[i].size() / 50);
            }
        } else {
            for (auto& bucket : predicate_accepted_elements) {
                bucket.clear();
            }
        }
        size_t total_size = bitset.size();
        is_cache_accepted_elements = true;

        if (bitset.count() > total_size / 2) {
            size_t pos = bitset.find_next_unset((size_t)-1);
            while (pos < total_size) {
                int cluster_id = labels->at(pos);
                predicate_accepted_elements[cluster_id].push_back(pos);
                pos = bitset.find_next_unset(pos);
            }
        } else {
            // fewer ones → scan set bits and fill the gaps
            size_t prev_set = (size_t)-1;
            size_t curr_set = bitset.find_next_set(prev_set);

            // zeros before the first 1
            for (size_t z = 0; z < curr_set && z < total_size; ++z) {
                int cid = labels->at(z);
                predicate_accepted_elements[cid].push_back((uint32_t)z);
            }

            // between each pair of set bits (including trailing)
            while (curr_set < total_size) {
                prev_set = curr_set;
                curr_set = bitset.find_next_set(prev_set);

                size_t start_zero = prev_set + 1;
                size_t end_zero = (curr_set < total_size ? curr_set : total_size);
                for (size_t z = start_zero; z < end_zero; ++z) {
                    int cid = labels->at(z);
                    predicate_accepted_elements[cid].push_back((uint32_t)z);
                }
            }
        }
    }

    bool is_cache_accepted_elements = false;

 protected:
    std::unique_ptr<vector<int>> labels;
    vector<vector<uint32_t>> clusters;
    std::unique_ptr<vector<vector<float>>> ivf_centroids;
    vector<vector<float>> ivf_centroids_data;

    vector<vector<uint32_t>> predicate_accepted_elements;

    float average_cluster_size = 0;
};

#endif /* IVF_H */
