/*
 * Created on Sun Jul 07 2024
 *
 * Use IVF index to help find connection in bitset filter
 *
 * Author: Chaoji Zuo
 * Email: chaoji.zuo@rutgers.edu
 *
 * Copyright (c) 2024 Rutgers University - Database Group
 */

#include <algorithm>
#include <ctime>
#include <iostream>
#include <memory>
#include <numeric>
#include <queue>
#include <vector>

#include "base_hnsw/hnswalg.h"
#include "base_hnsw/hnswlib.h"
#include "base_index.h"
#include "data_utils.h"
#include "ivf.h"
#include "reader.h"
#include "utils.h"

using namespace hnswlib;

template <typename dist_t>
class TwoPhaseBFS : public hnswlib::HierarchicalNSW<dist_t>, public baseindex::BaseIndex {

 public:
    TwoPhaseBFS(hnswlib::SpaceInterface<dist_t>* s, size_t max_elements,
                   size_t M = 16, size_t ef_construction = 200, size_t random_seed = 100)
        : HierarchicalNSW<dist_t>(s, max_elements, M, ef_construction, random_seed, false){};

    bool is_final_prune_on_knn = true;
    bool is_prune_every_reverse_add = false;
    bool is_add_nn_reverse = false;
    bool investigate_print = false;

    using BaseHNSW = hnswlib::HierarchicalNSW<dist_t>;

    // add search with bitset.
    // TODO: 为什么这里bitset不传引用？因为对象的构造很快吗。
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        typename BaseHNSW::CompareByFirst>
    searchBaseLayerST(tableint ep_id, const void* data_point, BitsetView bitset, size_t ef,
                      BaseFilterFunctor* isIdAllowed = nullptr,
                      BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const override {
        // Get visited list
        // uint32_t satisfied_count = bitset.count();
        // float selectivity = 1 - ((float)satisfied_count / this->cur_element_count);
        VisitedList* vl = this->visited_list_pool_->getFreeVisitedList();
        vl_type* visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        // Main priority queues - only keep the necessary ones
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                            typename BaseHNSW::CompareByFirst>
            candidate_set;  // For exploration - sorted by negative distance

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                            typename BaseHNSW::CompareByFirst>
            satisfied_candidates;  // For results - sorted by distance

#ifndef FAST_MODE
        timeval tt1, tt2;
        baseindex::SearchInfo current_info;
        gettimeofday(&tt1, NULL);
        current_info.sorted_visited_cluster.clear();
#endif


        // Basic counters
        size_t num_hop = 0;

        // Initialize with entry point
        dist_t lowerBound = std::numeric_limits<dist_t>::max();
        char* ep_data = this->getDataByInternalId(ep_id);
        dist_t ep_dist = this->fstdistfunc_(data_point, ep_data, this->dist_func_param_);

        visited_array[ep_id] = visited_array_tag;

        if (bitset.is_member(ep_id)) {
            satisfied_candidates.emplace(ep_dist, ep_id);
            lowerBound = ep_dist;
        }

        candidate_set.emplace(-ep_dist, ep_id);

        bool is_monotonic_phase = true;
        // moving average to judge the monotonicity
        size_t monotonicity_window = 3;
        float monotonicity_window_sum = ep_dist * monotonicity_window;
        vector<float> monotonicity_window_list(monotonicity_window, ep_dist);
        float last_sum = monotonicity_window_sum;
        double sum_candidate_dist = 0;
        // For estimate selectivity
        int total_score = 0;
        int total_explore = 0;

        float minimum_dis = ep_dist;
        float bfs_ep = ep_id;

        // std::unordered_map<uint32_t, float> centroids_dists;
        // float min_centroid_dist = std::numeric_limits<float>::max();

        int to_score_limit = this->maxM0_ * 6;
        int to_explore_limit = this->maxM0_;
        // to_score: the candidates that are satisfied the bitset filter
        tableint* to_score = new tableint[to_score_limit];
        // to_explore: the candidates that are not satisfied the bitset filter
        tableint* to_explore = new tableint[to_explore_limit];
        // Main search loop
        while (is_monotonic_phase && !candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            // Early termination check
            if (candidate_dist > lowerBound && satisfied_candidates.size() == ef) {
                break;
            }

	        size_t num_score = 0;
            size_t num_explore = 0;

            candidate_set.pop();
            tableint current_node_id = current_node_pair.second;
            int* data = (int*)this->get_linklist0(current_node_id);
            size_t size = this->getListCount((linklistsizeint*)data);


            num_hop++;

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
                _mm_prefetch((char*)(visited_array + *(data + j + 1)), _MM_HINT_T0);
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    if (!bitset.is_member(candidate_id)) {

                        if (num_explore < this->maxM0_) {
                            to_explore[num_explore++] = candidate_id;
                        }
                    } else {
                        to_score[num_score++] = candidate_id;
                    }
                }
            }


            this->search_distance_computations += num_score;
            // Note: 这里开始真正的距离计算
            for (size_t i = 0; i < num_score; i++) {
                _mm_prefetch(
                    this->data_level0_memory_ + to_score[i + 1] * this->size_data_per_element_ + this->offsetData_,
                    _MM_HINT_T0);
                char* currObj = this->getDataByInternalId(to_score[i]);
                dist_t dist = this->fstdistfunc_(data_point, currObj, this->dist_func_param_);

                if (dist < lowerBound || satisfied_candidates.size() < ef) {
                    candidate_set.emplace(-dist, to_score[i]);

                    satisfied_candidates.emplace(dist, to_score[i]);
                    sum_candidate_dist += dist;
                    if (satisfied_candidates.size() > ef) {
                        sum_candidate_dist -= satisfied_candidates.top().first;
                        satisfied_candidates.pop();
                    }
                    if (!satisfied_candidates.empty()) {
                        lowerBound = satisfied_candidates.top().first;
                    }

                    if (dist < minimum_dis) {
                        minimum_dis = dist;
                        bfs_ep = to_score[i];
                    }
                }
            }

            // Note: 在单调搜索阶段，不满足的点也会进行距离计算，来进行导航。这样做的目的是更快的到达邻近的区域。
            if (is_monotonic_phase) {
                this->search_distance_computations += num_explore;
#ifndef FAST_MODE
                current_info.distance_computations_graph_base += num_explore;
#endif
                for (size_t i = 0; i < num_explore; i++) {
#ifdef USE_SSE
                    _mm_prefetch(
                        this->data_level0_memory_ + to_explore[i] * this->size_data_per_element_ + this->offsetData_,
                        _MM_HINT_T0);
#endif
                    char* currObj = this->getDataByInternalId(to_explore[i]);
                    dist_t dist = this->fstdistfunc_(data_point, currObj, this->dist_func_param_);

                    if (dist < lowerBound || satisfied_candidates.size() < ef) {
                        candidate_set.emplace(-dist, to_explore[i]);

                        if (dist < minimum_dis) {
                            minimum_dis = dist;
                            bfs_ep = to_explore[i];
                        }
                    }
                }
                // // if the next candidate is worse than the current node, we stop the monotonic phase
                // if (!candidate_set.empty() && (-candidate_set.top().first) > candidate_dist) {
                //     is_monotonic_phase = false;
                // }

                // Note： 这里我用了一个moving average来判断是否需要结束单调搜索。
                // update the moving average distance
                last_sum = monotonicity_window_sum;
                monotonicity_window_sum -= monotonicity_window_list[0];
                monotonicity_window_list.push_back(-candidate_set.top().first);
                monotonicity_window_sum += monotonicity_window_list[monotonicity_window - 1];
                monotonicity_window_list.erase(monotonicity_window_list.begin());
                if (last_sum < monotonicity_window_sum) {
                    is_monotonic_phase = false;
                }
            }
        }

        /* In phase 2, use BFS starting from bfs_ep on HNSW to search the candidates*/
        std::queue<tableint> bfs_queue;
        bfs_queue.push(candidate_set.empty() ? bfs_ep : candidate_set.top().second);
        size_t bfs_evaluated = 0;
        while (!bfs_queue.empty()) {
            tableint current_node_id = bfs_queue.front();
            bfs_queue.pop();

            if (bfs_evaluated >= ef) {
                break;
            }

            int* data = (int*)this->get_linklist0(current_node_id);
            size_t size = this->getListCount((linklistsizeint*)data);

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    if (bitset.is_member(candidate_id)) {
                        char* currObj = this->getDataByInternalId(candidate_id);
                        dist_t dist = this->fstdistfunc_(data_point, currObj, this->dist_func_param_);
                        bfs_evaluated++;

                        if (dist < lowerBound || satisfied_candidates.size() < ef) {
                            candidate_set.emplace(-dist, candidate_id);

                            satisfied_candidates.emplace(dist, candidate_id);
                            if (satisfied_candidates.size() > ef) {
                                satisfied_candidates.pop();
                            }
                            if (!satisfied_candidates.empty()) {
                                lowerBound = satisfied_candidates.top().first;
                            }
                        }
                    }
                    bfs_queue.push(candidate_id);
                }
            }
        }

        delete[] to_score;
        delete[] to_explore;

#ifndef FAST_MODE
        gettimeofday(&tt2, NULL);
        AccumulateTime(tt1, tt2, current_info.graph_visit_time);
        current_info.num_hop = num_hop;
        search_info.graph_visit_time = current_info.graph_visit_time;
        search_info.num_hop = current_info.num_hop;
        search_info.distance_computations_graph_base = current_info.distance_computations_graph_base;
        search_info.distance_computations_cluster = current_info.distance_computations_cluster;
        search_info.distance_computations_graph_expand = current_info.distance_computations_graph_expand;
#endif

        this->visited_list_pool_->releaseVisitedList(vl);
        return satisfied_candidates;
    }

};
