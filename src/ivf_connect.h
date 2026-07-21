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

#ifdef __AVX2__
#include <immintrin.h>
// Increase chunk size to 8 for AVX2
size_t chunk_size = 8;
#elif defined(__AVX__)
#include <immintrin.h>
size_t chunk_size = 8;
#else
size_t chunk_size = 4;
#endif

using namespace hnswlib;

template <typename dist_t>
class IVFConnectHNSW : public hnswlib::HierarchicalNSW<dist_t>, public baseindex::BaseIndex {
 private:
    std::shared_ptr<IVFIndex> ivf_ptr;

 public:
    IVFConnectHNSW(std::shared_ptr<IVFIndex> ivf_index, hnswlib::SpaceInterface<dist_t>* s, size_t max_elements,
                   size_t M = 16, size_t ef_construction = 200, size_t random_seed = 100)
        : HierarchicalNSW<dist_t>(s, max_elements, M, ef_construction, random_seed, false), ivf_ptr(ivf_index) {};

    bool is_final_prune_on_knn = true;
    bool is_prune_every_reverse_add = false;
    bool is_add_nn_reverse = false;
    bool investigate_print = false;

    // Sensitivity-study knobs (R1-D4, R2-D1, R3-D3).
    // monotonicity_window_ is the window w used by the moving-average monotonicity check
    // that decides when to leave the navigation phase. Default 3 matches the paper.
    size_t monotonicity_window_ = 3;
    // replacement_ratio_ is the factor used to size the two-hop / IVF expansion budget,
    // i.e. x = ef * replacement_ratio_. Default 0.1 matches the paper (x = ef/10).
    float replacement_ratio_ = 0.1f;

    // C1 component-ablation toggles (R1-D5, R2-D2). Defaults = full \sys.
    //   ablation_force_navigate_: never leave the navigation phase -> the
    //     search degrades to predicate-constrained BFS (InlineFilter-HNSW).
    //   ablation_disable_twohop_: skip the two-hop neighbor expansion.
    //   ablation_disable_ivf_:    skip the IVF cluster augmentation.
    bool ablation_force_navigate_ = false;
    bool ablation_disable_twohop_ = false;
    bool ablation_disable_ivf_    = false;

    // C3 latency breakdown (R2-D2). Per-phase nanosecond accumulators, populated
    // by searchBaseLayerST and reset between ef blocks via reset_phase_timers().
    mutable uint64_t phase_graph_ns_  = 0;  // visiting current node's neighbor list + bitset check
    mutable uint64_t phase_twohop_ns_ = 0;  // two-hop expansion
    mutable uint64_t phase_ivf_ns_    = 0;  // IVF cluster augmentation
    mutable uint64_t phase_dist_ns_   = 0;  // distance computation (both satisfied + monotonic explore)
    mutable uint64_t phase_movavg_ns_ = 0;  // moving-average / monotonicity-check maintenance
    void reset_phase_timers() const {
        phase_graph_ns_ = phase_twohop_ns_ = phase_ivf_ns_ = phase_dist_ns_ = phase_movavg_ns_ = 0;
    }

    using BaseHNSW = hnswlib::HierarchicalNSW<dist_t>;

    // add search with bitset.
    // TODO: 为什么这里bitset不传引用？因为对象的构造很快吗。
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        typename BaseHNSW::CompareByFirst>
    searchBaseLayerOld(tableint ep_id, const void* data_point, BitsetView bitset, size_t ef,
                      BaseFilterFunctor* isIdAllowed = nullptr,
                      BaseSearchStopCondition<dist_t>* stop_condition = nullptr) {
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

        // For tracking clusters
        vector<char> visited_clusters(ivf_ptr->get_num_clusters(), 0);
        vector<size_t> cluster_visited_cursor(ivf_ptr->get_num_clusters(), 0);

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
            // if (search_info.is_investigate) {
            //     bool is_target = std::find(search_info.groundtruth_nns.begin(), search_info.groundtruth_nns.end(),
            //                                ep_id) != search_info.groundtruth_nns.end();
            //     if (is_target) {
            //         search_info.true_positive++;
            //     }
            // }
        }

        candidate_set.emplace(-ep_dist, ep_id);

        bool is_monotonic_phase = true;
        // moving average to judge the monotonicity
        size_t monotonicity_window = this->monotonicity_window_;
        float monotonicity_window_sum = ep_dist * monotonicity_window;
        vector<float> monotonicity_window_list(monotonicity_window, ep_dist);
        float last_sum = monotonicity_window_sum;
        double sum_candidate_dist = 0;
        // For estimate selectivity
        int total_score = 0;
        int total_explore = 0;

        // std::unordered_map<uint32_t, float> centroids_dists;
        // float min_centroid_dist = std::numeric_limits<float>::max();

        int to_score_limit = this->maxM0_ * 6;
        int to_explore_limit = this->maxM0_;
        // to_score: the candidates that are satisfied the bitset filter
        tableint* to_score = new tableint[to_score_limit];
        // to_explore: the candidates that are not satisfied the bitset filter
        tableint* to_explore = new tableint[to_explore_limit];
        // Main search loop
        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            // Early termination check
            if (candidate_dist > lowerBound && satisfied_candidates.size() == ef) {
                break;
            }

            // Note: this is for investigate mode, which will calculate the recall and precision at every step.
            // if (search_info.is_investigate) {
            //     bool is_target = std::find(search_info.groundtruth_nns.begin(), search_info.groundtruth_nns.end(),
            //                                current_node_pair.second) != search_info.groundtruth_nns.end();
            //     search_info.savePath(current_node_pair.second, candidate_dist, is_target,
            //                          bitset.is_member(current_node_pair.second), this->search_distance_computations);
            // }

            // vector<tableint> to_score();
            // vector<tableint> to_explore();
            //tableint* to_score = new tableint[this->maxM0_ * 6];
            //tableint* to_explore = new tableint[this->maxM0_];
            //memset(to_score, 0, sizeof(tableint)*this->maxM0_*6);
	    //memset(to_explore, 0, sizeof(tableint)*this->maxM0_);
	    size_t num_score = 0;
            size_t num_explore = 0;

            candidate_set.pop();
            tableint current_node_id = current_node_pair.second;
            int* data = (int*)this->get_linklist0(current_node_id);
            size_t size = this->getListCount((linklistsizeint*)data);

            // 这个好像没什么用，速度没变化，所以我注释掉了
// #ifdef USE_SSE
            // _mm_prefetch((char*)(visited_array + *(data + 1)), _MM_HINT_T0);
            // _mm_prefetch((char*)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
            // _mm_prefetch(this->data_level0_memory_ + (*(data + 1)) * this->size_data_per_element_ +
            // this->offsetData_,
            //              _MM_HINT_T0);
// #endif
            num_hop++;
            uint32_t visit_count = 0;

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
// #ifdef USE_SSE
                // _mm_prefetch((char*)(visited_array + *(data + j + 1)), _MM_HINT_T0);
// #endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visit_count++;
                    visited_array[candidate_id] = visited_array_tag;

                    if (!bitset.is_member(candidate_id)) {
                        // is this necessary?
                        if (num_explore < this->maxM0_) {
                            to_explore[num_explore++] = candidate_id;
                        }
                    } else {
                        to_score[num_score++] = candidate_id;
                    }
                }
            }

#ifndef FAST_MODE
            current_info.distance_computations_graph_base += num_score;
            current_info.num_satisfied_visited += num_score;
            current_info.num_unsatisfied_visited += num_explore;
#endif

            // Note: Check if we need to do two-hop expansion, this is tricky and hard to optimize.
            // float filtered_amount = (float)num_explore / (float)(num_explore + num_score + 1);
            // int max_to_score_count =
            //     (int)(size *
            //           std::min(10.0f,
            //                    1.0f / (1.0f - filtered_amount)));  // this is from Lucene, but it doesn't work well.
            // max_to_score_count = std::min(max_to_score_count, (int)this->maxM0_ * 2);
            // bool flag_neighbor_expand = filtered_amount > 0.10f && num_score > 0;

            // flag_neighbor_expand = num_explore > 0;
            int max_to_score_count = num_score + num_explore;

            total_score += num_score;
            total_explore += num_explore;

            // scan explore neighbors, when most of them are satisfied.
            //             if (num_explore < 4) {
            //                 this->search_distance_computations += num_explore;
            //                 for (size_t i = 0; i < num_explore; i++) {
            // #ifdef USE_SSE
            //                     _mm_prefetch(
            //                         this->data_level0_memory_ + to_explore[i] * this->size_data_per_element_ +
            //                         this->offsetData_, _MM_HINT_T0);
            // #endif
            //                     char* currObj = this->getDataByInternalId(to_explore[i]);
            //                     dist_t dist = this->fstdistfunc_(data_point, currObj, this->dist_func_param_);

            //                     if (dist < lowerBound || satisfied_candidates.size() < ef) {
            //                         candidate_set.emplace(-dist, to_explore[i]);
            //                     }
            //                 }
            //             } else

            // Note: 这个地方的重点是选择多少个two-hop neighbors，对于small selectivity,
            // 这里快点结束可以节省访问图的时间。一般而言，two hop
            // neighbors的质量会比较高，所以这里多选几个会比较好，但是对于large
            // selectivity，选太多了又会消耗太多时间，所以比较tricky，还有优化空间。 Note: when still in the monotonic
            // phase, we don't need to do two-hop expansion. because they are still far away from the query vector.
            if (!is_monotonic_phase && num_explore > 0) {
                // Note: this is for two-hop expansion.
                max_to_score_count = num_score + num_explore * 1;
                // max_to_score_count = num_score + num_explore * std::max(1.0, ((int)num_hop - 10) / 10.0);

                // Note: 这里在尝试利用距离信息来判断需要取多少个two hop neighbors。
                if (candidate_dist < (sum_candidate_dist / satisfied_candidates.size())) {
                    max_to_score_count = num_score + num_explore * static_cast<size_t>(ef * this->replacement_ratio_);
                }

                // Note: avoid overflow
                max_to_score_count = std::min(max_to_score_count, to_score_limit);

                bool is_two_hop_expand = true;
                // float expected_unvisited_satisfied_twohop = num_explore * selectivity * this->avg_deg * (((float)visit_count / (float)size));
                // is_two_hop_expand = expected_unvisited_satisfied_twohop > 1.0f;
                // float estimated_selectivity = total_score / (total_score + total_explore);
                // use the selectivity to judge whether to do two-hop expansion
                // if there is no satisfied two-hop neighbors, then don't do two-hop expansion
                // if ((estimated_selectivity * size * size) < (num_explore / 4)) {
                //     is_two_hop_expand = false;
                // }

                // if (search_info.distance_computations_graph_expand > 5000) {
                //     is_two_hop_expand = false;
                // }

                // TODO: use monotonicity to judge. 如果当前的candidate_dist 小于 平均值，那么就多选几个two-hop
                // neighbors。

                size_t num_two_hop_score = 0;
                if (is_two_hop_expand) {
                    // Note: grab all two hop neighbors
                    for (size_t j = 0; j < num_explore && num_score < max_to_score_count; j++){
                        int candidate_id = to_explore[j];
                    // for (size_t j = 1; j <= size && num_score < max_to_score_count; j++) {
                        // int candidate_id = *(data + j);
// #ifdef USE_SSE
//                         _mm_prefetch((char*)(visited_array + *(data + j + 1)), _MM_HINT_T0);
// #endif

                        // if (j == 8 && num_two_hop_score == 0) {
                        //     // early stop if no two-hop neighbors， small difference in speed, so I comment it out.
                        //     break;
                        // }
                        if (j < num_explore - 1)
                        {
                            // _mm_prefetch((char*)(visited_array + to_explore[j+1]), _MM_HINT_T0);
#ifdef USE_SSE
                            _mm_prefetch((char*)(this->data_level0_memory_ + to_explore[j+1] * this->size_data_per_element_), _MM_HINT_T0);
#endif
                        }

                        int* data_two_hop = (int*)this->get_linklist0(candidate_id);
                        size_t size_two_hop = this->getListCount((linklistsizeint*)data_two_hop);
                        for (size_t k = 1; k <= size_two_hop; k++) {
                            int two_hop_neighbor = *(data_two_hop + k);
                            if (visited_array[two_hop_neighbor] != visited_array_tag) {
                                visited_array[two_hop_neighbor] = visited_array_tag;
                                if (bitset.is_member(two_hop_neighbor)) {
                                    to_score[num_score++] = two_hop_neighbor;
                                    num_two_hop_score++;
#ifndef FAST_MODE
                                    current_info.distance_computations_graph_expand++;
#endif
                                }
                                //  else {
                                //     search_info.distance_computations_graph_expand++;
                                // }
                            }
                            if (num_score >= max_to_score_count) {
                                break;
                            }
                        }
                    }
}
                // else {
                // Cluster exploration if no satisfied neighbors found，这个在有的时候有提升
                // if (filtered_amount > 0.9) {
                if (true && num_score < max_to_score_count) {

                    auto cluster_id = ivf_ptr->get_cluster(current_node_id);
                    size_t neighbor_padding = 1;
                    // bool found = false;
                    // uint32_t count_ivf = 0;
                    // while (!found && neighbor_padding <= size) {
                        // Note: 这里的while循环是为了跳过那些已经被完全访问过的cluster
                    // while (num_score < max_to_score_count && neighbor_padding <= size) {
                        // Note: If the current point's cluster has been fully visited, visit the cluster containing its
                        // next neighbor
                        // while (visited_clusters[cluster_id] == 1 && neighbor_padding <= size) {
                        //     cluster_id = ivf_ptr->get_cluster(*(data + neighbor_padding));
                        //     neighbor_padding++;
                        // }

                        if (visited_clusters[cluster_id] == 0) {  // Note: this cluster is not fully visited
                            const uint32_t* cluster_ptr;
                            size_t cluster_size;
                            // if (centroids_dists.find(cluster_id) == centroids_dists.end()) {
                            //     auto centroid = ivf_ptr->get_cluster_centroid(cluster_id);
                            //     float centroid_dist =
                            //         this->fstdistfunc_(data_point, (char*)centroid.data(), this->dist_func_param_);
                            //     this->search_distance_computations++;
                            //     centroids_dists[cluster_id] = centroid_dist;
                            //     if (centroid_dist < min_centroid_dist) {
                            //         min_centroid_dist = centroid_dist;
                            //     }
                            // }
                            // if (centroids_dists[cluster_id] > 1.5 * min_centroid_dist)
                            // {
                            //     visited_clusters[cluster_id] = 1;
                            //     continue;
                            // }

                            // Note: 这里的优化很重要，cluster里邻居的bitset scan可能会耗时很长，存在优化空间
                            ivf_ptr->get_cluster_neighbors_raw(cluster_id, cluster_ptr, cluster_size);
                            // Note:
                            // 这个cursor是用来记录当前cluster里已经访问了多少个点，如果已经访问了全部的点，那么就标记为已访问。避免重复访问。
                            size_t cluster_cursor = cluster_visited_cursor[cluster_id];
                            int found_candidate = 0;

                            for (; cluster_cursor < cluster_size && max_to_score_count > num_score; cluster_cursor++) {
                                uint32_t cluster_node = cluster_ptr[cluster_cursor];
                                // if (cluster_cursor < cluster_size - 1)
                                // {
                                //     _mm_prefetch((char*)(visited_array + cluster_ptr[cluster_cursor + 1]), _MM_HINT_T0);
                                // }
                                // Note: 这里的优化很重要，cluster里邻居的bitset scan可能会耗时很长，存在优化空间
                                // 这里是只把符合bitset的点标记visited，感觉有待商榷。
                                if (visited_array[cluster_node] != visited_array_tag && bitset.is_member(cluster_node)) {
                                    visited_array[cluster_node] = visited_array_tag;
                                    // found = true;
                                    // count_ivf++;
    #ifndef FAST_MODE
                                    current_info.distance_computations_cluster++;
    #endif
                                    to_score[num_score++] = cluster_node;
                                    found_candidate++;
                                    if (num_score >= max_to_score_count) {
                                        break;
                                    }
                                }
                            }
                            cluster_visited_cursor[cluster_id] =
                                cluster_cursor;  // Note: update the cursor to record the position
                            if ((cluster_cursor + 1) >= cluster_size) {
                                visited_clusters[cluster_id] = 1;  // Note: mark the cluster as visited
                                search_info.distance_computations_cluster++;
    #ifndef FAST_MODE
                                current_info.sorted_visited_cluster.emplace_back(cluster_id);
    #endif
                            }
                        }

                    // }
                }
            }

            // cout << "hop: " << num_hop << " current dist: " << candidate_dist << " lowerBound: " << lowerBound <<
            // endl
            //      << "graph_base: " << current_info.distance_computations_graph_base
            //      << " graph_expand: " << current_info.distance_computations_graph_expand
            //      << " cluster_scan: " << current_info.distance_computations_cluster << endl;
            // cout << "score: " << num_score << " max_score: " << max_to_score_count << " explore: " << num_explore
            //      << endl;
            // cout << endl;

            // Prefetch first few vectors before distance computation
            // #ifdef USE_SSE
            //             _mm_prefetch(this->data_level0_memory_ + to_score[0] * this->size_data_per_element_ +
            //             this->offsetData_,
            //                          _MM_HINT_T0);
            // #endif

            this->search_distance_computations += num_score;
            // Note: 这里开始真正的距离计算
            for (size_t i = 0; i < num_score; i++) {
#ifdef USE_SSE
                // Prefetch next vector if available
                // if (i + 4 < num_score) {
                //     _mm_prefetch(this->getDataByInternalId(to_score[i + 4]), _MM_HINT_T0);
                // }
                // _mm_prefetch(this->getDataByInternalId(to_score[i + 1]), _MM_HINT_T0);
                _mm_prefetch(
                    this->data_level0_memory_ + to_score[i + 1] * this->size_data_per_element_ + this->offsetData_,
                    _MM_HINT_T0);
#endif
                char* currObj = this->getDataByInternalId(to_score[i]);
                dist_t dist = this->fstdistfunc_(data_point, currObj, this->dist_func_param_);

                if (search_info.is_investigate) {
                    bool is_target = std::find(search_info.groundtruth_nns.begin(), search_info.groundtruth_nns.end(),
                                               to_score[i]) != search_info.groundtruth_nns.end();
                    if (is_target) {
                        search_info.true_positive++;
                    }
                }

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


    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        typename BaseHNSW::CompareByFirst>
    searchBaseLayerST(tableint ep_id, const void* data_point, BitsetView bitset, size_t ef,
                      BaseFilterFunctor* isIdAllowed = nullptr,
                      BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const override {
        VisitedList* vl = this->visited_list_pool_->getFreeVisitedList();
        vl_type* visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        baseindex::SearchInfo current_info;
        current_info.sorted_visited_cluster.clear();

        // Main priority queues - only keep the necessary ones
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                            typename BaseHNSW::CompareByFirst>
            candidate_set;  // For exploration - sorted by negative distance

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                            typename BaseHNSW::CompareByFirst>
            satisfied_candidates;  // For results - sorted by distance

        // For tracking clusters
        vector<char> visited_clusters(ivf_ptr->get_num_clusters(), 0);
        vector<size_t> cluster_visited_cursor(ivf_ptr->get_num_clusters(), 0);

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
        size_t monotonicity_window = this->monotonicity_window_;
        float monotonicity_window_sum = ep_dist * monotonicity_window;
        vector<float> monotonicity_window_list(monotonicity_window, ep_dist);
        float last_sum = monotonicity_window_sum;
        double sum_candidate_dist = 0;
        // For estimate selectivity
        int total_score = 0;
        int total_explore = 0;

        int to_score_limit = this->maxM0_ * 6;
        int to_explore_limit = this->maxM0_;
        tableint* to_score = new tableint[to_score_limit];
        tableint* to_explore = new tableint[to_explore_limit];

        using clk = std::chrono::high_resolution_clock;
        auto ns_since = [](clk::time_point t0) {
            return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                clk::now() - t0).count();
        };

        // Main search loop
        while (!candidate_set.empty()) {
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
            uint32_t visit_count = 0;

            // ---- Phase: GRAPH (neighbor list scan + bitset partition) ----
            auto t_graph = clk::now();
            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visit_count++;
                    visited_array[candidate_id] = visited_array_tag;

                    if (!bitset.is_member(candidate_id)) {
                        // is this necessary?
                        if (num_explore < this->maxM0_) {
                            to_explore[num_explore++] = candidate_id;
                        }
                    } else {
                        to_score[num_score++] = candidate_id;
                    }
                }
            }
            phase_graph_ns_ += ns_since(t_graph);

            current_info.distance_computations_graph_base += num_score;
            current_info.num_satisfied_visited += num_score;
            current_info.num_unsatisfied_visited += num_explore;

            int max_to_score_count = num_score + num_explore;

            total_score += num_score;
            total_explore += num_explore;

            if (!is_monotonic_phase && num_explore > 0) {
                // Note: this is for two-hop expansion.
                max_to_score_count = num_score + num_explore * 1;

                if (candidate_dist < (sum_candidate_dist / satisfied_candidates.size())) {
                    max_to_score_count = num_score + num_explore * static_cast<size_t>(ef * this->replacement_ratio_);
                }

                max_to_score_count = std::min(max_to_score_count, to_score_limit);

                bool is_two_hop_expand = !ablation_disable_twohop_;
                size_t num_two_hop_score = 0;
                if (is_two_hop_expand) {
                    // ---- Phase: TWOHOP (two-hop neighbor expansion) ----
                    auto t_twohop = clk::now();
                    // Note: grab all two hop neighbors
                    for (size_t j = 0; j < num_explore && num_score < max_to_score_count; j++){
                        int candidate_id = to_explore[j];
                        if (j < num_explore - 1)
                        {
                            // _mm_prefetch((char*)(visited_array + to_explore[j+1]), _MM_HINT_T0);
#ifdef USE_SSE
                            _mm_prefetch((char*)(this->data_level0_memory_ + to_explore[j+1] * this->size_data_per_element_), _MM_HINT_T0);
#endif
                        }

                        int* data_two_hop = (int*)this->get_linklist0(candidate_id);
                        size_t size_two_hop = this->getListCount((linklistsizeint*)data_two_hop);
                        for (size_t k = 1; k <= size_two_hop; k++) {
                            int two_hop_neighbor = *(data_two_hop + k);
                            if (visited_array[two_hop_neighbor] != visited_array_tag) {
                                visited_array[two_hop_neighbor] = visited_array_tag;
                                if (bitset.is_member(two_hop_neighbor)) {
                                    to_score[num_score++] = two_hop_neighbor;
                                    num_two_hop_score++;
                                    current_info.distance_computations_graph_expand++;
                                }
                            }
                            if (num_score >= max_to_score_count) {
                                break;
                            }
                        }
                    }
                    phase_twohop_ns_ += ns_since(t_twohop);
                }

                if (!ablation_disable_ivf_ && num_score < max_to_score_count) {
                    // ---- Phase: IVF (cluster augmentation) ----
                    auto t_ivf = clk::now();

                    auto cluster_id = ivf_ptr->get_cluster(current_node_id);
                    size_t neighbor_padding = 1;

                    if (visited_clusters[cluster_id] == 0) {  // Note: this cluster is not fully visited
                        const uint32_t* cluster_ptr;
                        size_t cluster_size;

                        ivf_ptr->get_cluster_neighbors_raw(cluster_id, cluster_ptr, cluster_size);
                        size_t cluster_cursor = cluster_visited_cursor[cluster_id];
                        int found_candidate = 0;

                        for (; cluster_cursor < cluster_size && max_to_score_count > num_score; cluster_cursor++) {
                            uint32_t cluster_node = cluster_ptr[cluster_cursor];
                            if (visited_array[cluster_node] != visited_array_tag && bitset.is_member(cluster_node)) {
                                visited_array[cluster_node] = visited_array_tag;
                                current_info.distance_computations_cluster++;
                                to_score[num_score++] = cluster_node;
                                found_candidate++;
                                if (num_score >= max_to_score_count) {
                                    break;
                                }
                            }
                        }
                        cluster_visited_cursor[cluster_id] = cluster_cursor;
                        if ((cluster_cursor + 1) >= cluster_size) {
                            visited_clusters[cluster_id] = 1;
                            search_info.distance_computations_cluster++;
                            current_info.sorted_visited_cluster.emplace_back(cluster_id);
                        }
                    }
                    phase_ivf_ns_ += ns_since(t_ivf);
                }
            }


            // ---- Phase: DIST (distance computation for satisfying candidates) ----
            auto t_dist = clk::now();
            this->search_distance_computations += num_score;
            // Note: 这里开始真正的距离计算
            for (size_t i = 0; i < num_score; i++) {
#ifdef USE_SSE
                _mm_prefetch(
                    this->data_level0_memory_ + to_score[i + 1] * this->size_data_per_element_ + this->offsetData_,
                    _MM_HINT_T0);
#endif
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
                }
            }
            // Note: 在单调搜索阶段，不满足的点也会进行距离计算，来进行导航。这样做的目的是更快的到达邻近的区域。
            if (is_monotonic_phase) {
                this->search_distance_computations += num_explore;
                current_info.distance_computations_graph_base += num_explore;
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
                    }
                }
            }
            phase_dist_ns_ += ns_since(t_dist);

            // ---- Phase: MOVAVG (moving-average / monotonicity-check maintenance) ----
            if (is_monotonic_phase) {
                auto t_mv = clk::now();
                // update the moving average distance
                last_sum = monotonicity_window_sum;
                monotonicity_window_sum -= monotonicity_window_list[0];
                monotonicity_window_list.push_back(-candidate_set.top().first);
                monotonicity_window_sum += monotonicity_window_list[monotonicity_window - 1];
                monotonicity_window_list.erase(monotonicity_window_list.begin());
                if (!ablation_force_navigate_ && last_sum < monotonicity_window_sum) {
                    is_monotonic_phase = false;
                }
                phase_movavg_ns_ += ns_since(t_mv);
            }
        }

        delete[] to_score;
        delete[] to_explore;


        current_info.num_hop = num_hop;
        search_info.graph_visit_time = current_info.graph_visit_time;
        search_info.num_hop = current_info.num_hop;
        search_info.distance_computations_graph_base = current_info.distance_computations_graph_base;
        search_info.distance_computations_cluster = current_info.distance_computations_cluster;
        search_info.distance_computations_graph_expand = current_info.distance_computations_graph_expand;

        this->visited_list_pool_->releaseVisitedList(vl);
        return satisfied_candidates;
    }



    // Note: 这个是predicate push down的版本，传的是raw predicate 而不是bitset，但核心搜索部分是旧版本的了，需要更新。
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        typename BaseHNSW::CompareByFirst>
    searchBaseLayerST(tableint ep_id, const void* data_point, const PredicateBaseView& predicate, size_t ef,
                      BaseFilterFunctor* isIdAllowed = nullptr,
                      BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const override {
        // Get visited list
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

        // For tracking clusters
        vector<char> visited_clusters(ivf_ptr->get_num_clusters(), 0);
        vector<size_t> cluster_visited_cursor(ivf_ptr->get_num_clusters(), 0);

        // Basic counters
        size_t num_hop = 0;

        // Initialize with entry point
        dist_t lowerBound = std::numeric_limits<dist_t>::max();
        char* ep_data = this->getDataByInternalId(ep_id);
        dist_t ep_dist = this->fstdistfunc_(data_point, ep_data, this->dist_func_param_);

        visited_array[ep_id] = visited_array_tag;

        if (predicate.is_member(ep_id)) {
            satisfied_candidates.emplace(ep_dist, ep_id);
            lowerBound = ep_dist;
        }

        candidate_set.emplace(-ep_dist, ep_id);

        // Main search loop
        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            // Early termination check
            if (candidate_dist > lowerBound && satisfied_candidates.size() == ef) {
                break;
            }

            // vector<tableint> to_score();
            // vector<tableint> to_explore();
            tableint* to_score = new tableint[this->maxM0_ * 2];
            tableint* to_explore = new tableint[this->maxM0_];
            size_t num_score = 0;
            size_t num_explore = 0;

            candidate_set.pop();
            tableint current_node_id = current_node_pair.second;
            int* data = (int*)this->get_linklist0(current_node_id);
            size_t size = this->getListCount((linklistsizeint*)data);

#ifdef USE_SSE
            // _mm_prefetch((char*)(visited_array + *(data + 1)), _MM_HINT_T0);
            // _mm_prefetch((char*)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
            // _mm_prefetch(this->data_level0_memory_ + (*(data + 1)) * this->size_data_per_element_ +
            // this->offsetData_,
            //              _MM_HINT_T0);
#endif
            num_hop++;

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
#ifdef USE_SSE
                _mm_prefetch((char*)(visited_array + *(data + j + 1)), _MM_HINT_T0);
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    if (!predicate.is_member(candidate_id)) {
                        if (num_explore < this->maxM0_) {
                            to_explore[num_explore++] = candidate_id;
                        }
#ifndef FAST_MODE
                        current_info.num_satisfied_visited++;
#endif
                    } else {
                        to_score[num_score++] = candidate_id;
#ifndef FAST_MODE
                        current_info.distance_computations_graph_base++;
                        // if (search_info.groundtruth_nns.size() > 0) {
                        //     if (std::find(search_info.groundtruth_nns.begin(), search_info.groundtruth_nns.end(),
                        //                   candidate_id) != search_info.groundtruth_nns.end()) {
                        //     current_info.distance_computations_graph_base++;
                        // }
                        // }
#endif
                    }
                }
            }

            // Check if we need to do two-hop expansion
            float filtered_amount = (float)num_explore / (float)size;
            int max_to_score_count = (int)(size * std::min(10.0f, 1.0f / (1.0f - filtered_amount)));
            max_to_score_count = std::min(max_to_score_count, (int)this->maxM0_ * 2);
            bool flag_neighbor_expand = filtered_amount > 0.10f && num_score > 0;

            // Two-hop expansion if needed
            if (flag_neighbor_expand && max_to_score_count > num_score) {
                for (size_t i = 0; i < num_explore && max_to_score_count > num_score; i++) {
                    // if scan two list but still not satisfied, then stop
                    if (num_score == 0 && i >= 1) {
                        break;
                    }
                    int* data_two_hop = (int*)this->get_linklist0(to_explore[i]);
#ifdef USE_SSE
                    // _mm_prefetch((char*)(data_two_hop + 1), _MM_HINT_T0);  // Prefetch first neighbor
                    // _mm_prefetch(
                    //     this->data_level0_memory_ + (to_explore[i]) * this->size_data_per_element_ +
                    //     this->offsetData_, _MM_HINT_T0);  // Prefetch node data
#endif
                    size_t size_two_hop = this->getListCount((linklistsizeint*)data_two_hop);

                    for (size_t j = 1; j <= size_two_hop && max_to_score_count > num_score; j++) {
                        int two_hop_neighbor = *(data_two_hop + j);
#ifdef USE_SSE
                        _mm_prefetch((char*)(visited_array + *(data_two_hop + j + 1)), _MM_HINT_T0);
                        // // Prefetch the data of the next neighbor to check
                        // _mm_prefetch(this->data_level0_memory_ +
                        //                  *(data_two_hop + j + 1) * this->size_data_per_element_ +
                        //                  this->offsetData_,
                        //              _MM_HINT_T0);

#endif
                        if (visited_array[two_hop_neighbor] != visited_array_tag) {
                            visited_array[two_hop_neighbor] = visited_array_tag;
                            if (!predicate.is_member(two_hop_neighbor)) {
#ifndef FAST_MODE
                                current_info.num_unsatisfied_visited++;
#endif
                                continue;
                            }
                            to_score[num_score++] = two_hop_neighbor;

#ifndef FAST_MODE
                            current_info.distance_computations_graph_expand++;
                            // if (search_info.groundtruth_nns.size() > 0) {
                            //     if (std::find(search_info.groundtruth_nns.begin(),
                            //     search_info.groundtruth_nns.end(),
                            //                   two_hop_neighbor) != search_info.groundtruth_nns.end()) {
                            //         current_info.distance_computations_graph_expand++;
                            //     }
                            // }
#endif
                            if (num_score >= max_to_score_count) {
                                break;
                            }
                        }
                    }
                }
            }

            // Cluster exploration if no satisfied neighbors found
            if (num_score < 4) {
                for (size_t i = 0; i < num_explore && max_to_score_count > num_score; i++) {
                    auto cluster_id = ivf_ptr->get_cluster(to_explore[i]);
                    if (visited_clusters[cluster_id] == 0) {
                        const uint32_t* cluster_ptr;
                        size_t cluster_size;
                        ivf_ptr->get_cluster_neighbors_raw(cluster_id, cluster_ptr, cluster_size);
                        size_t cluster_cursor = cluster_visited_cursor[cluster_id];

                        // #ifdef USE_SSE
                        //                         // Prefetch first few cluster elements if available
                        //                         for (size_t pf = 0; pf < 4 && cluster_cursor + pf < cluster_size;
                        //                         pf++) {
                        //                             uint32_t prefetch_node = cluster_ptr[cluster_cursor + pf];
                        //                             _mm_prefetch((char*)(visited_array + prefetch_node),
                        //                             _MM_HINT_T0); _mm_prefetch(this->data_level0_memory_ +
                        //                             prefetch_node * this->size_data_per_element_ +
                        //                             this->offsetData_, _MM_HINT_T0);
                        //                         }
                        // #endif
                        for (; cluster_cursor < cluster_size && max_to_score_count > num_score; cluster_cursor++) {
                            uint32_t cluster_node = cluster_ptr[cluster_cursor];
// #ifdef USE_SSE
//                             // Prefetch ahead in the cluster
//                             if (cluster_cursor + 4 < cluster_size) {
//                                 uint32_t next_node = cluster_ptr[cluster_cursor + 4];
// _mm_prefetch((char*)(visited_array + cluster_ptr[cluster_cursor + 1]), _MM_HINT_T0);
//_mm_prefetch(this->data_level0_memory_ +
//                                 next_node * this->size_data_per_element_ +
//                                 this->offsetData_, _MM_HINT_T0);
//                             }
// #endif
#ifndef FAST_MODE
                            if (visited_array[cluster_node] != visited_array_tag &&
                                !predicate.is_member(cluster_node)) {
                                current_info.num_skipped++;
                            }
#endif
                            if (visited_array[cluster_node] != visited_array_tag && predicate.is_member(cluster_node)) {
                                visited_array[cluster_node] = visited_array_tag;
#ifndef FAST_MODE
                                current_info.distance_computations_cluster++;
                                //     current_info.distance_computations_cluster++;
#endif
                                to_score[num_score++] = cluster_node;
                            }
                        }

                        // cout << "scan cluster: " << cluster_id << " cluster_cursor: " << cluster_cursor
                        //      << " cluster_size: " << cluster_size << endl;
                        cluster_visited_cursor[cluster_id] = cluster_cursor;
                        if (cluster_cursor >= cluster_size) {
                            visited_clusters[cluster_id] = 1;
#ifndef FAST_MODE
                            current_info.sorted_visited_cluster.emplace_back(cluster_id);
#endif
                        }
                    }
                }
            }

            // cout << "hop: " << num_hop << " current dist: " << candidate_dist << " lowerBound: " << lowerBound <<
            // endl
            //      << "graph_base: " << current_info.distance_computations_graph_base
            //      << " graph_expand: " << current_info.distance_computations_graph_expand
            //      << " cluster_scan: " << current_info.distance_computations_cluster << endl;
            // cout << "score: " << num_score << " max_score: " << max_to_score_count << " explore: " << num_explore
            //      << endl;
            // cout << endl;

            // Prefetch first few vectors before distance computation
            // #ifdef USE_SSE
            //             _mm_prefetch(this->data_level0_memory_ + to_score[0] * this->size_data_per_element_ +
            //             this->offsetData_,
            //                          _MM_HINT_T0);
            // #endif

            for (size_t i = 0; i < num_score; i++) {
#ifdef USE_SSE
                // Prefetch next vector if available
                // if (i + 4 < num_score) {
                //     _mm_prefetch(this->getDataByInternalId(to_score[i + 4]), _MM_HINT_T0);
                // }
                // _mm_prefetch(this->getDataByInternalId(to_score[i + 1]), _MM_HINT_T0);
                _mm_prefetch(
                    this->data_level0_memory_ + to_score[i + 1] * this->size_data_per_element_ + this->offsetData_,
                    _MM_HINT_T0);
#endif
                char* currObj = this->getDataByInternalId(to_score[i]);
                dist_t dist = this->fstdistfunc_(data_point, currObj, this->dist_func_param_);

                if (dist < lowerBound || satisfied_candidates.size() < ef) {
                    candidate_set.emplace(-dist, to_score[i]);

                    satisfied_candidates.emplace(dist, to_score[i]);
                    if (satisfied_candidates.size() > ef) {
                        satisfied_candidates.pop();
                    }
                    if (!satisfied_candidates.empty()) {
                        lowerBound = satisfied_candidates.top().first;
                    }
                }
            }
        }

#ifndef FAST_MODE
        gettimeofday(&tt2, NULL);
        AccumulateTime(tt1, tt2, current_info.graph_visit_time);
        current_info.num_hop = num_hop;
        search_info = current_info;
#endif

        this->visited_list_pool_->releaseVisitedList(vl);
        return satisfied_candidates;
    }
};
