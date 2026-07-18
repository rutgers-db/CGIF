/*
 * Created on Fri Jul 05 2024
 *
 * Author: Chaoji Zuo
 * Email: chaoji.zuo@rutgers.edu
 *
 * Copyright (c) 2024 Rutgers University - Database Group
 */

#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "data_utils.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/index_io.h"
// #include "faiss_connect.h"
#include "faiss_ivf_adapter.h"
#include "ivf.h"
#include "ivf_connect.h"

#include "twophase_bfs.h"

using std::cin;
using std::cout;
using std::endl;

void
SaveOneRes(const std::vector<uint32_t>& vec, const std::string& path, const int K) {
    std::ofstream ofs(path, std::ios::out | std::ios::binary | std::ios::app);
    const uint32_t N = vec.size();
    assert(vec.size() == K);
    ofs.write(reinterpret_cast<char const*>(&vec[0]), K * sizeof(uint32_t));
    ofs.close();
}

void
log_result_recorder(const std::map<int, std::pair<float, float>>& result_recorder,
                    const std::map<int, std::pair<int, int>>& comparison_recorder, const int amount) {
    for (auto it : result_recorder) {
        cout << std::setiosflags(ios::fixed) << std::setprecision(4) << "range: " << it.first
             << "\t recall: " << it.second.first / (amount / result_recorder.size())
             << "\t QPS: " << std::setprecision(0) << (amount / result_recorder.size()) / it.second.second
             << "\t Comps: " << comparison_recorder.at(it.first).first / (amount / result_recorder.size()) << " / "
             << comparison_recorder.at(it.first).second / (amount / result_recorder.size()) << endl;
    }
}

void
log_result_recorder(const std::map<int, std::pair<float, float>>& result_recorder,
                    const std::map<int, size_t[5]>& comparison_recorder, const std::map<int, double[5]>& time_recorder,
                    const std::map<int, size_t[5]>& num_recorder, const std::map<int, double[5]>& cluster_recorder,
                    const std::map<int, float>& counter_recorder, const int amount, const int query_num) {
    for (auto it : result_recorder) {
        cout << std::setiosflags(ios::fixed) << std::setprecision(4) << "range: " << it.first
             << "\t recall: " << it.second.first / counter_recorder.at(it.first) << "\t QPS: " << std::setprecision(0)
             << counter_recorder.at(it.first) / it.second.second
             << "\t Comps: " << comparison_recorder.at(it.first)[0] / counter_recorder.at(it.first) << " / "
             << comparison_recorder.at(it.first)[1] / counter_recorder.at(it.first) << " / "
             << comparison_recorder.at(it.first)[2] / counter_recorder.at(it.first) << " : "
             << comparison_recorder.at(it.first)[3] / counter_recorder.at(it.first) << "  + "
             << comparison_recorder.at(it.first)[4] / counter_recorder.at(it.first) << endl;
    }

#ifndef FAST_MODE
    cout << endl << endl;
    cout << std::setiosflags(ios::fixed) << std::setprecision(4) << "range"
         << "\t dist graph"
         << "\t visit graph"
         << "\t other graph"
         << "\t dist cluster"
         << "\t other cluster" << endl;
    for (auto it : result_recorder) {
        cout << std::setiosflags(ios::fixed) << std::setprecision(4) << it.first << "\t\t"
             << time_recorder.at(it.first)[0] / 1.0 << "\t\t" << time_recorder.at(it.first)[1] / 1.0 << "\t\t"
             << time_recorder.at(it.first)[2] << "\t\t" << time_recorder.at(it.first)[3] << "\t\t"
             << time_recorder.at(it.first)[4] << endl;
    }

    cout << endl;
    cout << std::setiosflags(ios::fixed) << std::setprecision(4) << "range"
         << "\t sataisfied access"
         << "\t unsatsified access"
         << "\t skipped" << "\t # of hop" << "\t # of scan clusters" << endl;

    for (auto it : result_recorder) {
        cout << std::setiosflags(ios::fixed) << std::setprecision(0) << it.first << "\t\t"
             << num_recorder.at(it.first)[0] / counter_recorder.at(it.first) << "\t\t\t"
             << num_recorder.at(it.first)[1] / counter_recorder.at(it.first) << "\t\t\t"
             << num_recorder.at(it.first)[2] / counter_recorder.at(it.first) << "\t\t"
             << num_recorder.at(it.first)[3] / counter_recorder.at(it.first) << "\t\t"
             << num_recorder.at(it.first)[4] / counter_recorder.at(it.first) << endl;
    }

    cout << endl;
    cout << std::setiosflags(ios::fixed) << std::setprecision(4) << "range"
         << "\t # of scan clusters"
         << "\t scan cluster recall"
         << "\t top-10 recall" << "\t # cluster neighbors" << endl;

    for (auto it : cluster_recorder) {
        cout << std::setiosflags(ios::fixed) << std::setprecision(3) << it.first << "\t\t"
             << cluster_recorder.at(it.first)[0] / counter_recorder.at(it.first) << "\t\t\t"
             << cluster_recorder.at(it.first)[1] / counter_recorder.at(it.first) << "\t\t"
             << cluster_recorder.at(it.first)[2] / counter_recorder.at(it.first) << "\t\t"
             << std::setiosflags(ios::fixed) << std::setprecision(0)
             << cluster_recorder.at(it.first)[3] / counter_recorder.at(it.first) << " / "
             << cluster_recorder.at(it.first)[4] / counter_recorder.at(it.first) << std::setprecision(4) << endl;
    }
#endif
}

void
exp(string dataset, int data_size, string dataset_path, string method, string query_path, string ivf_index_path,
    string graph_path, string data_tag_path = "", string query_tag_path = "", string gt_path = "",
    bool log_mode = false, string detail_path = "", bool bm25_mode = false,
    int monotonicity_window = 3, float replacement_ratio = 0.1f, int num_runs = 5,
    int build_M = 32, int build_ef_construction = 400, bool build_only = false,
    float pruning_alpha = 1.0f, bool ablation_force_navigate = false,
    bool ablation_disable_twohop = false, bool ablation_disable_ivf = false) {
    cout << "method: " << method << endl;

    int query_num = 1000;
    if (dataset == "local") {
        query_num = 10;
    }
    // yfcc-arbitrary (C5): ~61626 single-filter queries; raise the cap so the
    // search loop's `one_id >= query_num` guard never skips a valid query.
    if (dataset == "yfcc-arbitrary") {
        query_num = 100000;
    }
    int query_k = 10;
    if (log_mode) {
        query_num = 3;
    }
    // Determine dataset type based on dataset name
    string dataset_type = "";
    if (bm25_mode) {
        dataset_type = "bm25";
    }

    bool detail_mode = false;
    if (!detail_path.empty()) {
        detail_mode = true;
    }

    // Create appropriate data wrapper
    std::unique_ptr<DataWrapper> data_wrapper;
    data_wrapper = createDataWrapperByType(dataset, dataset_type, query_num, query_k, data_size, dataset_path,
                                           query_path, data_tag_path, query_tag_path, gt_path);

    // IVF + HNSW
    size_t M = static_cast<size_t>(build_M);
    size_t ef_construction = static_cast<size_t>(build_ef_construction);
    vector<int> search_ef_list = {10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 150};
    //vector<int> search_ef_list = {70};
    {
        cout << "HNSW with bitmap, ivf connect satisfied cluster point during searching" << endl;

        // Load IVF index
        std::unique_ptr<vector<int>> ivf_data(new vector<int>());
        cout << "ivf_index_path: " << ivf_index_path << endl;
        if (ivf_index_path.empty()) {
            ivf_index_path =
                "../exp/ivf/" + dataset + "/" + dataset + "-hierarchical-clusters-base_1024-n_sub_16-level_0.npy";
        }
        std::shared_ptr<FaissIVFAdapter> faiss_adapter = nullptr;
        std::shared_ptr<IVFIndex> ivf_index = nullptr;

        if (ivf_index_path.find("faiss") != std::string::npos) {
            cout << "Loading faiss index from: " << ivf_index_path << endl;
            std::unique_ptr<faiss::Index> faiss_index(faiss::read_index(ivf_index_path.c_str()));
            faiss_adapter = std::make_shared<FaissIVFAdapter>(std::move(faiss_index));
            ivf_index = faiss_adapter;
            // release faiss_index
            faiss_index.reset();
        } else {
            read_ivf_clusters(ivf_index_path, ivf_data.get());
            cout << "ivf_data size: " << ivf_data->size() << endl;
            int num_clusters = *std::max_element(ivf_data->cbegin(), ivf_data->cend()) + 1;
            ivf_index = std::make_shared<IVFIndex>(std::move(ivf_data), num_clusters);
        }

        cout << "IVF index: " << ivf_index->get_avg_cluster_size() * ivf_index->get_num_clusters() << " points, "
             << ivf_index->get_num_clusters() << " clusters" << endl;

        // Create space and index
        std::unique_ptr<hnswlib::L2Space> space(new hnswlib::L2Space(data_wrapper->data_dim));
        std::unique_ptr<IVFConnectHNSW<float>> index(
            new IVFConnectHNSW<float>(ivf_index, space.get(), 2 * data_wrapper->data_size, M, ef_construction, 11));
        index->pruning_alpha_ = pruning_alpha;

        if (graph_path.empty() || build_only) {
            {
                cout << "Building graph (parallel, M=" << M
                     << ", efConstruction=" << ef_construction
                     << ", alpha=" << pruning_alpha << ")..." << endl;
                auto build_t0 = std::chrono::high_resolution_clock::now();
                #pragma omp parallel for schedule(dynamic, 64)
                for (size_t i = 0; i < data_wrapper->data_size; ++i) {
                    index->addPointFixedId(data_wrapper->nodes.at(i).data(), i);
                }
                auto build_t1 = std::chrono::high_resolution_clock::now();
                double build_seconds = std::chrono::duration<double>(build_t1 - build_t0).count();
                cout << "BUILD_TIME_SECONDS: " << build_seconds << endl;
                cout << "BUILD_M: " << M << "  EF_CONSTRUCTION: " << ef_construction
                     << "  DATASET: " << dataset << "  N: " << data_size << endl;

                // Save index to file (path can be overridden via -graph_path when build_only is set)
                string save_to_path = !graph_path.empty()
                    ? graph_path
                    : ("/data/local/repo/bitset-filtering-search/exp/graph/hnsw-graph-" + dataset + "-M0-" +
                       std::to_string(index->maxM0_) + "-" + std::to_string(data_size) + ".hnsw");
                cout << "Saving graph to: " << save_to_path << endl;
                index->saveIndex(save_to_path);
                if (build_only) {
                    return;
                }
                exit(0);
            }
        } else {
            // Load index from path
            cout << "Load index from: " << graph_path << endl;
            index->loadIndex(graph_path, space.get(), 2 * data_wrapper->data_size);
            cout << "max M0: " << index->maxM0_ << endl;
        }

        // Sensitivity-study knobs (R1-D4, R2-D1, R3-D3).
        index->monotonicity_window_ = static_cast<size_t>(monotonicity_window);
        index->replacement_ratio_ = replacement_ratio;
        cout << "monotonicity_window (w): " << index->monotonicity_window_
             << ", replacement_ratio (x/ef): " << index->replacement_ratio_ << endl;

        // C1 component-ablation toggles (R1-D5, R2-D2).
        index->ablation_force_navigate_ = ablation_force_navigate;
        index->ablation_disable_twohop_ = ablation_disable_twohop;
        index->ablation_disable_ivf_    = ablation_disable_ivf;
        cout << "ablation: force_navigate=" << ablation_force_navigate
             << " disable_twohop=" << ablation_disable_twohop
             << " disable_ivf=" << ablation_disable_ivf << endl;

        data_wrapper->nodes.clear();

        // Querying
        double next_unset_position_time = 0;
        double find_next_unset_time = 0;

        // index->search_info.is_investigate = true;
        // index->search_info.path("cgif-monotone-update-window3-update0604-2", dataset, data_wrapper->data_size);
        // query_num = 100;
        // data_wrapper->query_num = query_num;
        // search_ef_list = {120};

        for (auto search_ef : search_ef_list) {
            index->setEf(search_ef);
            double total_precision = 0;
            timeval tt3, tt4;
            std::map<int, std::pair<float, float>> result_recorder;  // first->precision, second->query_time
            std::map<int, size_t[5]> comparison_recorder;            // distance computation tracking
            std::map<int, double[5]> time_recorder;
            std::map<int, size_t[5]> num_recorder;
            std::map<int, double[5]> cluster_recorder;
            std::map<int, float> counter_recorder;

            // Repeat each ef configuration num_runs times so we can report a
            // trimmed-mean QPS (drop slowest+fastest runs, average the rest).
            // num_runs is a function parameter (default 5). With num_runs==3 the
            // trim leaves a single middle sample (the median).
            std::vector<double> per_run_total_time(num_runs, 0.0);
            int per_run_valid_queries = 0;

            // C3 latency breakdown: reset phase timers before this ef block
            index->reset_phase_timers();

            for (int run = 0; run < num_runs; ++run) {
            int run_valid_queries = 0;
            for (size_t idx = 0; idx < data_wrapper->query_ids.size(); idx++) {
                index->search_distance_computations = 0;
                index->cluster_distance_computations = 0;
                index->metric_distance_computations = 0;
                index->search_info.reset();
                int one_id = data_wrapper->query_ids.at(idx);
                if (one_id >= data_wrapper->query_num) {
                    continue;
                }
                if (data_wrapper->groundtruth.at(idx).size() == 0) {
                    continue;
                }
                auto bitset = BitsetView(data_wrapper->query_bitsets.at(idx).data(), data_wrapper->data_size);
                int query_range = data_wrapper->data_size - bitset.count();
                int raw_query_range = data_wrapper->data_size - bitset.count();

                // if (query_range < data_wrapper->data_size / 2) {
                //     auto start = std::chrono::high_resolution_clock::now();
                //     ivf_index->set_predicate_accepted_elements(bitset);
                //     auto end = std::chrono::high_resolution_clock::now();
                //     std::chrono::duration<double> duration = end - start;
                //     find_next_unset_time += duration.count();

                //     if (idx % 10000 == 0) {
                //         cout << "idx: " << idx << " query_range: " << query_range << endl;
                //         cout << "find_next_unset_time: " << find_next_unset_time << endl;
                //     }
                //     continue;
                // }

                // Group ranges for SPMAT format

                query_range = queryRangeAdjustment(query_range, dataset);
                if (log_mode) {
                    if (query_range != 500000 && query_range != 100000 && query_range != 10000 &&
                        query_range != 50000) {
                        continue;
                    }
                }

                if (log_mode && run == 0) {
                    cout << "idx: " << idx << " query_id: " << one_id << " query_range: " << query_range << endl;
                }

                // Set debug print flag
                if (one_id == 1) {
                    index->investigate_print = false;
                } else {
                    index->investigate_print = false;
                }
                if (index->investigate_print) {
                    cout << endl << endl << "Range: " << query_range << endl;
                }
                index->search_info.groundtruth_nns = data_wrapper->groundtruth.at(idx);
                // bitset.init_filter_id_map();
                // Search
                ivf_index->is_cache_accepted_elements = false;

                {
                    index->search_info.idx = idx;
                    index->search_info.query_id = one_id;
                    index->search_info.query_range = query_range;
                    index->search_info.search_ef = search_ef;
                    index->search_info.step = 0;
                    index->search_info.member_step = 0;
                    index->search_info.true_positive = 0;
                }

                vector<uint32_t> query_result;
                // gettimeofday(&tt3, NULL);
                auto search_start = std::chrono::high_resolution_clock::now();
                // if (raw_query_range <= 20000) {
                // // this is very useful for the query with low selectivity
                //     ivf_index->set_predicate_accepted_elements(bitset);
                // }
                auto res = index->searchKnn(data_wrapper->querys.at(one_id).data(), query_k, bitset);
                auto search_end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> search_diff = search_end - search_start;
                while (res.size() > 0) {
                    query_result.emplace_back(res.top().second);
                    res.pop();
                }
                // gettimeofday(&tt4, NULL);
                // auto query_time = CountTime(tt3, tt4);
                auto query_time = search_diff.count();
                per_run_total_time[run] += query_time;
                run_valid_queries++;
                // Analyze cluster statistics for groundtruth
                {
                    vector<size_t> groundtruth_buckets;
#ifndef FAST_MODE
                    for (auto ele : data_wrapper->groundtruth.at(idx)) {
                        groundtruth_buckets.emplace_back(ivf_index->get_cluster(ele));
                    }
#endif
                    auto top_10_bucket_recall = 0.0;
                    auto access_bucket_recall = 0.0;
#ifndef FAST_MODE
                    if (index->search_info.sorted_visited_cluster.size() > 0) {
                        vector<size_t> most_access_buckets(
                            index->search_info.sorted_visited_cluster.begin(),
                            index->search_info.sorted_visited_cluster.begin() +
                                std::min(10, (int)index->search_info.sorted_visited_cluster.size()));
                        top_10_bucket_recall = countPrecision(groundtruth_buckets, most_access_buckets);

                        vector<size_t> access_buckets(index->search_info.sorted_visited_cluster.begin(),
                                                      index->search_info.sorted_visited_cluster.begin() +
                                                          index->search_info.num_visited_clusters);
                        access_bucket_recall = countPrecision(groundtruth_buckets, access_buckets);
                    }
#endif

                    cluster_recorder[query_range][0] += index->search_info.num_visited_clusters;
                    cluster_recorder[query_range][1] += access_bucket_recall;
                    cluster_recorder[query_range][2] += top_10_bucket_recall;
                    cluster_recorder[query_range][3] += index->search_info.num_replaced_by_cluster_neighbor;
                    cluster_recorder[query_range][4] += index->search_info.num_cluster_satisfied_visited;
                    if (log_mode) {
                        cout << "truth buckets:";
                        print_set(groundtruth_buckets);
                    }
                }

                // Record query statistics

                auto precision = countPrecision(data_wrapper->groundtruth.at(idx), query_result);
                if (data_wrapper->groundtruth.at(idx).size() == 0) {
                    precision = 1;
                }
                if (log_mode && run == 0) {
                    cout << "groundtruth: ";
                    print_set(data_wrapper->groundtruth.at(idx));
                    cout << "query_result: ";
                    print_set(query_result);
                }

                counter_recorder[query_range] += 1;
                result_recorder[query_range].first += precision;
                result_recorder[query_range].second += query_time;
                comparison_recorder[query_range][0] += index->search_info.distance_computations_graph_base;
                comparison_recorder[query_range][1] += index->search_info.distance_computations_cluster;
                comparison_recorder[query_range][2] += index->search_info.distance_computations_graph_expand;
                comparison_recorder[query_range][3] += index->search_distance_computations;
                comparison_recorder[query_range][4] += index->metric_distance_computations;
                time_recorder[query_range][0] += index->search_info.graph_cal_dist_time;
                time_recorder[query_range][1] += index->search_info.graph_visit_time;
                time_recorder[query_range][2] += index->search_info.graph_other_time;
                time_recorder[query_range][3] += index->search_info.cluster_cal_dist_time;
                time_recorder[query_range][4] += index->search_info.cluster_other_time;

                num_recorder[query_range][0] += index->search_info.num_satisfied_visited;
                num_recorder[query_range][1] += index->search_info.num_unsatisfied_visited;
                num_recorder[query_range][2] += index->search_info.num_skipped;
                num_recorder[query_range][3] += index->search_info.num_hop;
                num_recorder[query_range][4] += index->search_info.num_visited_clusters;

                if (detail_mode && run == 0) {
                    SaveToCSVRow(detail_path, "ivf_connect_hnsw", idx, raw_query_range, index->maxM0_, search_ef,
                                 query_time, precision, index->search_distance_computations);
                }
            }
            // record once (all runs touch the same set of queries)
            if (run == 0) {
                per_run_valid_queries = run_valid_queries;
            }
            }  // end of num_runs trials

            // Compute and output statistics
            float recall_avg = 0;
            float query_time_avg = 0;
            float avg_distance_computation = 0;
            int valid_query_num = 0;
            for (auto it : result_recorder) {
                recall_avg += it.second.first;
                query_time_avg += it.second.second;
                valid_query_num += counter_recorder.at(it.first);
            }
            for (auto it: comparison_recorder){
                avg_distance_computation += (it.second[3] + it.second[4]);
            }
            recall_avg /= valid_query_num;
            avg_distance_computation /= valid_query_num;

            // Trimmed-mean QPS: drop slowest and fastest of the num_runs trials.
            std::vector<double> sorted_times = per_run_total_time;
            std::sort(sorted_times.begin(), sorted_times.end());
            double trimmed_time_sum = 0.0;
            int trimmed_count = 0;
            for (int r = 1; r + 1 < num_runs; ++r) {
                trimmed_time_sum += sorted_times[r];
                trimmed_count++;
            }
            double trimmed_mean_time = trimmed_count > 0
                ? trimmed_time_sum / trimmed_count
                : sorted_times[num_runs / 2];
            float qps_trimmed = per_run_valid_queries > 0 && trimmed_mean_time > 0
                ? static_cast<float>(per_run_valid_queries) / static_cast<float>(trimmed_mean_time)
                : 0.0f;
            // For reference, also report the legacy (sum-over-all-runs) QPS.
            float qps_legacy = (query_time_avg > 0)
                ? static_cast<float>(valid_query_num) / static_cast<float>(query_time_avg)
                : 0.0f;

            cout << "Average Recall: " << std::setprecision(6)  << recall_avg << endl;
            cout << "Average QPS: " << qps_trimmed << endl;
            cout << "Average QPS (legacy, no trim): " << qps_legacy << endl;
            cout << "Per-run total time (s): ";
            for (auto t : per_run_total_time) cout << t << " ";
            cout << " | trimmed-mean: " << trimmed_mean_time << endl;
            cout << "Average Distance Computations: " << avg_distance_computation << endl;
            // C3 latency breakdown: per-phase ns averaged over (num_runs * valid_queries)
            double denom = (double)num_runs * (double)per_run_valid_queries;
            if (denom > 0) {
                double g = index->phase_graph_ns_  / denom;
                double t = index->phase_twohop_ns_ / denom;
                double v = index->phase_ivf_ns_    / denom;
                double d = index->phase_dist_ns_   / denom;
                double m = index->phase_movavg_ns_ / denom;
                double tot = g + t + v + d + m;
                cout << "Phase Breakdown (ns/query): "
                     << "graph=" << (long long)g
                     << " twohop=" << (long long)t
                     << " ivf="    << (long long)v
                     << " dist="   << (long long)d
                     << " movavg=" << (long long)m
                     << " total="  << (long long)tot
                     << " | %: graph=" << (g/tot*100)
                     << " twohop=" << (t/tot*100)
                     << " ivf="    << (v/tot*100)
                     << " dist="   << (d/tot*100)
                     << " movavg=" << (m/tot*100)
                     << endl;
            }
            cout << "Result for Search ef: " << search_ef << endl;
            cout << "========================" << endl;
            log_result_recorder(result_recorder, comparison_recorder, time_recorder, num_recorder, cluster_recorder,
                                counter_recorder, valid_query_num, query_num);
            cout << "========================" << endl << endl;

            if ((dataset != "deep" && dataset != "wiki-image") && search_ef == search_ef_list.front()) {
                for (auto it : counter_recorder) {
                    cout << "range: " << it.first << "  " << it.second << endl;
                }
                cout << endl;
            }
        }
        cout << endl;
    }
}

int
main(int argc, char** argv) {
    string dataset = "biggraph";
    int data_size = 100000;
    string dataset_path = "";
    string method = "";
    string query_path = "";
    string ivf_index_path = "";
    string graph_path = "";
    string data_tag_path = "", query_tag_path = "", gt_path = "";
    bool log_mode = false;
    string detail_path = "";
    bool bm25_mode = false;
    int monotonicity_window = 3;
    float replacement_ratio = 0.1f;
    int num_runs = 5;
    int build_M = 32;
    int build_ef_construction = 400;
    bool build_only = false;
    float pruning_alpha = 1.0f;
    bool ablation_force_navigate = false;
    bool ablation_disable_twohop = false;
    bool ablation_disable_ivf = false;

    // Parse command line arguments
    for (int i = 0; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-dataset")
            dataset = string(argv[i + 1]);
        if (arg == "-N")
            data_size = atoi(argv[i + 1]);
        if (arg == "-dataset_path")
            dataset_path = string(argv[i + 1]);
        if (arg == "-method")
            method = string(argv[i + 1]);
        if (arg == "-query_path")
            query_path = string(argv[i + 1]);
        if (arg == "-ivf_index_path")
            ivf_index_path = string(argv[i + 1]);
        if (arg == "-graph_path")
            graph_path = string(argv[i + 1]);
        if (arg == "-data_tag_path")
            data_tag_path = string(argv[i + 1]);
        if (arg == "-query_tag_path")
            query_tag_path = string(argv[i + 1]);
        if (arg == "-gt_path")
            gt_path = string(argv[i + 1]);
        if (arg == "-log_mode")
            log_mode = true;
        if (arg == "-detail_path")
            detail_path = string(argv[i + 1]);
        if (arg == "-bm25_mode")
            bm25_mode = true;
        if (arg == "-monotonicity_window")
            monotonicity_window = atoi(argv[i + 1]);
        if (arg == "-replacement_ratio")
            replacement_ratio = static_cast<float>(atof(argv[i + 1]));
        if (arg == "-num_runs")
            num_runs = atoi(argv[i + 1]);
        if (arg == "-M")
            build_M = atoi(argv[i + 1]);
        if (arg == "-efConstruction")
            build_ef_construction = atoi(argv[i + 1]);
        if (arg == "-build_only")
            build_only = true;
        if (arg == "-alpha")
            pruning_alpha = static_cast<float>(atof(argv[i + 1]));
        if (arg == "-ablation_force_navigate")
            ablation_force_navigate = true;
        if (arg == "-ablation_disable_twohop")
            ablation_disable_twohop = true;
        if (arg == "-ablation_disable_ivf")
            ablation_disable_ivf = true;
    }
    if (num_runs < 1) num_runs = 1;

    exp(dataset, data_size, dataset_path, method, query_path, ivf_index_path, graph_path, data_tag_path, query_tag_path,
        gt_path, log_mode, detail_path, bm25_mode, monotonicity_window, replacement_ratio, num_runs,
        build_M, build_ef_construction, build_only, pruning_alpha,
        ablation_force_navigate, ablation_disable_twohop, ablation_disable_ivf);
    return 0;
}

// Example command line:
// ./bitset_exp/ivf_connect_hnsw -dataset yfcc -N 10000000 -dataset_path /path/to/base.data
// -query_path /path/to/query.data -ivf_index_path /path/to/ivf.index -data_tag_path /path/to/data.tag
// -query_tag_path /path/to/query.tag -gt_path /path/to/groundtruth.gt -graph_path /path/to/graph.hnsw
