/*
 * PostFilter-HNSW baseline (used in Fig. 8 of the paper).
 *
 * For every query: run an ordinary (unfiltered) HNSW search with beam width
 * ef, which returns the top-ef candidates; drop candidates that fail the
 * predicate bitset; return the 10 closest survivors. There is no re-search
 * and no ef enlargement when fewer than 10 candidates pass.
 *
 * The graph is loaded from -graph_path. Use the same graph file as CGIF
 * (built by `cgif_search -build_only`).
 *
 * Output per ef (parsed by reproduce/fig8/compare.py):
 *   Average Recall: <recall@10, mean over all (query, selectivity) rows>
 *   Average QPS:    <#rows / total search time>
 *   Search ef:      <ef>
 * followed by a per-selectivity breakdown.
 *
 * Copyright (c) 2024 Rutgers University - Database Group
 */

#include <sys/time.h>

#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "base_hnsw/hnswlib.h"
#include "data_utils.h"
#include "hnswalg.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

static void
exp(string dataset, int data_size, string dataset_path, string query_path, string graph_path, string gt_path) {
    if (graph_path.empty()) {
        std::cerr << "postfilter_hnsw: -graph_path is required" << endl;
        exit(1);
    }

    int query_num = 1000;
    int query_k = 10;
    string dataset_type = "";
    std::unique_ptr<DataWrapper> data_wrapper = createDataWrapperByType(
        dataset, dataset_type, query_num, query_k, data_size, dataset_path, query_path, "", "", gt_path);

    vector<int> search_ef_list = {100, 200, 300, 400, 500};

    std::unique_ptr<hnswlib::L2Space> space(new hnswlib::L2Space(data_wrapper->data_dim));
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index(
        new hnswlib::HierarchicalNSW<float>(space.get(), 2 * data_wrapper->data_size));
    cout << "Load index from: " << graph_path << endl;
    index->loadIndex(graph_path, space.get(), 2 * data_wrapper->data_size);
    cout << "max M0: " << index->maxM0_ << endl;

    for (int search_ef : search_ef_list) {
        index->setEf(search_ef);
        // per selectivity level: (sum recall, sum seconds), #rows, sum distance computations
        std::map<int, std::pair<double, double>> result_recorder;
        std::map<int, double> counter_recorder;
        std::map<int, double> comparison_recorder;

        for (size_t idx = 0; idx < data_wrapper->query_ids.size(); idx++) {
            int one_id = data_wrapper->query_ids.at(idx);
            if (one_id >= data_wrapper->query_num) continue;
            if (data_wrapper->groundtruth.at(idx).empty()) continue;

            auto bitset = BitsetView(data_wrapper->query_bitsets.at(idx).data(), data_wrapper->data_size);
            int query_range = queryRangeAdjustment(data_wrapper->data_size - bitset.count(), dataset);
            index->search_distance_computations = 0;
            index->metric_distance_computations = 0;

            vector<uint32_t> query_result;
            timeval t0, t1;
            gettimeofday(&t0, NULL);
            auto res = index->searchKnn(data_wrapper->querys.at(one_id).data(), search_ef);
            while (!res.empty()) {  // popped farthest-first
                if (bitset.is_member(res.top().second)) query_result.emplace_back(res.top().second);
                res.pop();
            }
            if (query_result.size() > static_cast<size_t>(query_k)) {  // keep the k closest
                query_result = vector<uint32_t>(query_result.end() - query_k, query_result.end());
            }
            gettimeofday(&t1, NULL);

            counter_recorder[query_range] += 1;
            result_recorder[query_range].first += countPrecision(data_wrapper->groundtruth.at(idx), query_result);
            result_recorder[query_range].second += CountTime(t0, t1);
            comparison_recorder[query_range] +=
                index->search_distance_computations + index->metric_distance_computations;
        }

        double recall_sum = 0, time_sum = 0, comps_sum = 0, rows = 0;
        for (auto& it : result_recorder) {
            recall_sum += it.second.first;
            time_sum += it.second.second;
            rows += counter_recorder.at(it.first);
            comps_sum += comparison_recorder.at(it.first);
        }
        cout << "Average Recall: " << std::setprecision(6) << recall_sum / rows << endl;
        cout << "Average QPS: " << rows / time_sum << endl;
        cout << "Average Distance Computations: " << comps_sum / rows << endl;
        cout << "Search ef: " << search_ef << endl << "========================" << endl;
        for (auto& it : result_recorder) {
            double n = counter_recorder.at(it.first);
            cout << std::fixed << std::setprecision(4) << "range: " << it.first
                 << "\t recall: " << it.second.first / n << "\t QPS: " << std::setprecision(0)
                 << n / it.second.second << "\t Comps: " << comparison_recorder.at(it.first) / n << endl;
        }
        cout.unsetf(std::ios::fixed);
        cout << "========================" << endl << endl;
    }
}

int
main(int argc, char** argv) {
    string dataset = "deep", dataset_path, query_path, graph_path, gt_path;
    int data_size = 1000000;
    for (int i = 0; i + 1 < argc; i++) {
        string arg = argv[i];
        if (arg == "-dataset") dataset = argv[i + 1];
        if (arg == "-N") data_size = atoi(argv[i + 1]);
        if (arg == "-dataset_path") dataset_path = argv[i + 1];
        if (arg == "-query_path") query_path = argv[i + 1];
        if (arg == "-graph_path") graph_path = argv[i + 1];
        if (arg == "-gt_path") gt_path = argv[i + 1];
    }
    exp(dataset, data_size, dataset_path, query_path, graph_path, gt_path);
    return 0;
}
