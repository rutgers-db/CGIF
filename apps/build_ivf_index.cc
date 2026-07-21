/*
 * Created on Fri Jul 05 2024
 *
 * Author: Chaoji Zuo
 * Email: chaoji.zuo@rutgers.edu
 *
 * Copyright (c) 2024 Rutgers University - Database Group
 */

#include <sys/time.h>

#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

// Faiss headers
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/index_io.h>  // For saving and loading indices
#include <faiss/utils/distances.h>

// Include your project's utilities
#include "data_utils.h"
#include "ivf.h"

using std::cin;
using std::cout;
using std::endl;
using std::ios;
using std::string;
using std::vector;

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
                    const std::map<int, size_t[3]>& comparison_recorder, const std::map<int, float>& counter_recorder) {
    for (auto it : result_recorder) {
        int query_range = it.first;
        float recall = it.second.first / counter_recorder.at(query_range);
        float qps = counter_recorder.at(query_range) / it.second.second;
        size_t comps = comparison_recorder.at(query_range)[0] / counter_recorder.at(query_range);

        cout << std::setiosflags(ios::fixed) << std::setprecision(4) << "range: " << query_range
             << "\t recall: " << recall << "\t QPS: " << std::setprecision(0) << qps << "\t Comps: " << comps << endl;
    }
}

// Faiss ID Selector implementation for bitset filtering
class BitsetIDSelector : public faiss::IDSelector {
 public:
    BitsetIDSelector(const BitsetView& bitset) : bitset_(bitset) {
    }

    bool
    is_member(faiss::idx_t id) const override {
        // Return true if the ID is NOT in the filter set
        return !bitset_.test(id);
    }

 private:
    const BitsetView& bitset_;
};

void
exp_ivf_flat(string dataset, int data_size, string dataset_path, string query_path, string data_tag_path = "",
             string query_tag_path = "", string gt_path = "", string save_index_path = "", string load_index_path = "",
             size_t num_clusters = 1024, bool bm25_mode = false, string detail_path = "") {
    cout << "==============================================" << endl;
    cout << "Faiss IVFFLAT Baseline for Filter Search" << endl;
    cout << "==============================================" << endl;

    int query_num = 1000;
    if (dataset == "local") {
        query_num = 10;
    }
    if (dataset == "yfcc-arbitrary") {     // C5: 61626 single-filter queries
        query_num = 100000;
    }
    int query_k = 10;

    // Determine dataset type based on dataset name
    string dataset_type = "";
    if (bm25_mode) {
        dataset_type = "bm25";
    }
    bool detail_mode = false;
    if (detail_path != "") {
        detail_mode = true;
    }

    // Create appropriate data wrapper
    std::unique_ptr<DataWrapper> data_wrapper;

    data_wrapper = createDataWrapperByType(dataset, dataset_type, query_num, query_k, data_size, dataset_path,
                                           query_path, data_tag_path, query_tag_path, gt_path);

    cout << "Data dimension: " << data_wrapper->data_dim << endl;
    cout << "Data size: " << data_wrapper->data_size << endl;
    cout << "Query size: " << data_wrapper->query_ids.size() << endl;

    timeval tt1, tt2;
    // Create or load the IVF index
    faiss::IndexIVFFlat* index_ptr = nullptr;
    if (load_index_path.empty()) {
        cout << "Creating new IVF index with " << num_clusters << " clusters" << endl;
        assert(!save_index_path.empty());

        // Generate the nodes_row format needed for Faiss
        std::unique_ptr<float[]> nodes_row;
        if (data_wrapper->nodes_row.empty()) {
            nodes_row = std::make_unique<float[]>(data_wrapper->data_size * data_wrapper->data_dim);

            for (size_t i = 0; i < data_wrapper->data_size; i++) {
                for (int j = 0; j < data_wrapper->data_dim; j++) {
                    nodes_row[i * data_wrapper->data_dim + j] = data_wrapper->nodes[i][j];
                }
            }
        }
        gettimeofday(&tt1, NULL);
        faiss::IndexFlatL2* quantizer = new faiss::IndexFlatL2(data_wrapper->data_dim);
        index_ptr = new faiss::IndexIVFFlat(quantizer, data_wrapper->data_dim, num_clusters, faiss::METRIC_L2);
        index_ptr->train(data_wrapper->data_size, nodes_row.get());
        index_ptr->add(data_wrapper->data_size, nodes_row.get());
        gettimeofday(&tt2, NULL);
        logTime(tt1, tt2, "IVFFlat index build time: ");

        if (!save_index_path.empty()) {
            cout << "Saving index to: " << save_index_path << endl;
            faiss::write_index(index_ptr, save_index_path.c_str());
        }
        nodes_row.reset();
        // Shortcut: if the user only wanted to build the index (e.g. for the
        // C2 nlist sensitivity sweep), don't run the full search experiment.
        const char* build_only_env = std::getenv("FAISS_IVF_BUILD_ONLY");
        if (build_only_env && std::string(build_only_env) == "1") {
            cout << "FAISS_IVF_BUILD_ONLY=1 set; skipping search workload." << endl;
            delete index_ptr;
            return;
        }
        // assert(false);
    } else {
        cout << "Loading index from: " << load_index_path << endl;
        index_ptr = dynamic_cast<faiss::IndexIVFFlat*>(faiss::read_index(load_index_path.c_str()));
        if (!index_ptr) {
            cout << "Failed to load index or not an IndexIVFFlat" << endl;
            return;
        }
    }

    faiss::IndexIVFFlat& index = *index_ptr;
    cout << "Index has " << index.ntotal << " vectors" << endl;
    cout << "Index has " << index.nlist << " clusters" << endl;

    // Metrics for different range buckets and nprobe values
    vector<int> nprobe_list = {16, 32, 64, 128};

    // data_wrapper->nodes.clear();

    // For each nprobe value, run search on all queries
    for (auto nprobe : nprobe_list) {
        index.nprobe = nprobe;
        faiss::indexIVF_stats.reset();
        double quantization_time = 0;
        double search_time = 0;

        std::map<int, std::pair<float, float>> result_recorder;  // range -> (recall, time)
        std::map<int, size_t[3]> comparison_recorder;
        std::map<int, float> counter_recorder;  // range -> count of queries

        // Process each query
        for (size_t idx = 0; idx < data_wrapper->query_ids.size(); idx++) {
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

            query_range = queryRangeAdjustment(query_range, dataset);
            // Count queries per range bucket
            counter_recorder[query_range] += 1.0f;

            // Create ID selector based on the bitset
            BitsetIDSelector selector(bitset);

            // Set up search parameters
            faiss::SearchParametersIVF params;
            params.sel = &selector;
            params.nprobe = nprobe;
            params.max_codes = 0;
            faiss::indexIVF_stats.reset();

            // Run the search with filtering
            std::vector<float> query_vector = data_wrapper->querys.at(one_id);
            const int query_k = data_wrapper->query_k;
            std::vector<faiss::idx_t> I(query_k);
            std::vector<float> D(query_k);

            timeval tt3, tt4;
            gettimeofday(&tt3, NULL);

            index.search(1, query_vector.data(), query_k, D.data(), I.data(), &params);

            std::vector<uint32_t> query_result;
            query_result.reserve(query_k);

            // Process search results
            for (int i = 0; i < query_k; i++) {
                if (I[i] >= 0) {
                    query_result.push_back(I[i]);
                } else {
                    query_result.push_back(0);  // Add dummy value for invalid indices
                }
            }

            // Ensure we have exactly query_k results
            query_result.resize(query_k, 0);

            gettimeofday(&tt4, NULL);
            double query_time = CountTime(tt3, tt4);

            // Evaluate precision
            float precision = countPrecision(data_wrapper->groundtruth.at(idx), query_result);

            result_recorder[query_range].first += precision;
            result_recorder[query_range].second += query_time;

            // Estimate distance computations
            int estimated_comps = faiss::indexIVF_stats.ndis * (query_range / (float)data_wrapper->data_size);
            comparison_recorder[query_range][0] += faiss::indexIVF_stats.ndis;
            comparison_recorder[query_range][1] += 0;
            comparison_recorder[query_range][2] += 0;

            quantization_time += faiss::indexIVF_stats.quantization_time;
            search_time += faiss::indexIVF_stats.search_time;
            if (detail_mode) {
                SaveToCSVRow(detail_path, "ivf_flat", idx, raw_query_range, index.nlist, nprobe, query_time, precision,
                             faiss::indexIVF_stats.ndis);
            }
        }

        // Calculate and output aggregate statistics
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
            avg_distance_computation += it.second[0];
        }

        if (result_recorder.size() > 0) {
            recall_avg /= valid_query_num;
            query_time_avg = valid_query_num / query_time_avg;
            avg_distance_computation /= valid_query_num;
            cout << "Average Recall: " << std::setprecision(4) << recall_avg << endl;
            cout << "Average QPS: " << query_time_avg << endl;
            cout << "Average Distance Computations: " << avg_distance_computation << endl;
            cout << "Average Quantization Time: " << quantization_time / 1000 << "s" << endl;
            cout << "Average Search Time: " << search_time / 1000 << "s" << endl;

            // Log results
            cout << "Results for nprobe = " << nprobe << ":" << endl;
            cout << "=========================" << endl;
            log_result_recorder(result_recorder, comparison_recorder, counter_recorder);
            cout << "=========================" << endl;

            // Output range distribution for the first nprobe value
            if (nprobe == nprobe_list.front() && dataset_type != "deep") {
                for (auto it : counter_recorder) {
                    cout << "range: " << it.first << "  " << it.second << endl;
                }
                cout << endl;
            }
            cout << endl;
        } else {
            cout << "No queries were processed for this nprobe value." << endl;
        }
    }

    // Clean up
    delete index_ptr;
}

int
main(int argc, char** argv) {
    string dataset = "biggraph";
    int data_size = 100000;
    string dataset_path = "";
    string query_path = "";
    string data_tag_path = "", query_tag_path = "", gt_path = "";
    string save_index_path = "";
    string load_index_path = "";
    size_t num_clusters = 1024;
    bool bm25_mode = false;
    string detail_path = "";

    for (int i = 0; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-dataset")
            dataset = string(argv[i + 1]);
        if (arg == "-N")
            data_size = atoi(argv[i + 1]);
        if (arg == "-dataset_path")
            dataset_path = string(argv[i + 1]);
        if (arg == "-query_path")
            query_path = string(argv[i + 1]);
        if (arg == "-data_tag_path")
            data_tag_path = string(argv[i + 1]);
        if (arg == "-query_tag_path")
            query_tag_path = string(argv[i + 1]);
        if (arg == "-gt_path")
            gt_path = string(argv[i + 1]);
        if (arg == "-save_index")
            save_index_path = string(argv[i + 1]);
        if (arg == "-ivf_index_path")
            load_index_path = string(argv[i + 1]);
        if (arg == "-num_clusters")
            num_clusters = atoi(argv[i + 1]);
        if (arg == "-bm25_mode")
            bm25_mode = true;
        if (arg == "-detail_path")
            detail_path = string(argv[i + 1]);
    }

    exp_ivf_flat(dataset, data_size, dataset_path, query_path, data_tag_path, query_tag_path, gt_path, save_index_path,
                 load_index_path, num_clusters, bm25_mode, detail_path);
    return 0;
}

// Example command line:
// ./bitset_exp/ivf_flat_baseline -dataset yfcc -N 10000000 -dataset_path /path/to/base.data
// -query_path /path/to/query.data -data_tag_path /path/to/data.tag -query_tag_path /path/to/query.tag
// -gt_path /path/to/groundtruth.gt -load_index /path/to/index.ivf -num_clusters 1024