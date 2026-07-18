/**
 * @file data_vecs.h
 * @author Chaoji Zuo (chaoji.zuo@rutgers.edu)
 * @brief Control the raw vector and querys
 * @date 2023-06-19
 *
 * @copyright Copyright (c) 2023
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "bitset.h"
#include "reader.h"
#include "tag_io.h"
#include "utils.h"

using std::pair;
using std::string;
using std::vector;

template <typename dist_t>
class BaseDataWrapper {
 public:
    BaseDataWrapper() {};
    BaseDataWrapper(int num, int k_, string dataset_name, int data_size_)
        : dataset(dataset_name), data_size(data_size_), query_num(num), query_k(k_) {};

    virtual ~BaseDataWrapper() {
        nodes.clear();
        nodes_row.clear();
        querys.clear();
    };

    // Common interface methods that can be overridden by derived classes
    virtual void
    readData(const string& dataset_path, const string& query_path = "") {
        // Default implementation can be empty or throw an error
        throw std::runtime_error("readData not implemented in base class");
    }

    virtual void
    LoadGroundtruth(const string& gt_path) {
        // Default implementation can be empty or throw an error
        throw std::runtime_error("LoadGroundtruth not implemented in base class");
    }

    // virtual void
    // generateBitsets();

    vector<vector<dist_t>> nodes;
    vector<dist_t> nodes_row;
    vector<vector<dist_t>> querys;  // raw querys;

    const string dataset;
    string version;
    int data_size;
    int query_num;

    const int query_k;
    size_t data_dim;

    // for each query vector, query with different query bitsets.
    vector<int> query_ids;
    vector<vector<uint8_t>> query_bitsets;

    vector<vector<uint32_t>> groundtruth;
};

// Standard DataWrapper with common functionality
class DataWrapper : public BaseDataWrapper<float> {
 public:
    DataWrapper(int num, int k_, string dataset_name, int data_size_)
        : BaseDataWrapper(num, k_, dataset_name, data_size_) {};

    virtual ~DataWrapper() = default;

    // Basic data operations
    void
    readData(const string& dataset_path, const string& query_path = "") override;
    void
    LoadGroundtruth(const string& gt_path) override;
    // void
    // generateBitsets() override;

    void
    generateGroundtruth(bool is_save = false, const string path = "");
    void
    nodes2row();
    void
    generateBitsetQueriesAndGroundtruth(bool is_save_to_file, const string save_path = "");

    // Factory method to create appropriate data wrapper
    static std::unique_ptr<DataWrapper>
    createDataWrapper(const std::string& dataset_type, int query_num, int query_k, const std::string& dataset_name,
                      int data_size);
};

// Tag distribution dataset wrapper
class TagDataWrapper : public DataWrapper {
 public:
    TagDataWrapper(int num, int k_, string dataset_name, int data_size_)
        : DataWrapper(num, k_, dataset_name, data_size_) {
    }

    void
    readData(const string& dataset_path, const string& query_path = "") override;

    void
    LoadGroundtruth(const string& gt_path) override;
    // void
    // generateBitsets() override;

    void
    loadWithTags(const string& dataset_path, const string& query_path, const string& data_tag_path,
                 const string& query_tag_path);

 private:
    // Additional data specific to tag distribution
    std::vector<std::vector<int32_t>> data_tags_;
    std::vector<std::vector<int32_t>> query_tags_;
};

// SPMAT dataset wrapper (transpose of tag) YFCC
class SPMATDataWrapper : public DataWrapper {
 public:
    SPMATDataWrapper(int num, int k_, string dataset_name, int data_size_)
        : DataWrapper(num, k_, dataset_name, data_size_) {
    }

    ~SPMATDataWrapper() {
        // if (raw_data_ != nullptr) {
        //     delete[] raw_data_;
        // }
    }

    void
    readData(const string& dataset_path, const string& query_path = "") override;
    void
    LoadGroundtruth(const string& gt_path) override;
    // void
    // generateBitsets() override;

    void
    loadWithTags(const string& dataset_path, const string& query_path, const string& data_tag_path,
                 const string& query_tag_path, const string& gt_path);

    void
    loadWithTagsOrCondition(const string& dataset_path, const string& query_path, const string& data_tag_path,
                            const string& query_tag_path, const string& gt_path, const string& new_gt_path = "");

    void
    loadWithTagsAndCondition(const string& dataset_path, const string& query_path, const string& data_tag_path,
                             const string& query_tag_path, const string& gt_path);

    // arxiv
    void
    loadArxivTags(const string& dataset_path, const string& query_path, const string& data_tag_path,
                  const string& query_tag_path, const string& gt_path);

    void
    generateGroundtruth(bool is_save = false, const string path = "");

 private:
    // Data specific to YFCC10M format
    // float* raw_data_ = nullptr;
};

class TextDataWrapper : public DataWrapper {
 public:
    TextDataWrapper(int num, int k_, string dataset_name, int data_size_)
        : DataWrapper(num, k_, dataset_name, data_size_) {
        data_words_ = std::make_shared<std::vector<std::vector<std::string>>>();
        query_words_ = std::make_shared<std::vector<std::vector<std::string>>>();
    }

    ~TextDataWrapper() {
        data_words_.reset();
        query_words_.reset();
    }

    // Add getter methods for data_words_ and query_words_
    std::shared_ptr<std::vector<std::vector<std::string>>>
    getDataWords() const {
        return data_words_;
    }

    std::shared_ptr<std::vector<std::vector<std::string>>>
    getQueryWords() const {
        return query_words_;
    }

    void
    readData(const string& dataset_path, const string& query_path = "") override {
        // Use ReadDataWrapper for base data
        string dataset_path_copy = dataset_path;  // Create non-const copies for ReadDataWrapper
        ReadDataWrapper(dataset, dataset_path_copy, this->nodes, data_size);

        if (!query_path.empty()) {
            string query_path_copy = query_path;
            ReadDataWrapper(dataset, query_path_copy, this->querys, query_num);
        }

        cout << "Load vecs from: " << dataset_path << endl;
        cout << "# of vecs: " << nodes.size() << endl;

        this->data_dim = this->nodes.front().size();
    }

    void
    LoadGroundtruth(const string& gt_path) override {
        // Similar implementation to DataWrapper::LoadGroundtruth
        cout << "Loading Groundtruth from " << gt_path << "..." << endl;
        this->groundtruth.clear();
        this->query_ids.clear();

        int counter = 0;
        ifstream infile;
        infile.open(gt_path, ios::in);
        assert(infile.is_open());
        string bline;
        string delim = ",";
        string space_delim = " ";
        while (getline(infile, bline, '\n')) {
            counter++;
            vector<uint32_t> one_gt;
            int one_id;
            vector<string> ret;
            Split(bline, delim, &ret);
            one_id = std::stoi(ret[0]);
            vector<string> str_gt;
            Split(ret[4], space_delim, &str_gt);
            str_gt.pop_back();
            for (auto ele : str_gt) {
                one_gt.emplace_back((uint32_t)std::stoi(ele));
            }
            this->groundtruth.emplace_back(one_gt);
            this->query_ids.emplace_back(one_id);
        }

        infile.close();

        string path_bitsets = gt_path;
        string str2 = ".gt";
        path_bitsets.replace(path_bitsets.find(str2), str2.length(), ".bitsets");
        LoadVecUintFromFile(this->query_bitsets, path_bitsets);

        std::cout<<"query_ids size: "<<this->query_ids.size()<<std::endl;

        assert(this->query_ids.size() == this->groundtruth.size());
        assert(this->query_bitsets.size() == this->query_ids.size());

        cout << "Done!" << endl;
    }

    void
    loadWithWords(const string& dataset_path, const string& query_path, const string& data_words_path,
                  const string& query_words_path, const string& gt_path) {
        string dataset_path_copy = dataset_path;  // Create non-const copies for ReadDataWrapper
        ReadDataWrapper(dataset, dataset_path_copy, this->nodes, data_size);

        if (!query_path.empty()) {
            string query_path_copy = query_path;
            ReadDataWrapper(dataset, query_path_copy, this->querys, query_num);
        }

        this->data_dim = this->nodes.front().size();

        cout << "# of vecs: " << nodes.size() << endl;
        cout << "# of queries: " << querys.size() << endl;
        cout << "data_dim: " << data_dim << endl;

        if (dataset == "laion") {
            // Load data words
            ReadCSVWords(data_words_path, this->data_words_, 3, 2);

            // Load query words
            ReadCSVWords(query_words_path, this->query_words_, 3, 0);
        } else if (dataset == "hotpotqa") {
            // Load data words
            ReadCSVWords(data_words_path, this->data_words_, 2, 1);

            // Load query words
            ReadCSVWords(query_words_path, this->query_words_, 2, 1);

        } else {
            cout << "No supporting read words for this dataset: " << dataset << endl;
        }
    }

    void
    loadHotpotqaGroundtruth(const string& gt_path, const string& bitset_path = "") {
        // Similar implementation to DataWrapper::LoadGroundtruth
        cout << "Loading Groundtruth from " << gt_path << "..." << endl;
        this->groundtruth.clear();
        this->query_ids.clear();

        int counter = 0;
        ifstream infile;
        infile.open(gt_path, ios::in);
        assert(infile.is_open());

        string bline;
        string delim = ",";
        string space_delim = " ";
        // Skip header line
        getline(infile, bline);
        while (getline(infile, bline, '\n')) {
            counter++;
            vector<uint32_t> one_gt;
            int one_id;
            vector<string> ret;
            Split(bline, delim, &ret);
            one_id = std::stoi(ret[0]);
            vector<string> str_gt;
            Split(ret[1], space_delim, &str_gt);
            for (auto ele : str_gt) {
                one_gt.emplace_back((uint32_t)std::stoi(ele));
            }
            this->groundtruth.emplace_back(one_gt);
            this->query_ids.emplace_back(one_id);
        }

        infile.close();

        string bitset_path_copy = bitset_path;
        if (bitset_path.empty()) {
            bitset_path_copy = gt_path;
            string str2 = ".gt";
            bitset_path_copy.replace(bitset_path_copy.find(str2), str2.length(), ".bitsets");
        }
        cout << "loading query_bitsets from: " << bitset_path_copy << endl;
        LoadVecUintFromFile(this->query_bitsets, bitset_path_copy);

        assert(this->query_ids.size() == this->groundtruth.size());
        assert(this->query_bitsets.size() == this->query_ids.size());

        cout << "Done!" << endl;
    }

    void
    generateTextMatchBitsets(const string& gt_path) {
        // Implement generateTextMatchBitsets for TextDataWrapper
        cout << "Generating text match bitsets for TextDataWrapper..." << endl;
        timeval t1, t2;
        string path_bitsets = gt_path;
        path_bitsets.replace(path_bitsets.find(".gt"), path_bitsets.length(), ".bitsets");
        double filter_time = 0;
        gettimeofday(&t1, NULL);
        std::unordered_map<std::string, std::vector<int>> word_to_docs;
        for (int i = 0; i < data_size; i++) {
            for (const auto& word : data_words_->at(i)) {
                word_to_docs[word].push_back(i);
            }
        }
        gettimeofday(&t2, NULL);
        logTime(t1, t2, "word_to_docs time");

        query_bitsets.resize(query_num);
        vector<int> query_range_list(query_num);
        gettimeofday(&t1, NULL);
#pragma omp parallel for
        for (int i = 0; i < query_num; i++) {
            auto query_words = query_words_->at(i);
            std::vector<uint8_t> bitset((data_size + 8 - 1) / 8, 0xFF);

            std::unordered_set<int> matching_docs;
            for (const auto& word : query_words) {
                auto it = word_to_docs.find(word);
                if (it != word_to_docs.end()) {
                    matching_docs.insert(it->second.begin(), it->second.end());
                }
            }
            for (int doc_id : matching_docs) {
                bitset[doc_id >> 3] &= ~(0x1 << (doc_id & 0x7));
            }
            BitsetView query_bitset(bitset.data(), data_size);
            auto query_range = data_size - query_bitset.count();
            query_bitsets[i].swap(bitset);
            query_range_list[i] = query_range;
        }
        gettimeofday(&t2, NULL);
        logTime(t1, t2, "generateTextMatchBitsets time");
        long int all_range = 0;
        for (int i = 0; i < query_num; i++) {
            all_range += query_range_list[i];
        }
        cout << "Average query range: " << all_range / query_num << endl;
        cout << "Save matching bitsets to path: " << path_bitsets << endl;
        SaveVecUintToFile(this->query_bitsets, path_bitsets);
    }

    void
    generateBM25Bitsets(const string& bitset_path, float threshold) {
        // Implement generateBM25Bitsets for TextDataWrapper
        cout << "Generating BM25 bitsets for TextDataWrapper..." << endl;
        cout << "threshold: " << threshold << endl;
        timeval t1, t2;
        BM25 bm25_index;
        // Build BM25 index
        gettimeofday(&t1, NULL);
        bm25_index.buildIndex(*this->data_words_);
        gettimeofday(&t2, NULL);
        logTime(t1, t2, "Build BM25 index time");
        cout << "data_size: " << data_size << endl;

        vector<int> query_range_list(query_num);
        query_bitsets.resize(query_num);
        gettimeofday(&t1, NULL);
        // Generate BM25 bitsets
#pragma omp parallel for
        for (int i = 0; i < query_num; i++) {
            auto query_words = query_words_->at(i);
            std::vector<uint8_t> bitset((data_size + 8 - 1) / 8, 0xFF);

            vector<size_t> filtered_docs = bm25_index.filterByThreshold(query_words, threshold);
            for (size_t doc_id : filtered_docs) {
                bitset[doc_id >> 3] &= ~(0x1 << (doc_id & 0x7));
            }
            BitsetView query_bitset(bitset.data(), data_size);
            size_t query_range = data_size - query_bitset.count();
            query_bitsets[i].swap(bitset);
            query_range_list[i] = query_range;
        }
        gettimeofday(&t2, NULL);
        logTime(t1, t2, "Generate BM25 Bitsets time");
        long int all_range = 0;
        for (int i = 0; i < query_num; i++) {
            all_range += query_range_list[i];
        }
        cout << "Average query range: " << all_range / query_num << endl;
        cout << "Save BM25 bitsets to path: " << bitset_path << endl;
        SaveVecUintToFile(this->query_bitsets, bitset_path);
    }

    void
    generateGroundtruth(bool is_save = false, const string path = "") {
        // Implement generateGroundtruth for TextDataWrapper
        cout << "Generating groundtruth for TextDataWrapper..." << endl;
        timeval t1, t2;
        double accu_time = 0;
        string path_bitsets;
        if (is_save) {
            path_bitsets = path;
            string str2 = ".gt";
            path_bitsets.replace(path_bitsets.find(str2), str2.length(), ".bitsets");
        }
        double filter_time = 0;
        long int all_range = 0;
        gettimeofday(&t1, NULL);
        // Create inverted index: word -> set of document IDs that contain the word
        std::unordered_map<std::string, std::vector<int>> word_to_docs;
        for (int j = 0; j < data_size; j++) {
            for (const auto& word : data_words_->at(j)) {
                word_to_docs[word].push_back(j);
            }
        }
        gettimeofday(&t2, NULL);
        logTime(t1, t2, "word_to_docs time");
        vector<double> query_time_list(query_num);
        vector<int> query_range_list(query_num);
        groundtruth.resize(query_num);
        query_ids.resize(query_num);
        query_bitsets.resize(query_num);

#pragma omp parallel for reduction(+ : filter_time, accu_time, all_range)
        for (int i = 0; i < query_num; i++) {
            if (i % 10000 == 0) {
                cout << "Processed " << i << " queries, filter time: " << filter_time << " , dist time: " << accu_time
                     << endl;
            }
            auto query_words = query_words_->at(i);
            auto query_vec = querys.at(i);
            std::vector<uint8_t> bitset((data_size + 8 - 1) / 8, 0xFF);

            double one_filter_time = 0;
            gettimeofday(&t1, NULL);
            // For each query word, get all matching documents
            std::unordered_set<int> matching_docs;
            for (const auto& word : query_words) {
                auto it = word_to_docs.find(word);
                if (it != word_to_docs.end()) {
                    matching_docs.insert(it->second.begin(), it->second.end());
                }
            }

            // Set bits for all matching documents
            for (int doc_id : matching_docs) {
                bitset[doc_id >> 3] &= ~(0x1 << (doc_id & 0x7));
            }
            gettimeofday(&t2, NULL);
            CountTime(t1, t2, one_filter_time);
            BitsetView query_bitset(bitset.data(), data_size);
            double greedy_time;
            gettimeofday(&t1, NULL);
            auto gt = greedyNearest(nodes, query_vec, query_k, query_bitset);
            gettimeofday(&t2, NULL);
            CountTime(t1, t2, greedy_time);
            // if (i == 0) {
            //     cout << "matching_docs: " << matching_docs.size() << endl;
            //     for (int doc_id : matching_docs) {
            //         cout << doc_id << ",";
            //         assert(query_bitset.is_member(doc_id));
            //     }
            //     cout << endl;
            // }
            auto query_range = data_size - query_bitset.count();
            groundtruth[i] = gt;
            query_ids[i] = i;
            query_bitsets[i].swap(bitset);
            query_range_list[i] = query_range;
            query_time_list[i] = greedy_time;
            accu_time += greedy_time;
            all_range += query_range;
            filter_time += one_filter_time;
        }

        cout << " Done!" << endl << "Total Groundtruth Time: " << accu_time << endl;
        cout << "filter time: " << filter_time << endl;
        cout << "Average query range: " << all_range / query_num << endl;
        // assert(false);
        if (is_save) {
            for (int i = 0; i < query_num; i++) {
                SaveToCSVRow(path, i, query_range_list[i], query_k, query_time_list[i], groundtruth[i], querys[i]);
            }
            cout << "Save GroundTruth to path: " << path << " and " << path_bitsets << endl;
            SaveVecUintToFile(this->query_bitsets, path_bitsets);
        }
    }

 private:
    // Additional data specific to tag distribution
    std::shared_ptr<std::vector<std::vector<std::string>>> data_words_;
    std::shared_ptr<std::vector<std::vector<std::string>>> query_words_;
};

class BM25DataWrapper : public DataWrapper {
 public:
    BM25DataWrapper(int num, int k_, string dataset_name, int data_size_)
        : DataWrapper(num, k_, dataset_name, data_size_) {
        data_words_ = std::make_shared<std::vector<std::vector<std::string>>>();
        query_words_ = std::make_shared<std::vector<std::vector<std::string>>>();
    }

    ~BM25DataWrapper() {
        data_words_.reset();
        query_words_.reset();
    }

    // Add getter methods for data_words_ and query_words_
    std::shared_ptr<std::vector<std::vector<std::string>>>
    getDataWords() const {
        return data_words_;
    }

    std::shared_ptr<std::vector<std::vector<std::string>>>
    getQueryWords() const {
        return query_words_;
    }

    void
    readData(const string& dataset_path, const string& query_path = "") override {
        // Use ReadDataWrapper for base data
        string dataset_path_copy = dataset_path;  // Create non-const copies for ReadDataWrapper
        ReadDataWrapper(dataset, dataset_path_copy, this->nodes, data_size);

        if (!query_path.empty()) {
            string query_path_copy = query_path;
            ReadDataWrapper(dataset, query_path_copy, this->querys, query_num);
        }

        cout << "Load vecs from: " << dataset_path << endl;
        cout << "# of vecs: " << nodes.size() << endl;

        this->data_dim = this->nodes.front().size();
    }

    void
    LoadGroundtruth(const string& gt_path) override {
        // Similar implementation to DataWrapper::LoadGroundtruth
        cout << "Loading Groundtruth from " << gt_path << "..." << endl;
        this->groundtruth.clear();
        this->query_ids.clear();

        int counter = 0;
        ifstream infile;
        infile.open(gt_path, ios::in);
        assert(infile.is_open());
        string bline;
        string delim = ",";
        string space_delim = " ";
        while (getline(infile, bline, '\n')) {
            counter++;
            vector<uint32_t> one_gt;
            int one_id;
            vector<string> ret;
            Split(bline, delim, &ret);
            one_id = std::stoi(ret[0]);
            vector<string> str_gt;
            Split(ret[4], space_delim, &str_gt);
            str_gt.pop_back();
            for (auto ele : str_gt) {
                one_gt.emplace_back((uint32_t)std::stoi(ele));
            }
            this->groundtruth.emplace_back(one_gt);
            this->query_ids.emplace_back(one_id);
        }

        infile.close();

        string path_bitsets = gt_path;
        string str2 = ".gt";
        path_bitsets.replace(path_bitsets.find(str2), str2.length(), ".bitsets");
        cout << "loading query_bitsets from: " << path_bitsets << endl;
        LoadVecUintFromFile(this->query_bitsets, path_bitsets);

        assert(this->query_ids.size() == this->groundtruth.size());
        assert(this->query_bitsets.size() == this->query_ids.size());

        cout << "Done!" << endl;
    }

    void
    LoadBM25Groundtruth(const string& dataset_path, const string& query_path, const string& data_words_path,
                        const string& query_words_path, const string& gt_path) {
        // Load base data
        string dataset_path_copy = dataset_path;  // Create non-const copies for ReadDataWrapper
        ReadDataWrapper(dataset, dataset_path_copy, this->nodes, data_size);

        if (!query_path.empty()) {
            string query_path_copy = query_path;
            ReadDataWrapper(dataset, query_path_copy, this->querys, query_num);
        }

        this->data_dim = this->nodes.front().size();

        cout << "# of vecs: " << nodes.size() << endl;
        cout << "# of queries: " << querys.size() << endl;
        cout << "data_dim: " << data_dim << endl;

        // Load data words
        ReadCSVWords(data_words_path, this->data_words_, 3, 2);

        // Load query words
        ReadCSVWords(query_words_path, this->query_words_, 3, 0);

        // Build BM25 index
        cout << "Building BM25 index..." << endl;
        bm25_index_.buildIndex(*data_words_);
        cout << "BM25 index built successfully" << endl;

        // Load groundtruth
        LoadGroundtruth(gt_path);
    }

 private:
    // Additional data specific to BM25
    std::shared_ptr<std::vector<std::vector<std::string>>> data_words_;
    std::shared_ptr<std::vector<std::vector<std::string>>> query_words_;
    BM25 bm25_index_;
};

// Factory implementation
inline std::unique_ptr<DataWrapper>
DataWrapper::createDataWrapper(const std::string& dataset_type, int query_num, int query_k,
                               const std::string& dataset_name, int data_size) {
    if (dataset_type == "bitset" || dataset_type.empty()) {
        return std::make_unique<DataWrapper>(query_num, query_k, dataset_name, data_size);
    } else if (dataset_type == "tag") {
        return std::make_unique<TagDataWrapper>(query_num, query_k, dataset_name, data_size);
    } else if (dataset_type == "spmat") {
        return std::make_unique<SPMATDataWrapper>(query_num, query_k, dataset_name, data_size);
    } else if (dataset_type == "text") {
        return std::make_unique<TextDataWrapper>(query_num, query_k, dataset_name, data_size);
    } else if (dataset_type == "bm25") {
        return std::make_unique<BM25DataWrapper>(query_num, query_k, dataset_name, data_size);
    } else {
        throw std::runtime_error("Unknown dataset type: " + dataset_type);
    }
}

// Implementations of DataWrapper methods
void
DataWrapper::readData(const string& dataset_path, const string& query_path) {
    string dataset_path_copy = dataset_path;  // Create non-const copies for ReadDataWrapper
    ReadDataWrapper(dataset, dataset_path_copy, this->nodes, data_size);

    if (!query_path.empty()) {
        string query_path_copy = query_path;
        ReadDataWrapper(dataset, query_path_copy, this->querys, query_num);
    }

    cout << "Load vecs from: " << dataset_path << endl;
    cout << "# of vecs: " << nodes.size() << endl;

    this->data_dim = this->nodes.front().size();
}

// void
// DataWrapper::readData(const string& dataset_path) {
//     ReadDataWrapper(dataset, dataset_path, this->nodes, data_size);
//     cout << "Load vecs from: " << dataset_path << endl;
//     cout << "# of vecs: " << nodes.size() << endl;

//     this->data_dim = this->nodes.front().size();
// }

void
DataWrapper::nodes2row() {
    nodes_row.resize(data_size * data_dim);
    for (size_t i = 0; i < data_size * data_dim; i++) {
        nodes_row[i] = nodes.at(i / data_dim).at(i % data_dim);
    }
    vector<vector<float>>().swap(nodes);
}

void
DataWrapper::LoadGroundtruth(const string& gt_path) {
    // Existing LoadGroundtruth implementation
    cout << "Loading Groundtruth from " << gt_path << "..." << endl;
    this->groundtruth.clear();
    this->query_ids.clear();

    int counter = 0;
    ifstream infile;
    infile.open(gt_path, ios::in);
    assert(infile.is_open());
    string bline;
    string delim = ",";
    string space_delim = " ";
    while (getline(infile, bline, '\n')) {
        counter++;
        vector<uint32_t> one_gt;
        int one_id;
        vector<string> ret;
        Split(bline, delim, &ret);
        one_id = std::stoi(ret[0]);
        vector<string> str_gt;
        Split(ret[4], space_delim, &str_gt);
        str_gt.pop_back();
        for (auto ele : str_gt) {
            one_gt.emplace_back((uint32_t)std::stoi(ele));
        }
        this->groundtruth.emplace_back(one_gt);
        this->query_ids.emplace_back(one_id);
    }

    infile.close();

    string path_bitsets = gt_path;
    string str2 = ".gt";
    path_bitsets.replace(path_bitsets.find(str2), str2.length(), ".bitsets");
    LoadVecUintFromFile(this->query_bitsets, path_bitsets);

    assert(this->query_ids.size() == this->groundtruth.size());
    assert(this->query_bitsets.size() == this->query_ids.size());

    cout << "Done!" << endl;
}

/// @brief Save knng in binary format (uint32_t) with name "output.bin"
/// @param knng a (N * 100) shape 2-D vector
/// @param path target save path, the output knng should be named as
/// "output.bin" for evaluation
void
SaveKNNG(const std::vector<std::vector<uint32_t>>& knng, const std::string& path, const size_t K) {
    std::ofstream ofs(path, std::ios::out | std::ios::binary);
    const uint32_t N = knng.size();
    std::cout << "Saving KNN Groundtruth (" << knng.size() << " X " << K << ") to " << path << std::endl;
    assert(knng.front().size() == K);
    for (unsigned i = 0; i < knng.size(); ++i) {
        auto const& knn = knng[i];
        ofs.write(reinterpret_cast<char const*>(&knn[0]), K * sizeof(uint32_t));
    }
    ofs.close();
}

void
DataWrapper::generateGroundtruth(bool is_save_to_file, const string save_path) {
    timeval t1, t2;

    cout << "Generating Groundtruth..." << endl;

    // generate groundtruth

    groundtruth.resize(this->querys.size());
    gettimeofday(&t1, NULL);

#pragma omp parallel for
    for (int i = 0; i < this->querys.size(); i++) {
        auto gt = greedyNearest(this->nodes, this->querys.at(i), this->query_k);
        groundtruth[i].swap(gt);
    }
    gettimeofday(&t2, NULL);

    cout << "  Done!" << endl;
    logTime(t1, t2, "Groundtruth Time");

    if (is_save_to_file)
        SaveKNNG(groundtruth, save_path, query_k);
}

void
DataWrapper::generateBitsetQueriesAndGroundtruth(bool is_save_to_file, const string save_path) {
    std::default_random_engine e;
    timeval t1, t2, t3;
    double accu_time = 0.0;

    vector<int> query_range_list;
    float scale = 0.05;
    if (this->dataset == "local") {
        scale = 0.1;
    }
    int init_range = this->data_size * scale;
    while (init_range <= this->data_size) {
        query_range_list.emplace_back(init_range);
        init_range += this->data_size * scale;
    }
#ifdef LOG_DEBUG_MODE
    if (this->dataset == "local") {
        query_range_list.erase(query_range_list.begin(), query_range_list.begin() + 4 * query_range_list.size() / 9);
    }
#endif
    if (0.01 * this->data_size > 100)
        query_range_list.insert(query_range_list.begin(), 0.01 * this->data_size);

    if (0.001 * this->data_size > 100)
        query_range_list.insert(query_range_list.begin(), 0.001 * this->data_size);

    if (0.0001 * this->data_size > 100)
        query_range_list.insert(query_range_list.begin(), 0.0001 * this->data_size);
    if (this->data_size == 1000000) {
        query_range_list = {1000,   2000,   3000,   4000,   5000,   6000,   7000,   8000,   9000,   10000,
                            20000,  30000,  40000,  50000,  60000,  70000,  80000,  90000,  100000, 200000,
                            300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000};
    }
    if (this->data_size == 10000000) {
        query_range_list = {10000,   20000,   30000,   40000,   50000,   60000,   70000,   80000,   90000,   100000,
                            200000,  300000,  400000,  500000,  600000,  700000,  800000,  900000,  1000000, 2000000,
                            3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000};
    }
    cout << "Generating Groundtruth...\nRanges: ";
    print_set(query_range_list);
    // generate groundtruth

    for (auto range : query_range_list) {
        double one_range_total_greedy_time = 0;
        // Pre-allocate space for this range's results
        std::vector<std::vector<uint32_t>> range_groundtruth(this->querys.size());
        std::vector<int> range_query_ids(this->querys.size());
        std::vector<std::vector<uint8_t>> range_query_bitsets(this->querys.size());
        std::vector<double> range_greedy_time(this->querys.size());
        // Parallelize the query loop
        gettimeofday(&t3, NULL);
#pragma omp parallel for
        for (int i = 0; i < this->querys.size(); i++) {
            auto bitset_data = GenRandomBitset(this->data_size, this->data_size - range);
            BitsetView bitset(bitset_data.data(), this->data_size);

            double greedy_time;
            gettimeofday(&t1, NULL);
            auto gt = greedyNearest(this->nodes, this->querys.at(i), this->query_k, bitset);
            gettimeofday(&t2, NULL);
            CountTime(t1, t2, greedy_time);

            range_groundtruth[i] = gt;
            range_query_ids[i] = i;
            range_query_bitsets[i] = std::move(bitset_data);
            range_greedy_time[i] = greedy_time;
        }
        gettimeofday(&t1, NULL);
        CountTime(t3, t1, one_range_total_greedy_time);
        // Append results after parallel region
        groundtruth.insert(groundtruth.end(), range_groundtruth.begin(), range_groundtruth.end());
        query_ids.insert(query_ids.end(), range_query_ids.begin(), range_query_ids.end());
        query_bitsets.insert(query_bitsets.end(), range_query_bitsets.begin(), range_query_bitsets.end());

        cout << "Range: " << range << " \t greedy time:   " << one_range_total_greedy_time << endl;

        for (int i = 0; i < this->querys.size(); i++) {
            SaveToCSVRow(save_path, i, range, this->query_k, range_greedy_time[i], range_groundtruth[i],
                         this->querys.at(i));
        }
    }

    cout << " Done!" << endl << "Total Groundtruth Time: " << accu_time << endl;
    if (is_save_to_file) {
        cout << "Save GroundTruth to path: " << save_path << endl;
        string path_bitsets = save_path;
        string str2 = ".gt";
        path_bitsets.replace(path_bitsets.find(str2), str2.length(), ".bitsets");
        SaveVecUintToFile(this->query_bitsets, path_bitsets);
    }
}

// Implementations of TagDistributionDataWrapper methods
void
TagDataWrapper::readData(const string& dataset_path, const string& query_path) {
    string dataset_path_copy = dataset_path;
    ReadDataWrapper(dataset, dataset_path_copy, this->nodes, data_size);
    if (!query_path.empty()) {
        string query_path_copy = query_path;
        ReadDataWrapper(dataset, query_path_copy, this->querys, 10000);
        this->query_num = this->querys.size();
    }

    cout << "Load vecs from: " << dataset_path << endl;
    cout << "# of vecs: " << nodes.size() << endl;

    this->data_dim = this->nodes.front().size();
}

void
TagDataWrapper::LoadGroundtruth(const string& gt_path) {
    // Similar implementation to DataWrapper::LoadGroundtruth
    DataWrapper::LoadGroundtruth(gt_path);
}

// void
// TagDataWrapper::generateBitsets() {
//     // Initialize empty bitsets
//     query_bitsets.clear();
//     query_bitsets.reserve(query_num);

//     vector<uint8_t> empty_bitset((data_size + 8 - 1) / 8, 0);
//     for (int i = 0; i < query_num; i++) {
//         query_bitsets.emplace_back(empty_bitset);
//     }

//     // Generate bitsets based on tag matching logic would go here
//     // Specific to tag distribution dataset
// }

void
TagDataWrapper::loadWithTags(const string& dataset_path, const string& query_path, const string& data_tag_path,
                             const string& query_tag_path) {
    // Load base data
    readData(dataset_path);
    query_num = querys.size();

    // Load query tags
    int64_t query_rows, query_cols, query_nnz;
    int64_t* query_row_index = nullptr;
    int32_t* query_col_index = nullptr;
    float* query_col_value = nullptr;
    read_sparse_matrix(query_tag_path.c_str(), query_rows, query_cols, query_nnz, query_row_index, query_col_index,
                       query_col_value);

    // Load data tags
    int64_t base_rows, base_cols, base_nnz;
    int64_t* base_row_index = nullptr;
    int32_t* base_col_index = nullptr;
    float* base_col_value = nullptr;
    read_sparse_matrix(data_tag_path.c_str(), base_rows, base_cols, base_nnz, base_row_index, base_col_index,
                       base_col_value);

    // Prepare query data
    querys.clear();
    query_ids.clear();
    querys.resize(query_num);
    query_ids.resize(query_num);
    std::iota(query_ids.begin(), query_ids.end(), 0);

    // Generate bitsets based on tag matching
    query_bitsets.clear();
    query_bitsets.reserve(query_num);
    vector<uint8_t> empty_bitset((data_size + 8 - 1) / 8, 0xFF);
    for (int i = 0; i < query_num; i++) {
        query_bitsets.emplace_back(empty_bitset);
    }

// Match tags and build bitsets
#pragma omp parallel for
    for (unsigned long i = 0; i < data_size; i++) {
        std::vector<int32_t> base_filters(base_col_index + base_row_index[i], base_col_index + base_row_index[i + 1]);
        for (unsigned long j = 0; j < query_num; j++) {
            bool match = true;
            for (int64_t p = query_row_index[j]; p < query_row_index[j + 1]; p++) {
                if (std::find(base_filters.begin(), base_filters.end(), query_col_index[p]) == base_filters.end()) {
                    match = false;
                    break;
                }
            }
            if (!match)
                query_bitsets[j][i >> 3] |= (0x1 << (i & 0x7));
        }
    }
}

// Implementations of SPMATDataWrapper methods
void
SPMATDataWrapper::readData(const string& dataset_path, const string& query_path) {
    // Use ReadDataWrapper for base data
    string dataset_path_copy = dataset_path;
    ReadDataWrapper(dataset, dataset_path_copy, this->nodes, data_size);

    if (!query_path.empty()) {
        string query_path_copy = query_path;
        ReadDataWrapper(dataset, query_path_copy, this->querys, query_num);
    }

    cout << "Load vecs from: " << dataset_path << endl;
    cout << "# of vecs: " << nodes.size() << endl;

    this->data_dim = this->nodes.front().size();
}

void
SPMATDataWrapper::LoadGroundtruth(const string& gt_path) {
    uint32_t nq, gt_dim;
    uint32_t* gt_ids = nullptr;
    float* gt_dis = nullptr;
    read_gt(gt_path.c_str(), nq, gt_dim, gt_ids, gt_dis);

    groundtruth.clear();
    groundtruth.resize(nq);
    for (int i = 0; i < nq; i++) {
        std::vector<uint32_t> one_gt{gt_ids + i * gt_dim, gt_ids + (i + 1) * gt_dim};
        groundtruth[i].swap(one_gt);
    }

    delete[] gt_ids;
    delete[] gt_dis;
}

void
SPMATDataWrapper::loadWithTags(const string& dataset_path, const string& query_path, const string& data_tag_path,
                               const string& query_tag_path, const string& gt_path) {
    // Load base data
    cout << "Load vecs from: " << dataset_path << endl;
    readData(dataset_path, query_path);
    cout << "Amount of vecs: " << nodes.size() << endl;
    cout << "Amount of queries: " << querys.size() << endl;
    cout << "Amount of data_dim: " << data_dim << endl;

    cout << "Note! for yfcc, please load raw u8bin query and base, they will be transform to float internally" << endl;

    query_num = querys.size();

    query_ids.resize(query_num);
    std::iota(query_ids.begin(), query_ids.end(), 0);

    // Load tags - NOTE: using transpose of tag spmat for YFCC10M
    int64_t query_rows, query_cols, query_nnz;
    int64_t* query_row_index = nullptr;
    int32_t* query_col_index = nullptr;
    float* query_col_value = nullptr;
    read_sparse_matrix(query_tag_path.c_str(), query_rows, query_cols, query_nnz, query_row_index, query_col_index,
                       query_col_value);

    int64_t base_rows, base_cols, base_nnz;
    int64_t* base_row_index = nullptr;
    int32_t* base_col_index = nullptr;
    float* base_col_value = nullptr;
    read_sparse_matrix(data_tag_path.c_str(), base_rows, base_cols, base_nnz, base_row_index, base_col_index,
                       base_col_value);

    // Load ground truth
    LoadGroundtruth(gt_path);

    cout << "Generate bitsets..." << endl;

    // Generate bitsets - for YFCC10M we initialize with all 1s
    query_bitsets.clear();
    query_bitsets.reserve(query_num);
    vector<uint8_t> empty_bitset((data_size + 8 - 1) / 8, 255);  // set all to 1 at first
    for (int i = 0; i < query_num; i++) {
        query_bitsets.emplace_back(empty_bitset);
    }

// Special bitset generation for YFCC10M (using transposed tag files)
// Note: this is for matching "or" condition on multiple tags
#pragma omp parallel for
    for (unsigned long j = 0; j < query_num; j++) {
        for (int64_t p = query_row_index[j]; p < query_row_index[j + 1]; p++) {
            auto tag = query_col_index[p];
            std::vector<int32_t> satisfy_points(base_col_index + base_row_index[tag],
                                                base_col_index + base_row_index[tag + 1]);

            for (auto i : satisfy_points) {
                query_bitsets[j][(size_t)i >> 3] &= ~(0x1 << ((size_t)i & 0x7));  // set satisfied points to 0
            }
        }
    }

    cout << "Generate bitsets done!" << endl;
}

void
SPMATDataWrapper::loadArxivTags(const string& dataset_path, const string& query_path, const string& data_tag_path,
                                const string& query_tag_path, const string& gt_path) {
    // Load base data
    cout << "Load vecs from: " << dataset_path << endl;
    readData(dataset_path, query_path);
    cout << "Amount of vecs: " << nodes.size() << endl;
    cout << "Amount of queries: " << querys.size() << endl;
    cout << "Amount of data_dim: " << data_dim << endl;

    query_num = querys.size();

    query_ids.resize(query_num);
    std::iota(query_ids.begin(), query_ids.end(), 0);

    // Load tags - NOTE: using transpose of tag spmat for YFCC10M
    int64_t query_rows, query_cols, query_nnz;
    int64_t* query_row_index = nullptr;
    int32_t* query_col_index = nullptr;
    float* query_col_value = nullptr;
    read_sparse_matrix(query_tag_path.c_str(), query_rows, query_cols, query_nnz, query_row_index, query_col_index,
                       query_col_value);

    int64_t base_rows, base_cols, base_nnz;
    int64_t* base_row_index = nullptr;
    int32_t* base_col_index = nullptr;
    float* base_col_value = nullptr;
    read_sparse_matrix(data_tag_path.c_str(), base_rows, base_cols, base_nnz, base_row_index, base_col_index,
                       base_col_value);

    cout << "Generate bitsets..." << endl;

    // Generate bitsets - for YFCC10M we initialize with all 1s
    query_bitsets.clear();
    query_bitsets.reserve(query_num);
    vector<uint8_t> empty_bitset((data_size + 8 - 1) / 8, 255);  // set all to 1 at first
    for (int i = 0; i < query_num; i++) {
        query_bitsets.emplace_back(empty_bitset);
    }

// Special bitset generation for YFCC10M (using transposed tag files)
// Note: this is for matching "or" condition on multiple tags
#pragma omp parallel for
    for (unsigned long j = 0; j < query_num; j++) {
        for (int64_t p = query_row_index[j]; p < query_row_index[j + 1]; p++) {
            auto tag = query_col_index[p];
            std::vector<int32_t> satisfy_points(base_col_index + base_row_index[tag],
                                                base_col_index + base_row_index[tag + 1]);

            for (auto i : satisfy_points) {
                query_bitsets[j][(size_t)i >> 3] &= ~(0x1 << ((size_t)i & 0x7));  // set satisfied points to 0
            }
        }
    }

    cout << "Generate bitsets done!" << endl;

    // load groundtruth
    if (!gt_path.empty()) {
        int counter = 0;
        ifstream infile;
        infile.open(gt_path, ios::in);
        assert(infile.is_open());

        string bline;
        string delim = ",";
        string space_delim = " ";
        // Skip header line
        // getline(infile, bline);
        while (getline(infile, bline, '\n')) {
            counter++;
            vector<uint32_t> one_gt;
            // int one_id; // don't need this
            vector<string> ret;
            Split(bline, delim, &ret);
            // one_id = std::stoi(ret[0]);
            vector<string> str_gt;
            Split(ret[4], space_delim, &str_gt);
            str_gt.pop_back();
            for (auto ele : str_gt) {
                one_gt.emplace_back((uint32_t)std::stoi(ele));
            }
            this->groundtruth.emplace_back(one_gt);
            // this->query_ids.emplace_back(one_id);
        }
        cout << "Amount of groundtruth: " << groundtruth.size() << endl;

        infile.close();
    }
}

void
SPMATDataWrapper::generateGroundtruth(bool is_save, const string path) {
    cout << "Generate SMPAT groundtruth..." << endl;
    groundtruth.clear();
    groundtruth.resize(query_num);
    // this is for generating groundtruth
    if (is_save) {
        timeval t1, t2;
        vector<int> query_range_list(query_num, 0);
        vector<double> query_time_list(query_num, 0);
        // #pragma omp parallel for
        for (int i = 0; i < query_num; i++) {
            BitsetView query_bitset(query_bitsets[i].data(), data_size);
            double greedy_time;
            gettimeofday(&t1, NULL);
            auto gt = greedyNearest(nodes, querys[i], query_k, query_bitset);
            gettimeofday(&t2, NULL);
            CountTime(t1, t2, greedy_time);
            auto query_range = data_size - query_bitset.count();
            query_range_list[i] = query_range;
            query_time_list[i] = greedy_time;
            groundtruth[i].swap(gt);
        }
        int counter = 0;
        for (int i = 0; i < query_num; i++) {
            SaveToCSVRow(path, i, query_range_list[i], query_k, query_time_list[i], groundtruth[i], querys[i]);
            if (query_range_list[i] > 0) {
                counter++;
            }
        }
        cout << "Save GroundTruth to path: " << path << endl;
        cout << "Amount of queries: " << counter << endl;
    }
}

void
SPMATDataWrapper::loadWithTagsAndCondition(const string& dataset_path, const string& query_path,
                                           const string& data_tag_path, const string& query_tag_path,
                                           const string& gt_path) {
    // Load base data
    cout << "Load vecs from: " << dataset_path << endl;
    readData(dataset_path, query_path);
    cout << "Amount of vecs: " << nodes.size() << endl;
    cout << "Amount of queries: " << querys.size() << endl;
    cout << "Amount of data_dim: " << data_dim << endl;

    cout << "Note! for yfcc, please load raw u8bin query and base, they will be transform to float internally" << endl;

    query_num = querys.size();

    query_ids.resize(query_num);
    std::iota(query_ids.begin(), query_ids.end(), 0);

    // Load tags - NOTE: using transpose of tag spmat for YFCC10M
    int64_t query_rows, query_cols, query_nnz;
    int64_t* query_row_index = nullptr;
    int32_t* query_col_index = nullptr;
    float* query_col_value = nullptr;
    read_sparse_matrix(query_tag_path.c_str(), query_rows, query_cols, query_nnz, query_row_index, query_col_index,
                       query_col_value);

    int64_t base_rows, base_cols, base_nnz;
    int64_t* base_row_index = nullptr;
    int32_t* base_col_index = nullptr;
    float* base_col_value = nullptr;
    read_sparse_matrix(data_tag_path.c_str(), base_rows, base_cols, base_nnz, base_row_index, base_col_index,
                       base_col_value);

    // Load ground truth
    LoadGroundtruth(gt_path);

    cout << "Generate bitsets..." << endl;

    timeval t1, t2;
    gettimeofday(&t1, NULL);

    // Generate bitsets - for YFCC10M we initialize with all 1s
    // query_bitsets.clear();
    // query_bitsets.reserve(query_num);
    // vector<uint8_t> empty_bitset((data_size + 8 - 1) / 8, 255);  // set all to 1 at first
    // for (int i = 0; i < query_num; i++) {
    //     query_bitsets.emplace_back(empty_bitset);
    // }

    // 2) Initialize all‐1 bitsets for each query
    size_t Bsize = (data_size + 8 - 1) / 8;
    query_bitsets.clear();
    query_bitsets.resize(query_num, vector<uint8_t>(Bsize, 0xFF));

    // 1) Build inverted index (once) for data-tags:
    //    tag_postings[t] = sorted list of data‐IDs having tag ‘t’
    vector<vector<uint32_t>> tag_postings(base_rows);
    for (int64_t t = 0; t < base_rows; ++t) {
        auto start = base_col_index + base_row_index[t];
        auto end = base_col_index + base_row_index[t + 1];
        auto& plist = tag_postings[t];
        plist.reserve(end - start);
        for (auto ptr = start; ptr < end; ++ptr) {
            plist.push_back((uint32_t)*ptr);
        }
        // they’re already sorted if your CSR is in row‐major order
    }

    // 3) For each query, intersect its tag‐posting lists
#pragma omp parallel for schedule(dynamic)
    for (size_t j = 0; j < query_num; ++j) {
        // fetch which tags this query has
        auto qstart = query_row_index[j];
        auto qend = query_row_index[j + 1];
        size_t tcount = qend - qstart;
        if (tcount == 1) {
            // skip single tag query
            groundtruth[j].clear();
            // no tags ⇒ keep all bits = 1 (all accepted)
            continue;
        }

        // gather the tag IDs into a small vector
        vector<uint32_t> tags(query_col_index + qstart, query_col_index + qend);

        // sort tags by posting‐list length (smallest first)
        sort(tags.begin(), tags.end(),
             [&](uint32_t a, uint32_t b) { return tag_postings[a].size() < tag_postings[b].size(); });

        // start the intersection with the smallest list
        vector<uint32_t> common = tag_postings[tags[0]];

        // intersect with each subsequent tag’s list
        for (size_t ti = 1; ti < tags.size() && !common.empty(); ++ti) {
            auto& plist = tag_postings[tags[ti]];
            vector<uint32_t> tmp;
            tmp.reserve(std::min(common.size(), plist.size()));
            std::set_intersection(common.begin(), common.end(), plist.begin(), plist.end(), back_inserter(tmp));
            common.swap(tmp);
        }

        // 4) “Turn off” exactly those IDs in the bitset
        auto& bs = query_bitsets[j];
        for (auto id : common) {
            bs[id >> 3] &= static_cast<uint8_t>(~(0x1 << (id & 7)));
        }
    }

    gettimeofday(&t2, NULL);
    logTime(t1, t2, "Generate bitsets time");
    // cout << "Generate bitsets done!" << endl;
}

void
SPMATDataWrapper::loadWithTagsOrCondition(const string& dataset_path, const string& query_path,
                                          const string& data_tag_path, const string& query_tag_path,
                                          const string& gt_path, const string& new_gt_path) {
    // Load base data
    cout << "Load vecs from: " << dataset_path << endl;
    readData(dataset_path, query_path);
    cout << "Amount of vecs: " << nodes.size() << endl;
    cout << "Amount of queries: " << querys.size() << endl;
    cout << "Amount of data_dim: " << data_dim << endl;

    cout << "Note! for yfcc, please load raw u8bin query and base, they will be transform to float internally" << endl;

    query_num = querys.size();

    // query_ids.resize(query_num);
    // std::iota(query_ids.begin(), query_ids.end(), 0);

    // Load tags - NOTE: using transpose of tag spmat for YFCC10M
    int64_t query_rows, query_cols, query_nnz;
    int64_t* query_row_index = nullptr;
    int32_t* query_col_index = nullptr;
    float* query_col_value = nullptr;
    read_sparse_matrix(query_tag_path.c_str(), query_rows, query_cols, query_nnz, query_row_index, query_col_index,
                       query_col_value);

    int64_t base_rows, base_cols, base_nnz;
    int64_t* base_row_index = nullptr;
    int32_t* base_col_index = nullptr;
    float* base_col_value = nullptr;
    read_sparse_matrix(data_tag_path.c_str(), base_rows, base_cols, base_nnz, base_row_index, base_col_index,
                       base_col_value);

    // for no groundtruth
    groundtruth.clear();
    // groundtruth.resize(query_num);

    // load groundtruth
    if (new_gt_path.empty()) {
        int counter = 0;
        ifstream infile;
        infile.open(gt_path, ios::in);
        assert(infile.is_open());

        string bline;
        string delim = ",";
        string space_delim = " ";
        // Skip header line
        // getline(infile, bline);
        while (getline(infile, bline, '\n')) {
            counter++;
            vector<uint32_t> one_gt;
            // int one_id; // don't need this
            vector<string> ret;
            Split(bline, delim, &ret);
            // one_id = std::stoi(ret[0]);
            vector<string> str_gt;
            Split(ret[4], space_delim, &str_gt);
            str_gt.pop_back();
            for (auto ele : str_gt) {
                one_gt.emplace_back((uint32_t)std::stoi(ele));
            }
            this->groundtruth.emplace_back(one_gt);
            // this->query_ids.emplace_back(one_id);
        }

        infile.close();
    }

    cout << "Amount of groundtruth: " << groundtruth.size() << endl;

    cout << "Generate bitsets..." << endl;

    // Generate bitsets - for YFCC10M we initialize with all 1s
    query_bitsets.clear();
    query_bitsets.reserve(query_num);
    vector<uint8_t> empty_bitset((data_size + 8 - 1) / 8, 255);  // set all to 1 at first
    for (int i = 0; i < query_num; i++) {
        query_bitsets.emplace_back(empty_bitset);
        this->query_ids.emplace_back(i);
    }

    vector<bool> is_or_condition(query_num, false);

// for "or" condition on multiple tags
#pragma omp parallel for
    for (unsigned long j = 0; j < query_num; j++) {
        int target_count = query_row_index[j + 1] - query_row_index[j];
        if (target_count == 1) {
            // this is for keep "or" condition query only
            if (groundtruth.size() > 0) {
                groundtruth[j].clear();
            }
            query_bitsets[j].clear();
            continue;
        } else {
            is_or_condition[j] = true;
            for (int64_t p = query_row_index[j]; p < query_row_index[j + 1]; p++) {
                auto tag = query_col_index[p];
                std::vector<int32_t> satisfy_points(base_col_index + base_row_index[tag],
                                                    base_col_index + base_row_index[tag + 1]);

                for (auto i : satisfy_points) {
                    query_bitsets[j][(size_t)i >> 3] &= ~(0x1 << ((size_t)i & 0x7));  // set satisfied points to 0
                }
            }
        }
    }

    // assert(query_num == groundtruth.size());
    assert(query_num == query_ids.size());
    assert(query_num == query_bitsets.size());

    // this is for generating groundtruth
    if (!new_gt_path.empty()) {
        cout << "Generate groundtruth..." << endl;
        groundtruth.clear();
        groundtruth.resize(query_num);
        timeval t1, t2;
        vector<int> query_range_list(query_num, 0);
        vector<double> query_time_list(query_num, 0);
        // #pragma omp parallel for
        for (int i = 0; i < query_num; i++) {
            if (is_or_condition[i]) {
                BitsetView query_bitset(query_bitsets[i].data(), data_size);
                double greedy_time;
                gettimeofday(&t1, NULL);
                auto gt = greedyNearest(nodes, querys[i], query_k, query_bitset);
                gettimeofday(&t2, NULL);
                CountTime(t1, t2, greedy_time);
                auto query_range = data_size - query_bitset.count();
                query_range_list[i] = query_range;
                query_time_list[i] = greedy_time;
                groundtruth[i].swap(gt);
            } else {
                query_range_list[i] = 0;
                query_time_list[i] = 0;
            }
        }
        int counter = 0;
        for (int i = 0; i < query_num; i++) {
            SaveToCSVRow(new_gt_path, i, query_range_list[i], query_k, query_time_list[i], groundtruth[i], querys[i]);
            if (query_range_list[i] > 0) {
                counter++;
            }
        }
        cout << "Save GroundTruth to path: " << new_gt_path << endl;
        cout << "Amount of 'or' condition queries: " << counter << endl;
    }

    cout << "Generate bitsets done!" << endl;
}

class RangeDataWrapper : public DataWrapper {
 public:
    RangeDataWrapper(int num, int k_, string dataset_name, int data_size_)
        : DataWrapper(num, k_, dataset_name, data_size_) {};

    void
    LoadGroundtruth(const string& gt_path) override {
        cout << "Load range-filter groundtruth from: " << gt_path << endl;
        this->groundtruth.clear();
        this->query_ids.clear();
        this->query_bitsets.clear();
        vector<std::pair<int, int>> query_ranges;

        ifstream infile;
        string bline;
        string delim = ",";
        string space_delim = " ";

        int numCols = 0;
        infile.open(gt_path, ios::in);
        assert(infile.is_open());

        int counter = 0;
        while (getline(infile, bline, '\n')) {
            counter++;
            vector<uint32_t> one_gt;
            std::pair<int, int> one_range;
            int one_id;
            vector<string> ret;
            Split(bline, delim, &ret);
            one_id = std::stoi(ret[0]);
            one_range.first = std::stoi(ret[1]);
            one_range.second = std::stoi(ret[2]);
            vector<string> str_gt;
            Split(ret[7], space_delim, &str_gt);
            str_gt.pop_back();
            for (auto ele : str_gt) {
                one_gt.emplace_back((uint32_t)std::stoi(ele));
            }
            this->groundtruth.emplace_back(one_gt);
            this->query_ids.emplace_back(one_id);
            query_ranges.emplace_back(one_range);
        }

        size_t Bsize = (data_size + 8 - 1) / 8;
        this->query_bitsets.resize(this->query_ids.size(), vector<uint8_t>(Bsize, 0xFF));

        for (int i = 0; i < this->query_ids.size(); i++) {
            auto& bs = this->query_bitsets[i];
            for (int j = query_ranges[i].first; j <= query_ranges[i].second; j++) {
                // set the position to 0
                bs[j >> 3] &= ~(0x1 << (j & 0x7));
            }

            // BitsetView bs_view(bs.data(), data_size);
            // cout << "range:" << query_ranges[i].second - query_ranges[i].first << endl;
            // cout << "bitset:" << bs_view.count() << endl;
            // cout << "(" << query_ranges[i].first << ", " << query_ranges[i].second << ")" << endl;
            // cout << "groundtruth:" << endl;
            // print_set(this->groundtruth[i]);
        }

        cout << "Amount of query_ids: " << this->query_ids.size() << endl;
    }

    void
    readData(const string& dataset_path, const string& query_path) override {
        DataWrapper::readData(dataset_path, query_path);
    }

    void
    SaveToCSVRow(const string& path, const int idx, const int l_bound, const int r_bound, const int pos_range,
                 const int real_search_key_range, const int K_neighbor, const double& search_time,
                 const vector<int>& gt, const vector<float>& pts) {
        std::ofstream file;
        file.open(path, std::ios_base::app);
        if (file) {
            file << idx << "," << l_bound << "," << r_bound << "," << pos_range << "," << real_search_key_range << ","
                 << K_neighbor << "," << search_time << ",";
            for (auto ele : gt) {
                file << ele << " ";
            }
            file << "\n";
        }
        file.close();
    }

    void
    generateRangeFilteringQueriesAndGroundtruth(const string save_path) {
        std::default_random_engine e;
        timeval t1, t2;
        double accu_time = 0.0;

        vector<int> query_range_list;

        if (this->data_size == 1000000) {
            query_range_list = {1000,   2000,   3000,   4000,   5000,   6000,   7000,   8000,   9000,   10000,
                                20000,  30000,  40000,  50000,  60000,  70000,  80000,  90000,  100000, 200000,
                                300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000};
        } else {
            assert(false);
        }

        cout << "Generating Groundtruth...\nRanges: ";
        print_set(query_range_list);
        // generate groundtruth
        vector<std::pair<int, int>> query_ranges;
        for (auto range : query_range_list) {
            std::uniform_int_distribution<int> u_lbound(0, this->data_size - range - 80);
            for (int i = 0; i < this->querys.size(); i++) {
                int l_bound = u_lbound(e);
                int r_bound = l_bound + range - 1;
                if (range == this->data_size) {
                    l_bound = 0;
                    r_bound = this->data_size - 1;
                }
                int search_key_range = r_bound - l_bound + 1;
                query_ranges.emplace_back(std::make_pair(l_bound, r_bound));
                double greedy_time;
                gettimeofday(&t1, NULL);
                auto gt = greedyNearest(this->nodes, this->querys.at(i), l_bound, r_bound, this->query_k);

                vector<uint32_t> gt_vec(gt.begin(), gt.end());
                gettimeofday(&t2, NULL);
                CountTime(t1, t2, greedy_time);
                groundtruth.emplace_back(gt_vec);
                query_ids.emplace_back(i);
                accu_time += greedy_time;
                SaveToCSVRow(save_path, i, l_bound, r_bound, range, search_key_range, this->query_k, greedy_time, gt,
                             this->querys.at(i));
            }
        }

        cout << " Done!" << endl << "Groundtruth Time: " << accu_time << endl;
        cout << "Save GroundTruth to path: " << save_path << endl;
    }
};

// Utility function to create data wrapper based on dataset type and configuration
// This function can be reused across different executable methods
inline std::unique_ptr<DataWrapper>
createDataWrapperByType(std::string& dataset, std::string& dataset_type, int& query_num, int query_k, int data_size,
                        const std::string& dataset_path, const std::string& query_path,
                        const std::string& data_tag_path, const std::string& query_tag_path, const std::string& gt_path,
                        bool use_text_wrapper_for_hotpotqa = false) {
    std::unique_ptr<DataWrapper> data_wrapper;

    if (dataset_type.empty()) {
        dataset_type = "bitset";
    }
    // yfcc-arbitrary: a heterogeneous-predicate workload (tag AND id-range)
    // on the YFCC base vectors. The combined predicates are precomputed and
    // stored as a .gt + .bitsets pair, so it is consumed in plain bitset
    // mode -- but the base/query vectors still need YFCC's u8bin loader,
    // hence the DataWrapper is constructed with the internal name "yfcc".
    if (dataset == "yfcc-arbitrary") {
        auto w = std::make_unique<DataWrapper>(query_num, query_k, "yfcc", data_size);
        w->readData(dataset_path, query_path);
        w->LoadGroundtruth(gt_path);
        return w;
    }
    if (dataset == "yfcc" || dataset == "tripclick" || dataset == "arxiv") {
        dataset_type = "spmat";
    }

    if (dataset == "wiki-image-range") {
        dataset = "wiki-image";
        dataset_type = "range";
    }

    if (dataset == "hotpotqa") {
        if (use_text_wrapper_for_hotpotqa) {
            // Special case for hotpotqa with TextDataWrapper (used in some executables)
            auto hotpotqa_wrapper = std::make_unique<TextDataWrapper>(query_num, query_k, dataset, data_size);
            hotpotqa_wrapper->loadWithWords(dataset_path, query_path, data_tag_path, query_tag_path, gt_path);
            data_wrapper = std::move(hotpotqa_wrapper);
        } else {
            // Standard hotpotqa handling
            auto hotpotqa_wrapper = std::make_unique<DataWrapper>(query_num, query_k, dataset, data_size);
            hotpotqa_wrapper->readData(dataset_path, query_path);
            data_wrapper = std::move(hotpotqa_wrapper);
        }
    } else if (dataset_type == "bm25") {
        query_num = 100000;
        auto bm25_wrapper = std::make_unique<BM25DataWrapper>(query_num, query_k, dataset, data_size);
        bm25_wrapper->LoadBM25Groundtruth(dataset_path, query_path, data_tag_path, query_tag_path, gt_path);
        data_wrapper = std::move(bm25_wrapper);
    } else if (dataset == "laion") {
        query_num = 100000;
        auto text_wrapper = std::make_unique<TextDataWrapper>(query_num, query_k, dataset, data_size);
        text_wrapper->loadWithWords(dataset_path, query_path, data_tag_path, query_tag_path, gt_path);
        if (gt_path.empty()) {
            text_wrapper->LoadGroundtruth("../exp/groundtruth-text-querys-top10-" + dataset + "-" +
                                          std::to_string(data_size) + ".gt");
        } else {
            text_wrapper->LoadGroundtruth(gt_path);
        }
        data_wrapper = std::move(text_wrapper);
    } else if (dataset_type == "spmat") {
        if (dataset == "arxiv") {
            assert(data_size == 2735264);
            query_num = 10000;
            auto spmat_wrapper = std::make_unique<SPMATDataWrapper>(query_num, query_k, dataset, data_size);
            spmat_wrapper->loadArxivTags(dataset_path, query_path, data_tag_path, query_tag_path, gt_path);
            data_wrapper = std::move(spmat_wrapper);

        } else if (dataset == "yfcc") {
            auto spmat_wrapper = std::make_unique<SPMATDataWrapper>(query_num, query_k, dataset, data_size);

            // for single filter
            if (gt_path.find("single") != std::string::npos) {
                std::cout << " single tags query only " << std::endl;
                spmat_wrapper->loadWithTags(dataset_path, query_path, data_tag_path, query_tag_path, gt_path);
            } else if (gt_path.find("_or_") != std::string::npos) {
                std::cout << " multi tags or condition only " << std::endl;
                spmat_wrapper->loadWithTagsOrCondition(dataset_path, query_path, data_tag_path, query_tag_path,
                                                       gt_path);
            } else {
                std::cout << " multi tags and condition only " << std::endl;
                spmat_wrapper->loadWithTagsAndCondition(dataset_path, query_path, data_tag_path, query_tag_path,
                                                        gt_path);
            }
            data_wrapper = std::move(spmat_wrapper);
        }
    } else if (dataset_type == "bitset") {
        auto bitset_wrapper = std::make_unique<DataWrapper>(query_num, query_k, dataset, data_size);
        bitset_wrapper->readData(dataset_path, query_path);
        data_wrapper = std::move(bitset_wrapper);
        data_wrapper->LoadGroundtruth(gt_path);
    } else if (dataset_type == "range") {
        auto range_wrapper = std::make_unique<RangeDataWrapper>(query_num, query_k, dataset, data_size);
        range_wrapper->readData(dataset_path, query_path);
        range_wrapper->LoadGroundtruth(gt_path);
        data_wrapper = std::move(range_wrapper);
    }

    return data_wrapper;
}

int
queryRangeAdjustment(int query_range, std::string dataset) {
    if (dataset == "arxiv") {
        if (query_range <= 200000) {
            query_range = 200000;
        } else if (query_range <= 400000) {
            query_range = 400000;
        } else {
            query_range = 1000000;
        }
    }

    if (dataset == "yfcc") {
        if (query_range <= 10000) {
            query_range = 10000;
        } else if (query_range <= 100000) {
            query_range = 100000;
        } else if (query_range < 1000000) {
            query_range = 1000000;
        } else {
            query_range = 10000000;
        }
    }

    if (dataset == "laion") {
        if (query_range < 12000) {
            query_range = 12000;
        } else if (query_range < 30000) {
            query_range = 30000;
        } else {
            query_range = 1000000;
        }
    }

    if (dataset == "hotpotqa") {
        if (query_range < 1000000) {
            query_range = 1000000;
        } else if (query_range < 2000000) {
            query_range = 2000000;
        } else if (query_range < 3000000) {
            query_range = 3000000;
        } else if (query_range < 4000000) {
            query_range = 4000000;
        } else if (query_range < 5000000) {
            query_range = 5000000;
        } else {
            query_range = 6000000;
        }
    }
    return query_range;
}