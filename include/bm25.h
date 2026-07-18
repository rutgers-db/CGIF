#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <unordered_set>

using std::cout;
using std::endl;
class BM25 {
 public:
    BM25() = default;
    ~BM25() = default;

    void
    buildIndex(const std::vector<std::vector<std::string>>& documents) {
        // Clear existing index
        term_doc_freq_.clear();
        doc_lengths_.clear();
        doc_term_freq_.clear();
        term_to_docs_.clear();

        // Calculate document lengths and term frequencies
        doc_lengths_.resize(documents.size());
        doc_term_freq_.resize(documents.size());
        for (size_t doc_id = 0; doc_id < documents.size(); ++doc_id) {
            const auto& doc = documents[doc_id];
            doc_lengths_[doc_id] = doc.size();

            // Count term frequencies in this document
            std::unordered_map<std::string, int> term_freq;
            for (const auto& term : doc) {
                term_freq[term]++;
                term_doc_freq_[term]++;
                term_to_docs_[term].insert(doc_id);
            }
            doc_term_freq_[doc_id] = std::move(term_freq);
        }

        // Calculate average document length
        avg_doc_length_ = 0.0;
        for (size_t len : doc_lengths_) {
            avg_doc_length_ += len;
        }
        avg_doc_length_ /= doc_lengths_.size();
    }

    std::vector<std::pair<size_t, float>>
    search(const std::vector<std::string>& query_terms, size_t top_k = 10, float k1 = 1.2f, float b = 0.75f) {
        std::vector<std::pair<size_t, float>> scores;
        scores.reserve(doc_lengths_.size());

        // Calculate IDF for query terms
        std::unordered_map<std::string, float> term_idf;
        for (const auto& term : query_terms) {
            auto it = term_doc_freq_.find(term);
            if (it != term_doc_freq_.end()) {
                float idf = std::log((doc_lengths_.size() - it->second + 0.5f) / (it->second + 0.5f));
                term_idf[term] = idf;
            }
        }

        // Score each document
        for (size_t doc_id = 0; doc_id < doc_lengths_.size(); ++doc_id) {
            float score = 0.0f;
            const auto& doc_terms = doc_term_freq_[doc_id];
            float doc_len = doc_lengths_[doc_id];

            for (const auto& term : query_terms) {
                auto term_it = doc_terms.find(term);
                if (term_it != doc_terms.end()) {
                    float tf = term_it->second;
                    float idf = term_idf[term];

                    // BM25 scoring formula
                    float numerator = tf * (k1 + 1);
                    float denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_length_));
                    score += idf * (numerator / denominator);
                }
            }

            if (score > 0) {
                scores.emplace_back(doc_id, score);
            }
        }

        // Sort by score and return top-k
        std::sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) { return a.second > b.second; });

        if (scores.size() > top_k) {
            scores.resize(top_k);
        }

        return scores;
    }

    float
    getScore(const std::vector<std::string>& query_terms, size_t doc_id, float k1 = 1.2f, float b = 0.75f) {
        if (doc_id >= doc_lengths_.size()) {
            return 0.0f;
        }

        float score = 0.0f;
        const auto& doc_terms = doc_term_freq_[doc_id];
        float doc_len = doc_lengths_[doc_id];

        for (const auto& term : query_terms) {
            auto term_it = doc_terms.find(term);
            if (term_it != doc_terms.end()) {
                float tf = term_it->second;
                float idf =
                    std::log((doc_lengths_.size() - term_doc_freq_[term] + 0.5f) / (term_doc_freq_[term] + 0.5f));

                // BM25 scoring formula
                float numerator = tf * (k1 + 1);
                float denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_length_));
                score += idf * (numerator / denominator);
            }
        }

        return score;
    }

    // Get a reasonable threshold for filtering based on query terms
    float
    getFilterThreshold(const std::vector<std::string>& query_terms,
                       float percentile = 0.1f,  // Keep top 10% by default
                       float k1 = 1.2f, float b = 0.75f) {
        if (query_terms.empty() || doc_lengths_.empty()) {
            return 0.0f;
        }

        // Get scores for all documents
        std::vector<float> scores;
        scores.reserve(doc_lengths_.size());

        for (size_t doc_id = 0; doc_id < doc_lengths_.size(); ++doc_id) {
            float score = getScore(query_terms, doc_id, k1, b);
            if (score > 0) {
                scores.push_back(score);
            }
        }

        if (scores.empty()) {
            return 0.0f;
        }

        // Sort scores in descending order
        std::sort(scores.begin(), scores.end(), std::greater<float>());

        // Calculate the index for the desired percentile
        size_t threshold_idx = static_cast<size_t>(scores.size() * percentile);
        if (threshold_idx >= scores.size()) {
            threshold_idx = scores.size() - 1;
        }

        return scores[threshold_idx];
    }

    // Get documents that contain at least one query term
    std::unordered_set<size_t>
    getCandidateDocs(const std::vector<std::string>& query_terms) {
        std::unordered_set<size_t> candidate_docs;
        for (const auto& term : query_terms) {
            auto it = term_doc_freq_.find(term);
            if (it != term_doc_freq_.end()) {
                // For each document containing this term
                for (size_t doc_id = 0; doc_id < doc_lengths_.size(); ++doc_id) {
                    if (doc_term_freq_[doc_id].find(term) != doc_term_freq_[doc_id].end()) {
                        candidate_docs.insert(doc_id);
                    }
                }
            }
        }
        return candidate_docs;
    }

    // Get documents that pass the threshold (original version)
    std::vector<size_t>
    filterByThreshold(const std::vector<std::string>& query_terms, float threshold, float k1 = 1.2f, float b = 0.75f) {
        std::vector<size_t> filtered_docs;
        filtered_docs.reserve(doc_lengths_.size());

        for (size_t doc_id = 0; doc_id < doc_lengths_.size(); ++doc_id) {
            float score = getScore(query_terms, doc_id, k1, b);
            if (score >= threshold) {
                filtered_docs.push_back(doc_id);
            }
        }

        return filtered_docs;
    }

    // Get documents that pass the threshold (optimized version)
    std::vector<size_t>
    filterByThresholdFast(const std::vector<std::string>& query_terms, float threshold, float k1 = 1.2f, float b = 0.75f) {
        std::vector<size_t> filtered_docs;
        filtered_docs.reserve(doc_lengths_.size());

        // First collect all documents that contain at least one query term
        std::unordered_set<size_t> candidate_docs;
        for (const auto& term : query_terms) {
            auto it = term_to_docs_.find(term);
            if (it != term_to_docs_.end()) {
                // Add all documents containing this term
                candidate_docs.insert(it->second.begin(), it->second.end());
            }
        }

        // Only check documents that contain at least one query term
        for (size_t doc_id : candidate_docs) {
            float score = getScore(query_terms, doc_id, k1, b);
            if (score >= threshold) {
                filtered_docs.push_back(doc_id);
            }
        }

        return filtered_docs;
    }

 private:
    std::unordered_map<std::string, int> term_doc_freq_;               // Number of documents containing each term
    std::vector<int> doc_lengths_;                                     // Length of each document
    std::vector<std::unordered_map<std::string, int>> doc_term_freq_;  // Term frequencies per document
    std::unordered_map<std::string, std::unordered_set<size_t>> term_to_docs_;  // Mapping from terms to documents containing them, inverted index
    float avg_doc_length_;                                             // Average document length
};