/*
 * Created on Mon Mar 31 2025
 *
 * Adapter to use faiss::IndexIVFFLAT within the IVFConnect framework
 *
 * Author: Chaoji Zuo
 * Email: chaoji.zuo@rutgers.edu
 *
 * Copyright (c) 2025 Rutgers University - Database Group
 */

#pragma once
#include <memory>
#include <unordered_map>
#include <vector>

#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "ivf.h"

using std::vector;

class FaissIVFAdapter : public IVFIndex {
 private:
    std::unique_ptr<faiss::IndexIVFFlat> faiss_ivf;
    // No need to duplicate these - they're already in the base class
    // vector<vector<uint32_t>> clusters; 
    // std::unique_ptr<vector<int>> labels;
    // float average_cluster_size = 0;

 public:
    faiss::IVFSearchParameters ivf_params;

    FaissIVFAdapter(std::unique_ptr<faiss::Index> index) {
        faiss::IndexIVFFlat* ivf_index = dynamic_cast<faiss::IndexIVFFlat*>(index.release());
        if (!ivf_index) {
            throw std::runtime_error("Error: The provided index is not an IVF index");
        }
        faiss_ivf.reset(ivf_index);

        // Initialize base class data members
        labels = std::make_unique<vector<int>>(faiss_ivf->ntotal, -1);
        clusters.resize(faiss_ivf->nlist);
        
        // Use invlists to get the clusters
        for (size_t i = 0; i < faiss_ivf->nlist; i++) {
            const auto& ids = faiss_ivf->invlists->get_ids(i);
            for (size_t j = 0; j < faiss_ivf->invlists->list_size(i); j++) {
                clusters[i].push_back(ids[j]);
                labels->at(ids[j]) = i;
            }
        }

        faiss::Index* raw_quantizer = ivf_index->quantizer;

        // 2. Cast the quantizer to the correct type (IndexFlat or IndexFlatL2)
        // The quantizer is usually an IndexFlatL2 in this setup.
        faiss::IndexFlat* flat_quantizer = dynamic_cast<faiss::IndexFlat*>(raw_quantizer);

        if (flat_quantizer) {
            // 3. Access the centroids from the 'xb' member
            // The centroids are stored as a flat array of floats.
            const float* centroids = flat_quantizer->get_xb();
            size_t num_centroids = flat_quantizer->ntotal;
            int d = flat_quantizer->d;

            std::cout << "Number of centroids: " << num_centroids << std::endl;
            std::cout << "Dimension of centroids: " << d << std::endl;
            ivf_centroids_data.resize(num_centroids, vector<float>(d));
            for (size_t i = 0; i < num_centroids; i++) {
                memcpy(ivf_centroids_data[i].data(), centroids + i * d, d * sizeof(float));
            }
            ivf_centroids = std::make_unique<vector<vector<float>>>(ivf_centroids_data);
            std::cout << "Centroids extracted successfully." << std::endl;
            
        } else {
            std::cerr << "Could not cast quantizer to IndexFlat." << std::endl;
        }
    }

    ~FaissIVFAdapter() = default;

    // Only override methods that need special behavior
    float
    get_avg_cluster_size() override {
        if (average_cluster_size != 0) {
            return average_cluster_size;
        }

        float total = faiss_ivf->ntotal;
        float num_clusters = faiss_ivf->nlist;
        average_cluster_size = total / num_clusters;
        return average_cluster_size;
    }

    // // Add method to extract centroids from FAISS
    // vector<float>
    // get_cluster_centroid(size_t id) const override {
    //     // const float* centroids = faiss_ivf->quantizer->xb.data();
    //     // size_t d = faiss_ivf->d;
    //     // vector<float> centroid(centroids + id * d, centroids + (id + 1) * d);
    //     // return centroid;
    //     // TODO: implement this
    //     return vector<float>();
    // }

    // Add FAISS-specific functionality
    void
    search_ivf(size_t n, const void* x, size_t target_num_clusters, size_t search_ef, const faiss::idx_t* keys,
               const float* coarse_dis, float* distances, faiss::idx_t* labels, faiss::IndexIVFStats* stats = nullptr) {
        ivf_params.nprobe = target_num_clusters;
        faiss_ivf->invlists->prefetch_lists(keys, target_num_clusters);
        faiss_ivf->search_preassigned(n, static_cast<const float*>(x), search_ef, keys, coarse_dis, distances, labels,
                                      false, &ivf_params, stats);
    }

    faiss::IndexIVF*
    get_ivf_index() {
        return faiss_ivf.get();
    }
};
