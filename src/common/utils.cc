#include "utils.h"

#include "base_hnsw/hnswlib.h"
#include "bitset.h"

using std::cout;
using std::endl;
using std::getline;
using std::ifstream;
using std::ios;
using std::string;
using std::vector;

// l2 norm
float
EuclideanDistance(const vector<float>& lhs, const vector<float>& rhs, const int& startDim, int lensDim) {
    float ans = 0.0;
    if (lensDim == 0) {
        lensDim = lhs.size();
    }

    for (int i = startDim; i < startDim + lensDim; ++i) {
        ans += (lhs[i] - rhs[i]) * (lhs[i] - rhs[i]);
    }
    return std::sqrt(ans);
}

// l2sqr
float
EuclideanDistanceSquare(const vector<float>& lhs, const vector<float>& rhs) {
    float ans = 0.0;

    for (int i = 0; i < lhs.size(); ++i) {
        ans += (lhs[i] - rhs[i]) * (lhs[i] - rhs[i]);
    }
    return ans;
}

void
testUTIL2() {
    cout << "hello" << endl;
}

float
EuclideanDistance(const vector<float>& lhs, const vector<float>& rhs) {
    return EuclideanDistance(lhs, rhs, 0, 0);
}

// t1:begin, t2:end
void
AccumulateTime(timeval& t1, timeval& t2, double& val_time) {
    val_time += (t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) * 1.0 / CLOCKS_PER_SEC);
}

void
CountTime(timeval& t1, timeval& t2, double& val_time) {
    val_time = 0;
    val_time += (t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) * 1.0 / CLOCKS_PER_SEC);
}

double
CountTime(timeval& t1, timeval& t2) {
    double val_time = 0.0;
    val_time += (t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) * 1.0 / CLOCKS_PER_SEC);
    return val_time;
}

void
logTime(timeval& begin, timeval& end, const string& log) {
    gettimeofday(&end, NULL);
    fprintf(stdout, ("# " + log + ": %.7fs\n").c_str(),
            end.tv_sec - begin.tv_sec + (end.tv_usec - begin.tv_usec) * 1.0 / CLOCKS_PER_SEC);
};

// template <typename T>
// double
// countPrecision(const vector<T>& truth, const vector<T>& pred) {
//     double num_right = 0;
//     for (auto one : truth) {
//         if (find(pred.begin(), pred.end(), one) != pred.end()) {
//             num_right += 1;
//         }
//     }
//     return num_right / truth.size();
// }

// double
// countPrecision(const vector<uint32_t>& truth, const vector<uint32_t>& pred) {
//     double num_right = 0;
//     for (auto one : truth) {
//         if (find(pred.begin(), pred.end(), one) != pred.end()) {
//             num_right += 1;
//         }
//     }
//     return num_right / truth.size();
// }

// double
// countPrecision(const vector<size_t>& truth, const vector<size_t>& pred) {
//     double num_right = 0;
//     for (auto one : truth) {
//         if (find(pred.begin(), pred.end(), one) != pred.end()) {
//             num_right += 1;
//         }
//     }
//     return num_right / truth.size();
// }

double
countApproximationRatio(const vector<vector<float>>& raw_data, const vector<int>& truth, const vector<int>& pred,
                        const vector<float>& query) {
    if (pred.size() == 0) {
        return 0;
    }
    vector<float> truth_dist;
    vector<float> pred_dist;
    for (auto vec : truth) {
        truth_dist.emplace_back(EuclideanDistance(query, raw_data[vec]));
    }
    for (auto vec : pred) {
        pred_dist.emplace_back(EuclideanDistance(query, raw_data[vec]));
    }
    auto max_truth = *max_element(truth_dist.begin(), truth_dist.end());
    auto max_pred = *max_element(pred_dist.begin(), pred_dist.end());
    if (pred.size() < truth.size()) {
        nth_element(truth_dist.begin(), truth_dist.begin() + pred.size() - 1, truth_dist.end());
        max_truth = truth_dist[pred.size() - 1];
    }
    if (max_truth != 0)
        return max_pred / max_truth;
    cout << "ERROR: empty pred!" << endl;
    return -1;
}

void
print_memory() {
#ifdef __linux__
    struct sysinfo memInfo;

    sysinfo(&memInfo);
    // long long totalVirtualMem = memInfo.totalram;
    // // Add other values in next statement to avoid int overflow on right hand
    // // side...
    // totalVirtualMem += memInfo.totalswap;
    // totalVirtualMem *= memInfo.mem_unit;

    // long long virtualMemUsed = memInfo.totalram - memInfo.freeram;
    // // Add other values in next statement to avoid int overflow on right hand
    // // side...
    // virtualMemUsed += memInfo.totalswap - memInfo.freeswap;
    // virtualMemUsed *= memInfo.mem_unit;
    // cout << "Total Virtual Memory: " << totalVirtualMem << endl;
    // cout << "Used Virtual Memory: " << virtualMemUsed << endl;

    long long totalPhysMem = memInfo.totalram;
    // Multiply in next statement to avoid int overflow on right hand side...
    totalPhysMem *= memInfo.mem_unit;

    long long physMemUsed = memInfo.totalram - memInfo.freeram;
    // Multiply in next statement to avoid int overflow on right hand side...
    physMemUsed *= memInfo.mem_unit;

    // cout << "Total Physical Memory: " << totalPhysMem << endl;
    cout << "Used Physical Memory: " << physMemUsed << endl;
#elif __APPLE__
    vm_size_t page_size;
    mach_port_t mach_port;
    mach_msg_type_number_t count;
    vm_statistics64_data_t vm_stats;

    mach_port = mach_host_self();
    count = sizeof(vm_stats) / sizeof(natural_t);
    if (KERN_SUCCESS == host_page_size(mach_port, &page_size) &&
        KERN_SUCCESS == host_statistics64(mach_port, HOST_VM_INFO, (host_info64_t)&vm_stats, &count)) {
        long long free_memory = (int64_t)vm_stats.free_count * (int64_t)page_size;

        long long used_memory =
            ((int64_t)vm_stats.active_count + (int64_t)vm_stats.inactive_count + (int64_t)vm_stats.wire_count) *
            (int64_t)page_size;
        printf("free memory: %lld\nused memory: %lld\n", free_memory, used_memory);
    }
#endif
}

void
record_memory(long long& memory) {
#ifdef __linux__
    struct sysinfo memInfo;
    sysinfo(&memInfo);
    long long physMemUsed = memInfo.totalram - memInfo.freeram;
    physMemUsed *= memInfo.mem_unit;
    memory = physMemUsed;
#elif __APPLE__
    vm_size_t page_size;
    mach_port_t mach_port;
    mach_msg_type_number_t count;
    vm_statistics64_data_t vm_stats;

    mach_port = mach_host_self();
    count = sizeof(vm_stats) / sizeof(natural_t);
    if (KERN_SUCCESS == host_page_size(mach_port, &page_size) &&
        KERN_SUCCESS == host_statistics64(mach_port, HOST_VM_INFO, (host_info64_t)&vm_stats, &count)) {
        memory = ((int64_t)vm_stats.active_count + (int64_t)vm_stats.inactive_count + (int64_t)vm_stats.wire_count) *
                 (int64_t)page_size;
    }
#endif
}

vector<uint32_t>
greedyNearest(const vector<vector<float>>& dpts, const vector<float> query, const int k_smallest) {
    std::priority_queue<std::pair<float, int>> top_candidates;
    float lower_bound = _INT_MAX;
    for (size_t i = 0; i < dpts.size(); i++) {
        float dist = EuclideanDistanceSquare(query, dpts[i]);
        if (top_candidates.size() < k_smallest || dist < lower_bound) {
            top_candidates.push(std::make_pair(dist, i));
            if (top_candidates.size() > k_smallest) {
                top_candidates.pop();
            }

            lower_bound = top_candidates.top().first;
        }
    }
    vector<uint32_t> res;
    while (!top_candidates.empty()) {
        res.emplace_back(top_candidates.top().second);
        top_candidates.pop();
    }
    std::reverse(res.begin(), res.end());
    return res;
}

vector<uint32_t>
greedyNearest(const vector<vector<float>>& dpts, const vector<float> query, const int k_smallest,
              const BitsetView bitset) {
    std::priority_queue<std::pair<float, int>> top_candidates;
    float lower_bound = _INT_MAX;
    std::unique_ptr<hnswlib::L2Space> space(new hnswlib::L2Space(query.size()));
    auto dist_func = space->get_dist_func();
    auto dist_func_param = space->get_dist_func_param();
    for (size_t i = 0; i < dpts.size(); i++) {
        if (!bitset.test(i)) {
            float dist = dist_func(query.data(), dpts[i].data(), dist_func_param);
            if (top_candidates.size() < k_smallest || dist < lower_bound) {
                top_candidates.push(std::make_pair(dist, i));
                if (top_candidates.size() > k_smallest) {
                    top_candidates.pop();
                }

                lower_bound = top_candidates.top().first;
            }
        }
    }
    vector<uint32_t> res;
    while (!top_candidates.empty()) {
        res.emplace_back(top_candidates.top().second);
        top_candidates.pop();
    }
    std::reverse(res.begin(), res.end());
    return res;
}

vector<int> greedyNearest(const vector<vector<float>> &dpts,
                          const vector<float> query, const int l_bound,
                          const int r_bound, const int k_smallest) {
  std::priority_queue<std::pair<float, int>> top_candidates;
  float lower_bound = _INT_MAX;
  for (size_t i = l_bound; i <= r_bound; i++) {
    float dist = EuclideanDistance(query, dpts[i]);
    if (top_candidates.size() < k_smallest || dist < lower_bound) {
      top_candidates.push(std::make_pair(dist, i));
      if (top_candidates.size() > k_smallest) {
        top_candidates.pop();
      }

      lower_bound = top_candidates.top().first;
    }
  }
  vector<int> res;
  while (!top_candidates.empty()) {
    res.emplace_back(top_candidates.top().second);
    top_candidates.pop();
  }
  return res;
}

// void evaluateKNNG(const vector<vector<int>> &gt,
//                   const vector<vector<int>> &knng, const int K, double
//                   &recall, double &precision) {
//   assert(gt.size() == knng.size());

//   double all_right = 0;
//   int knng_amount = 0;

// #pragma omp parallel for reduction(+ : all_right) reduction(+ : knng_amount)
//   for (unsigned n = 0; n < gt.size(); n++) {
//     double num_right = 0;
//     // skip first, itself
//     for (unsigned i = 1; i < K + 1; i++) {
//       int one = gt[n][i];
//       if (find(knng[n].begin(), knng[n].end(), one) != knng[n].end()) {
//         num_right += 1;
//       }
//     }
//     all_right += num_right;
//     knng_amount += knng[n].size();
//   }
//   recall = (double)all_right / (K * gt.size());
//   precision = (double)all_right / (float)knng_amount;
// }

void
SaveToCSVRow(const string& path, const int idx, const int search_range, const int K_neighbor, const double& search_time,
             const vector<uint32_t>& gt, const vector<float>& pts) {
    std::ofstream file;
    file.open(path, std::ios_base::app);
    if (file) {
        file << idx << "," << search_range << "," << K_neighbor << "," << search_time << ",";
        for (auto ele : gt) {
            file << ele << " ";
        }
        // file << ",";
        // for (auto ele : pts) {
        //   file << ele << " ";
        // }
        file << "\n";
    }
    file.close();
}

void
SaveToCSVRow(const string& path, const string& method, const int idx, const int search_range, const int m, const int search_ef,
             const double& search_time, const double& recall, const int comps) {
    std::ofstream file;
    file.open(path, std::ios_base::app);
    if (file) {
        file << method << "," << idx << "," << search_range << "," << m << "," << search_ef << "," << search_time << ","
             << recall << "," << comps << "\n";
    }
    else {
        std::cerr << "Could not open the file for writing!" << std::endl;
        assert(false);
    }
    file.close();
}

void
SaveVecUintToFile(const std::vector<std::vector<uint8_t>>& vec, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Could not open the file for writing!" << std::endl;
        return;
    }

    // Write the number of rows and columns
    size_t rows = vec.size();
    size_t cols = vec.empty() ? 0 : vec[0].size();
    outFile.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    outFile.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

    for (const auto& row : vec) {
        // Write the data of the current row
        outFile.write(reinterpret_cast<const char*>(row.data()), cols * sizeof(uint8_t));
    }

    outFile.close();
}

void
LoadVecUintFromFile(std::vector<std::vector<uint8_t>>& vec, const std::string& filename) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Could not open the file for reading!" << std::endl;
        return;
    }

    // Read the number of rows and columns
    size_t rows, cols;
    inFile.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    inFile.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    cout << "rows: " << rows << "  cols: " << cols << endl;

    vec.clear();
    vec.resize(rows);

    for (size_t i = 0; i < rows; ++i) {
        vec[i].reserve(cols);
        // Read the data of the current row
        inFile.read(reinterpret_cast<char*>(vec[i].data()), cols * sizeof(uint8_t));
    }

    inFile.close();
    return;
}
