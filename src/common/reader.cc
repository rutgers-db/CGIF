#include "reader.h"

#include <assert.h>

using std::vector;
FvecsItrReader::FvecsItrReader(std::string filename) {
    ifs.open(filename, std::ios::binary);
    assert(ifs.is_open());
    Next();
}

bool
FvecsItrReader::IsEnd() {
    return eof_flag;
}

std::vector<float>
FvecsItrReader::Next() {
    std::vector<float> prev_vec = vec;  // return the currently stored vec
    int D;
    if (ifs.read((char*)&D, sizeof(int))) {  // read "D"
        // Then, read a D-dim vec
        vec.resize(D);                                   // allocate D-dim
        ifs.read((char*)vec.data(), sizeof(float) * D);  // no assert version
        // assert(ifs.read((char *)vec.data(), sizeof(float) * D));  // Read D * float.
        eof_flag = false;
    } else {
        vec.clear();
        eof_flag = true;
    }
    return prev_vec;
}

BvecsItrReader::BvecsItrReader(std::string filename) {
    ifs.open(filename, std::ios::binary);
    assert(ifs.is_open());
    Next();
}

bool
BvecsItrReader::IsEnd() {
    return eof_flag;
}

std::vector<float>
BvecsItrReader::Next() {
    std::vector<float> prev_vec = vec;  // return the currently stored vec
    int D;
    if (ifs.read((char*)&D, sizeof(int))) {  // read "D"
        // Then, read a D-dim vec
        vec.resize(D);  // allocate D-dim
        std::vector<unsigned char> buff(D);

        ifs.read((char*)buff.data(),
                 sizeof(unsigned char) * D);  // no assert version
        // assert(ifs.read((char *)buff.data(),
        //                 sizeof(unsigned char) * D));  // Read D * uchar.

        // Convert uchar to float
        for (int d = 0; d < D; ++d) {
            vec[d] = static_cast<float>(buff[d]);
        }

        eof_flag = false;
    } else {
        vec.clear();
        eof_flag = true;
    }
    return prev_vec;
}

ItrReader::ItrReader(std::string filename, std::string ext) {
    if (ext == "fvecs") {
        m_reader = (I_ItrReader*)new FvecsItrReader(filename);
    } else if (ext == "bvecs") {
        m_reader = (I_ItrReader*)new BvecsItrReader(filename);
    } else {
        std::cerr << "Error: strange ext type: " << ext << "in ItrReader" << std::endl;
        exit(1);
    }
}

ItrReader::~ItrReader() {
    delete m_reader;
}

bool
ItrReader::IsEnd() {
    return m_reader->IsEnd();
}

std::vector<float>
ItrReader::Next() {
    return m_reader->Next();
}

std::vector<std::vector<float>>
ReadTopN(std::string filename, std::string ext, int top_n) {
    std::vector<std::vector<float>> vecs;
    if (top_n != -1) {
        vecs.reserve(top_n);
    }
    ItrReader reader(filename, ext);
    while (!reader.IsEnd()) {
        if (top_n != -1 && top_n <= (int)vecs.size()) {
            break;
        }
        vecs.push_back(reader.Next());
    }
    cout << "Finish Reading Data with " << vecs.size() << " vectors" << endl;
    return vecs;
}

/// @brief Reading binary data vectors. Raw data store as a (N x num_dimensions)
/// binary file.
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
/// @param N Reading top N vectors
/// @param num_dimensions dimension of dataset
void
ReadFvecsTopN(const std::string& file_path, std::vector<std::vector<float>>& data, const uint32_t N,
              const int num_dimensions) {
    std::cout << "Reading Data: " << file_path << std::endl;
    std::ifstream ifs;
    ifs.open(file_path, std::ios::binary);
    assert(ifs.is_open());

    data.resize(N);
    std::vector<float> buff(num_dimensions);
    int counter = 0;
    while ((counter < N) && (ifs.read((char*)buff.data(), num_dimensions * sizeof(float)))) {
        std::vector<float> row(num_dimensions);
        for (int d = 0; d < num_dimensions; d++) {
            row[d] = static_cast<float>(buff[d]);
        }
        data[counter++] = std::move(row);
    }

    ifs.close();
    std::cout << "Finish Reading Data with " << counter << " vectors" << endl;
    assert(counter == N);
}

/// @brief Reading binary data vectors. Raw data store as a (N x num_dimensions)
/// binary file.
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
/// @param N Reading top N vectors
/// @param num_dimensions dimension of dataset
void
ReadDoubleVecsTopN(const std::string& file_path, std::vector<std::vector<float>>& data, const uint32_t N,
                   const int num_dimensions) {
    std::cout << "Reading Data: " << file_path << std::endl;
    std::ifstream ifs;
    ifs.open(file_path, std::ios::binary);
    assert(ifs.is_open());

    data.resize(N);
    std::vector<double> buff(num_dimensions);
    int counter = 0;
    while ((counter < N) && (ifs.read((char*)buff.data(), num_dimensions * sizeof(double)))) {
        std::vector<float> row(num_dimensions);
        for (int d = 0; d < num_dimensions; d++) {
            row[d] = static_cast<float>(buff[d]);
        }
        data[counter++] = std::move(row);
    }

    ifs.close();
    std::cout << "Finish Reading Data with " << counter << " vectors" << endl;
    assert(counter == N);
}

/// @brief Reading binary data vectors. Raw data store as a (N x 100)
/// binary file. Skip some nodes, for reading querys
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
/// @param N Reading top N vectors
/// @param num_dimensions dimension of dataset
void
ReadFvecsSkipTop(const std::string& file_path, std::vector<std::vector<float>>& data, const uint32_t N,
                 const int num_dimensions, const int skip_num) {
    std::cout << "Query Start From Position: " << skip_num << std::endl;
    std::ifstream ifs;
    ifs.open(file_path, std::ios::binary);
    assert(ifs.is_open());
    ifs.seekg(num_dimensions * sizeof(double) * skip_num);

    data.resize(N);
    std::vector<double> buff(num_dimensions);
    int counter = 0;
    while ((counter < N) && ifs.read((char*)buff.data(), num_dimensions * sizeof(double))) {
        std::vector<float> row(num_dimensions);
        for (int d = 0; d < num_dimensions; d++) {
            row[d] = static_cast<float>(buff[d]);
        }
        data[counter++] = std::move(row);
    }

    ifs.close();
}

/// @brief Reading metadata information, stored in uint32_t format
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
/// @param N Reading top N vectors
/// @param num_dimensions dimension of dataset
void
ReadIvecsTopN(const std::string& file_path, std::vector<int>& keys, const uint32_t N, const int num_dimensions,
              const int position) {
    std::cout << "Reading Keys: " << file_path << std::endl;
    std::ifstream ifs;
    ifs.open(file_path, std::ios::binary);
    assert(ifs.is_open());

    keys.resize(N);
    std::vector<uint64_t> buff(num_dimensions);
    int counter = 0;
    while ((counter < N) && ifs.read((char*)buff.data(), num_dimensions * sizeof(uint64_t))) {
        keys[counter++] = static_cast<uint64_t>(buff[position]);
    }

    ifs.close();
    std::cout << "Finish Reading Keys" << endl;
}

void
Split(std::string& s, std::string& delim, std::vector<std::string>* ret) {
    size_t last = 0;
    size_t index = s.find_first_of(delim, last);
    while (index != std::string::npos) {
        ret->push_back(s.substr(last, index - last));
        last = index + 1;
        index = s.find_first_of(delim, last);
    }
    if (index - last > 0) {
        ret->push_back(s.substr(last, index - last));
    }
}

// load txt matrix data
void
ReadMatFromTxt(const string& path, vector<vector<float>>& data, const int length_limit = -1) {
    ifstream infile;
    string bline;
    string delim = " ";
    int numCols = 0;
    infile.open(path, ios::in);
    if (getline(infile, bline, '\n')) {
        vector<string> ret;
        Split(bline, delim, &ret);
        numCols = ret.size();
    }
    infile.close();
    // cout << "Reading " << path << " ..." << endl;
    // cout << "# of columns: " << numCols << endl;

    int counter = 0;
    if (length_limit == -1)
        counter = -9999999;
    // TODO: read sparse matrix
    infile.open(path, ios::in);
    while (getline(infile, bline, '\n')) {
        if (counter >= length_limit)
            break;
        counter++;

        vector<string> ret;
        Split(bline, delim, &ret);
        vector<float> arow(numCols);
        assert(ret.size() == numCols);
        for (int i = 0; i < ret.size(); i++) {
            arow[i] = static_cast<float>(stod(ret[i]));
        }
        data.emplace_back(arow);
    }
    infile.close();
    // cout << "# of rows: " << data.size() << endl;
}

void
ReadMatFromTxtTwitter(const string& path, vector<vector<float>>& data, const int length_limit = -1) {
    ifstream infile;
    string bline;
    string delim = " ";
    int numCols = 0;
    infile.open(path, ios::in);
    if (getline(infile, bline, '\n')) {
        vector<string> ret;
        Split(bline, delim, &ret);
        numCols = ret.size() - 1;
    }
    infile.close();
    cout << "Reading " << path << " ..." << endl;

    cout << "# of columns: " << numCols << endl;

    int counter = 0;
    if (length_limit == -1)
        counter = -9999999;
    // TODO: read sparse matrix
    infile.open(path, ios::in);
    while (getline(infile, bline, '\n')) {
        if (counter >= length_limit)
            break;
        counter++;

        vector<string> ret;
        Split(bline, delim, &ret);
        vector<float> arow(numCols);
        assert(ret.size() == numCols + 1);
        for (int i = 1; i < ret.size(); i++) {
            arow[i - 1] = static_cast<float>(stod(ret[i]));
        }
        data.emplace_back(arow);
    }
    infile.close();
    cout << "# of rows: " << data.size() << endl;
}

void
ReadMatFromTsv(const string& path, vector<vector<float>>& data, const int length_limit = -1) {
    ifstream infile;
    string bline;
    string delim = "\t";
    int numCols = 0;
    infile.open(path, ios::in);
    getline(infile, bline, '\n');
    if (getline(infile, bline, '\n')) {
        vector<string> ret;
        Split(bline, delim, &ret);
        numCols = ret.size();
    }
    infile.close();
    cout << "Reading " << path << " ..." << endl;
    cout << "# of columns: " << numCols << endl;

    int counter = 0;
    if (length_limit == -1)
        counter = -9999999;
    infile.open(path, ios::in);
    // skip the first line
    getline(infile, bline, '\n');
    while (getline(infile, bline, '\n')) {
        if (counter >= length_limit)
            break;
        counter++;

        vector<string> ret;
        Split(bline, delim, &ret);
        vector<float> arow(numCols - 1);
        assert(ret.size() == numCols);
        for (int i = 0; i < ret.size() - 1; i++) {
            arow[i] = static_cast<float>(stod(ret[i + 1]));
        }
        data.emplace_back(arow);
    }
    infile.close();
    cout << "# of rows: " << data.size() << endl;
}

int
YT8M2Int(const string id) {
    int res = 0;
    for (size_t i = 0; i < 4; i++) {
        res *= 100;
        res += (int)id[i] - 38;
    }
    return res;
}

void
ReadMatFromTsvYT8M(const string& path, vector<vector<float>>& data, vector<int>& search_keys, const int length_limit) {
    ifstream infile;
    string bline;
    string delim = ",";
    int numCols = 0;
    infile.open(path, ios::in);
    getline(infile, bline, '\n');
    if (getline(infile, bline, '\n')) {
        vector<string> ret;
        Split(bline, delim, &ret);
        numCols = ret.size();
    }
    infile.close();
    cout << "Reading " << path << " ..." << endl;
    cout << "# of columns: " << numCols << endl;

    int counter = 0;
    if (length_limit == -1)
        counter = -9999999;
    infile.open(path, ios::in);
    string delim_embed = " ";

    while (getline(infile, bline, '\n')) {
        if (counter >= length_limit)
            break;
        counter++;

        vector<string> ret;
        Split(bline, delim, &ret);
        assert(ret.size() == numCols);

        // str 'id' to int 'id'
        // int one_search_key = YT8M2Int(ret[0]);
        int one_search_key = (int)stod(ret[1]);

        // add embedding
        string embedding_str = ret[2];
        vector<string> embedding_vec;
        vector<float> arow(1024);
        Split(embedding_str, delim_embed, &embedding_vec);
        assert(embedding_vec.size() == 1024);
        for (int i = 0; i < embedding_vec.size() - 1; i++) {
            arow[i] = static_cast<float>(stod(embedding_vec[i + 1]));
        }
        search_keys.emplace_back(one_search_key);
        data.emplace_back(arow);
    }
    infile.close();
    cout << "# of rows: " << data.size() << endl;
}

void
ReadMatFromTsvYT8M(const string& path, vector<vector<float>>& data, const int length_limit) {
    ifstream infile;
    string bline;
    string delim = ",";
    int numCols = 0;
    infile.open(path, ios::in);
    getline(infile, bline, '\n');
    if (getline(infile, bline, '\n')) {
        vector<string> ret;
        Split(bline, delim, &ret);
        numCols = ret.size();
    }
    infile.close();
    cout << "Reading " << path << " ..." << endl;
    cout << "# of columns: " << numCols << endl;

    int counter = 0;
    if (length_limit == -1)
        counter = -9999999;
    infile.open(path, ios::in);
    string delim_embed = " ";

    while (getline(infile, bline, '\n')) {
        if (counter >= length_limit)
            break;
        counter++;

        vector<string> ret;
        Split(bline, delim, &ret);
        assert(ret.size() == numCols);

        // add embedding
        string embedding_str = ret[2];
        vector<string> embedding_vec;
        vector<float> arow(1024);
        Split(embedding_str, delim_embed, &embedding_vec);
        assert(embedding_vec.size() == 1024);
        for (int i = 0; i < embedding_vec.size() - 1; i++) {
            arow[i] = static_cast<float>(stod(embedding_vec[i + 1]));
        }
        data.emplace_back(arow);
    }
    infile.close();
    cout << "# of rows: " << data.size() << endl;
}

void
ReadGroundtruthQuery(vector<vector<int>>& gt, vector<std::pair<int, int>>& query_ranges, vector<int>& query_ids,
                     string gt_path) {
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
        vector<int> one_gt;
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
            one_gt.emplace_back(std::stoi(ele));
        }
        gt.emplace_back(one_gt);
        query_ranges.emplace_back(one_range);
        query_ids.emplace_back(one_id);
    }
}

void
fvecs2csv(const string& output_path, const vector<vector<float>>& nodes) {
    std::ofstream file;
    file.open(output_path, std::ios_base::app);
    for (auto row : nodes) {
        if (file) {
            for (auto ele : row) {
                file << ele << " ";
            }
            file << "\n";
        }
    }
    file.close();
}

template <typename T>
void
read_bin(const char* filename, uint32_t& nd, uint32_t& dim, T*& data) {
    std::ifstream reader(filename, std::ios::in);
    reader.read((char*)&nd, sizeof(uint32_t));
    reader.read((char*)&dim, sizeof(uint32_t));
    if (reader.fail()) {
        std::cerr << "Failed to read from file. Check permissions or file state." << std::endl;
        assert(false);
    }
    if (data != nullptr) {
        delete[] data;
    }
    std::cout << nd << ", " << dim << std::endl;
    size_t data_size = (size_t)nd * (size_t)dim;
    data = new T[data_size];
    reader.read((char*)data, sizeof(T) * data_size);
    reader.close();
}

// load data and querys
void
ReadDataWrapper(const string& dataset, string& dataset_path, vector<vector<float>>& raw_data, const int data_size) {
    raw_data.clear();
    if (dataset == "glove" || dataset == "glove25" || dataset == "glove50" || dataset == "glove100" ||
        dataset == "glove200") {
        if (dataset_path == "")
            dataset_path = "/common/home/cz296/workspace/rank21/glove/glove.twitter.27B.50d.txt";
        ReadMatFromTxtTwitter(dataset_path, raw_data, data_size);
    } else if (dataset == "ml25m") {
        if (dataset_path == "")
            dataset_path =
                "/common/home/cz296/workspace/rank21/model_data/"
                "model-vectors/users.txt";
        ReadMatFromTxt(dataset_path, raw_data, data_size);
    } else if (dataset == "sift") {
        if (dataset_path == "")
            dataset_path = "/data/local/pqdata/bigann/base.bvecs";
        raw_data = ReadTopN(dataset_path, "bvecs", data_size);

    } else if (dataset == "biggraph") {
        if (dataset_path == "")
            dataset_path = "/data/local/biggraph/embedding/wikidata_translation_v1.tsv";
        ReadMatFromTsv(dataset_path, raw_data, data_size);
    } else if (dataset == "local") {
        if (dataset_path == "")
            dataset_path =
                "/Users/ciuji/dd_workspace/project7-rangeKNN/"
                "range-knn-code/data/siftsmall_base.fvecs";
        cout << dataset_path << endl;
        raw_data = ReadTopN(dataset_path, "fvecs", data_size);
    } else if (dataset == "deep") {
        if (dataset_path == "")
            dataset_path = "/data/local/pqdata/deep1b/base.fvecs";
        raw_data = ReadTopN(dataset_path, "fvecs", data_size);

    } else if (dataset == "deep10m") {
        if (dataset_path == "")
            dataset_path = "/common/home/cz296/dataset/deep10M.fvecs";
        raw_data = ReadTopN(dataset_path, "fvecs", data_size);

    } else if (dataset == "yt8m") {
        if (dataset_path == "")
            dataset_path =
                "/data/local/embedding_dataset/youtube-8m/"
                "yt8m-video-embedding-publish-timestamp.csv";
        ReadMatFromTsvYT8M(dataset_path, raw_data, data_size);
    } else if (dataset == "yt8m-video") {
        ReadDoubleVecsTopN(dataset_path, raw_data, data_size, 1024);

    } else if (dataset == "yt8m-audio") {
        ReadDoubleVecsTopN(dataset_path, raw_data, data_size, 128);

    } else if (dataset == "wiki-image") {
        if (dataset_path == "")
            dataset_path =
                "/data/local/embedding_dataset/wikipedia-image-caption/"
                "embedding_metadata_anns/wiki_image_embedding.fvecs";
        ReadDoubleVecsTopN(dataset_path, raw_data, data_size, 2048);

    } else if (dataset == "laion") {
        ReadFvecsTopN(dataset_path, raw_data, data_size, 768);
    } else if (dataset == "tripclick") {
        ReadFvecsTopN(dataset_path, raw_data, data_size, 768);
    } else if (dataset == "hotpotqa") {
        ReadFvecsTopN(dataset_path, raw_data, data_size, 1024);
    } else if (dataset == "yfcc") {
        // Read binary data for SPMAT format
        uint32_t nd, dim;
        uint8_t* temp_data = nullptr;
        read_bin(dataset_path.c_str(), nd, dim, temp_data);

        raw_data.resize(nd);
        for (int i = 0; i < nd; i++) {
            std::vector<float> one_vec(dim);
            for (int j = 0; j < dim; j++) {
                one_vec[j] = static_cast<float>(temp_data[i * dim + j]);
            }
            raw_data[i] = std::move(one_vec);
        }

        delete[] temp_data;
    } else if (dataset == "arxiv") {
        ReadFvecsTopN(dataset_path, raw_data, data_size, 4096);
    } else {
        std::cerr << "Wrong Datset!" << endl;
        assert(false);
    }
}

void
read_ivf_clusters(const string& ivf_index_path, vector<int>* ivf_data) {
    // Open the binary file in read mode
    std::ifstream ifs(ivf_index_path, std::ios::binary);
    if (!ifs) {
        std::cerr << "Could not open the file!" << std::endl;
        assert(false);
    }

    // Determine the file size
    auto fileSize = std::filesystem::file_size(ivf_index_path);

    // Calculate the number of elements (assuming uint8_t type)
    size_t numElements = fileSize / sizeof(int32_t);

    std::cout << "# of elements of ivf: " << numElements << std::endl;

    ivf_data->clear();
    ivf_data->resize(numElements);

    // Read the binary data into the vector
    ifs.read(reinterpret_cast<char*>(ivf_data->data()), fileSize);

    // Close the file
    ifs.close();

    std::cout << "max cluster: " << *std::max_element(ivf_data->cbegin(), ivf_data->cend()) + 1 << std::endl;
}

void
read_ivf_centroids(const string& ivf_index_path, vector<vector<float>>* ivf_centroids, size_t num_centroids,
                   size_t dim) {
    // Open the binary file in read mode
    std::ifstream ifs(ivf_index_path, std::ios::binary);
    if (!ifs) {
        std::cerr << "Could not open the centroids file: " << ivf_index_path << std::endl;
        assert(false);
    }

    std::cout << "Reading " << num_centroids << " centroids with dimension " << dim << std::endl;

    // Calculate expected file size
    size_t fileSize = num_centroids * dim * sizeof(float);

    // Resize the output vector to hold all centroids
    ivf_centroids->clear();
    ivf_centroids->resize(num_centroids);

    // Read each centroid
    for (int i = 0; i < num_centroids; i++) {
        ivf_centroids->at(i).resize(dim);
        ifs.read(reinterpret_cast<char*>(ivf_centroids->at(i).data()), dim * sizeof(float));
    }

    // check the end of the file
    ifs.seekg(0, std::ios::end);
    auto end_pos = ifs.tellg();
    if (end_pos != fileSize) {
        std::cerr << "Error: End of file does not match expected size!" << std::endl;
        assert(false);
    }

    // Close the file
    ifs.close();

    std::cout << "Successfully read " << ivf_centroids->size() << " centroids" << std::endl;
}

void
ReadTagTsv(const string& path, std::vector<TagData>& tag_data) {
    ifstream infile;
    string bline;
    string delim = "\t";

    cout << "Reading tag data from: " << path << endl;
    infile.open(path, ios::in);
    if (!infile.is_open()) {
        cout << "Failed to open tag file: " << path << endl;
        assert(false);
    }

    // Skip header line
    getline(infile, bline);

    while (getline(infile, bline)) {
        vector<string> ret;
        Split(bline, delim, &ret);
        if (ret.size() != 5) {
            cout << "Invalid line format in tag file: " << bline << endl;
            continue;
        }

        TagData data;
        data.query_id = std::stoi(ret[0]);

        // Parse tags (can be comma-separated)
        string tag_str = ret[1];
        if (!tag_str.empty()) {
            vector<string> tag_tokens;
            string tag_delim = ",";
            Split(tag_str, tag_delim, &tag_tokens);
            for (const auto& tag : tag_tokens) {
                if (!tag.empty()) {
                    data.tags.push_back(std::stoi(tag));
                }
            }
        }

        data.query_text = ret[2];
        data.doc_id = std::stoi(ret[3]);
        data.date_created = std::stoll(ret[4]);

        tag_data.push_back(data);
    }

    infile.close();
    cout << "Read " << tag_data.size() << " tag entries" << endl;
}

void
ReadCSVWords(const string& path, std::shared_ptr<std::vector<std::vector<std::string>>> words, const int n_cols, const int word_col_idx) {
    ifstream infile;
    string bline;
    string delim = ",";
    string space_delim = " ";
    words->clear();

    cout << "Reading LAION lemmas from: " << path << endl;
    infile.open(path, ios::in);
    if (!infile.is_open()) {
        cout << "Failed to open LAION metadata file: " << path << endl;
        assert(false);
    }

    // Skip header line
    getline(infile, bline);

    while (getline(infile, bline)) {
        vector<string> ret;
        Split(bline, delim, &ret);
        if (ret.size() < n_cols) {
            cout << "Invalid line format in LAION metadata file: " << bline << endl;
            continue;
        }

        // Split the lemmas (third column) into individual words
        vector<string> lemma_words;
        Split(ret[word_col_idx], space_delim, &lemma_words);
        words->emplace_back(lemma_words);
    }

    infile.close();
    cout << "Read " << words->size() << " word rows" << endl;
}


void
ReadCSVWords(const string& path, std::shared_ptr<std::vector<std::unordered_set<std::string>>> words, const int n_cols, const int word_col_idx) {
    ifstream infile;
    string bline;
    string delim = ",";
    string space_delim = " ";
    words->clear();

    cout << "Reading LAION lemmas from: " << path << endl;
    infile.open(path, ios::in);
    if (!infile.is_open()) {
        cout << "Failed to open LAION metadata file: " << path << endl;
        assert(false);
    }

    // Skip header line
    getline(infile, bline);

    while (getline(infile, bline)) {
        vector<string> ret;
        Split(bline, delim, &ret);
        if (ret.size() < n_cols) {
            cout << "Invalid line format in LAION metadata file: " << bline << endl;
            continue;
        }

        // Split the lemmas (third column) into individual words
        vector<string> lemma_words;
        Split(ret[word_col_idx], space_delim, &lemma_words);
        std::unordered_set<string> one_set(lemma_words.begin(), lemma_words.end());
        words->emplace_back(one_set);
    }

    infile.close();
    cout << "Read " << words->size() << " word rows" << endl;
}

