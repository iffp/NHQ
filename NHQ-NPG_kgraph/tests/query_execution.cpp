#include <chrono>

#include "efanna2e/index_random.h"
#include "efanna2e/index_graph.h"
#include "efanna2e/util.h"

#include "fanns_survey_helpers.cpp"

using namespace std;

int main(int argc, char **argv){
    // Get number of threads
    unsigned int nthreads = std::thread::hardware_concurrency();
    std::cout << "Number of threads: " << nthreads << std::endl;

    // Parameters
    std::string path_query_vectors;
    std::string path_query_attributes;
    std::string path_groundtruth;
    std::string path_index;
    int k;
    int weight_search;
    int L_search;
	int n_items;

    // Check if the number of arguments is correct
    if (argc != 9)
    {
        fprintf(stderr, "Usage: %s <path_query_vectors> <path_query_attributes> <path_groundtruth> <path_index> <k> <weight_search> <L_search> <n_items>\n", argv[0]);
        exit(1);
    }

    // Read command line arguments
    path_query_vectors = argv[1];
    path_query_attributes = argv[2];
    path_groundtruth = argv[3];
    path_index = argv[4];
    k = atoi(argv[5]);
    weight_search = atoi(argv[6]);
	L_search = atoi(argv[7]);
	n_items = atoi(argv[8]);

	// Setting seed
	unsigned seed = 161803398;
	srand(seed);

	// Read query vectors
	int n_queries;
	int d;
	float *query_vectors = nullptr;
	load_data(path_query_vectors.c_str(), query_vectors, n_queries, d);
	query_vectors = efanna2e::data_align(query_vectors, n_queries, d);

	// Read query attributes
	vector<vector<int>> query_attributes = read_one_int_per_line(path_query_attributes);
	assert(query_attributes.size() == n_queries);

    // Transform query attributes into format required by NHQ
    std::vector<std::vector<std::string>> query_attributes_str;
    for (std::size_t i = 0; i < query_attributes.size(); ++i) {
        query_attributes_str.push_back({std::to_string(query_attributes[i])});
    }

	// Read ground-truth
	vector<vector<int>> groundtruth = read_ivecs(path_groundtruth);
	assert(groundtruth.size() == n_queries);

	// Truncate ground-truth to at most k items
    for (std::vector<int>& vec : groundtruth) {
        if (vec.size() > k) {
            vec.resize(k);
        }
    }

	// Load NHQ index
	efanna2e::IndexRandom init_index(d, n_items);
	efanna2e::IndexGraph nhq_index(d, n_items, efanna2e::FAST_L2, (efanna2e::Index *)(&init_index));
    std::string index_path_model = path_index + "_model";
    std::string index_path_attribute_table = path_index + "_attribute_table";
	nhq_index.Load(index_path_model.c_str());
	nhq_index.LoadAttributeTable(index_path_attribute_table.c_str());

	// Prepare search parameters
	efanna2e::Parameters paras;
	paras.Set<unsigned>("L_search", L_search);
	paras.Set<float>("weight_search", weight_search);

	// Prepare results
	std::vector<std::vector<unsigned>> result(n_queries);
	for (unsigned i = 0; i < n_queries; i++){
		result[i].resize(k);
	}

	// Perform the search (this is timed)
	auto start_time = std::chrono::high_resolution_clock::now();
	for (unsigned i = 0; i < n_queries; i++)
	{
		nhq_index.SearchWithOptGraph(query_attributes_str[i], query_vectors + i * d, k, paras, result[i].data());
	}
	auto end_time = std::chrono::high_resolution_clock::now();

	// Compute search time
	std::chrono::duration<double> diff = end_time - start_time;
	double query_execution_time = diff.count();

	// Compute recall
	size_t match_count = 0;
	size_t total_count = 0;
	for (int i = 0; i < n_queries; i++){
		int n_valid_neighbors = min(k, (int)groundtruth[i].size());
		vector<int> groundtruth_q = groundtruth[i];
		vector<int> result_q;
		for (int j = 0; j < k; j++){
			result_q.push_back(result[i][j]);
		}
		sort(groundtruth_q.begin(), groundtruth_q.end());
		sort(result_q.begin(), result_q.end());
		vector<int> intersection;
		set_intersection(groundtruth_q.begin(), groundtruth_q.end(), result_q.begin(), result_q.end(), back_inserter(intersection));
		match_count += intersection.size();
		total_count += n_valid_neighbors;
	}

	// Report results
	double recall = (double)match_count / total_count;	
	double qps = n_queries / query_execution_time;
	peak_memory_footprint();
	printf("Queries per second: %.3f\n", qps);
	printf("Recall: %.3f\n", recall);
	
	return 0;
}
