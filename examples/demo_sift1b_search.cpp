#include "pq_table.h"
#include "utils.h"



float eucl_dist_vec(std::vector<float> a, std::vector<float> b){
    float res = 0;
#pragma omp parallel for
    for (size_t i = 0; i < a.size(); ++i){
        res += (a[i]-b[i])* (a[i]-b[i]);
    }

    return res;
}
int main(int argc, char *argv []){
    double t0 = pqtable::Elapsed();
    int top_k;
    assert(argc == 1 || argc == 2);
    if(argc == 1){
        top_k = 1;
    }else{
        top_k = atoi(argv[1]);
    }
    std::cout << "top_k: " << top_k << std::endl;


    // (1) Make sure you've already run "demo_sift1b_train", "demo_sift1b_encode",
    //     "demo_sift1b_build_table". The "pqtable" dir must be in the bin dir.

    // (2) Read query vectors
    std::vector<std::vector<float> > queries = pqtable::ReadTopN("../../data/bigann_query.bvecs", "bvecs");

    std::cout<<"queries data shape: "<<std::endl;
    std::cout<<queries.size()<<" * "<<queries[0].size()<<std::endl;

    // (3) Read the PQTable
    pqtable::PQTable table("pqtable");

    // (4) Search

    // ranked_scores[q][k] : top-k th result of the q-th query.
    std::vector<std::vector<std::pair<int, float> > >
            ranked_scores(queries.size(), std::vector<std::pair<int, float> >(top_k));

    std::cout<<"Start searching: "<<std::endl;

    double t1 = pqtable::Elapsed() -t0;

    for(int q = 0; q < (int) queries.size(); ++q){
        ranked_scores[q] = table.Query(queries[q], top_k);
    }
    double t2 = pqtable::Elapsed() -t0;

    std::cout << (t2-t1) << " [sec] for " <<  queries.size()<< " queries"<< std::endl;

    // (5) Write scores
    pqtable::WriteScores("score.txt", ranked_scores);




    std::vector<int> gt_index;
    gt_index.resize( queries.size());

    std::vector<float> gt_dis;
    gt_dis.resize( queries.size());
    //float tmp_res = 0;
    std::cout << "=== Find ground truth ===" << std::endl;

/*
    pqtable::ItrReader reader("../../data/gnd/idx_1000M.ivecs", "fvecs");
    std::vector<std::vector<float> > buff;  // Buffer
    int id_encoded = 0;

    std::cout << "Start reading" << std::endl;
    while(!reader.IsEnd()){
        buff.push_back(reader.Next());  // Read a line (a vector) into the buffer
        id_encoded++;
        if (id_encoded % 10000 == 0)
            std::cout << "Read "<<id_encoded <<" data." << std::endl;
    }

    for(int q = 0; q < (int) queries.size(); ++q){
        int min_idx = -1;
        float min_dis = 1e20;
        for (size_t i = 0; i < buff.size(); ++i){
            tmp_res = eucl_dist_vec(buff[i],queries[q]);
            if (tmp_res < min_dis){
                min_dis = tmp_res;
                min_idx = i;
            }
        }
        gt_index[q] = min_idx;
        gt_dis[q] = min_dis;
    }

*/

    size_t nq ;
    size_t topk ;
    int *gt_knn = pqtable::ivecs_read("../../data/gnd/idx_1000M.ivecs", &topk, &nq);


    std::cout << "ground truth shape: "<< nq<<" * " << topk << std::endl;

    int n_1 = 0, n_10 = 0, n_100 = 0;
    for(size_t i = 0; i < queries.size(); i++) {
        int gt_nn = gt_knn[ i*topk ];
         std::cout << i << "th query: nearest_id = " << gt_nn  << std::endl;
        // std::cout << "PQ's nearest_id=" << ranked_scores[i][0].first  << ", with real dist = " << sqrt(eucl_dist_vec(queries[i], bases[gt_nn]) ) << std::endl;
        for(int j = 0; j < top_k; j++) {
            if (ranked_scores[i][j].first == gt_nn ){
                if(j < 1) n_1++;
                if(j < 10) n_10++;
                if(j < 100) n_100++;
            }
        }
    }
    printf("R@1 = %.3f\n", n_1 / float(queries.size()));
    printf("R@10 = %.3f\n", n_10 / float(queries.size()));
    printf("R@100 = %.3f\n", n_100 / float(queries.size()));

    return 0;
}
