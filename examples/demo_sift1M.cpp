//
// Created by 刘嘉晨 on 2020-01-09.
//

#include "pq_table.h"
#include "utils.h"
#include "math.h"


float eucl_dist_vec(std::vector<float> a, std::vector<float> b){
    float res = 0;
#pragma omp parallel for
    for (size_t i = 0; i < a.size(); ++i){
        res += (a[i]-b[i])* (a[i]-b[i]);
    }

    return res;
}
int main(int argc, char *argv []){
    // (1) Make sure you have already downloaded siftsmall data in data/ by scripts/download_siftsmall.sh

    // (2) Read vectors
    std::vector<std::vector<float> > queries = pqtable::ReadTopN("../../data/sift/sift_query.fvecs", "fvecs");  // Because top_n is not set, read all vectors
    std::vector<std::vector<float> > bases = pqtable::ReadTopN("../../data/sift/sift_base.fvecs", "fvecs");
    std::vector<std::vector<float> > learns = pqtable::ReadTopN("../../data/sift/sift_learn.fvecs", "fvecs");
    //std::vector<std::vector<int> > label = pqtable::ReadTopN("../../data/siftsmall/siftsmall_learn.fvecs", "fvecs");

    std::cout<<"bases data shape: "<<std::endl;
    std::cout<<bases.size()<<" * "<<bases[0].size()<<std::endl;

    std::cout<<"learn data shape: "<<std::endl;
    std::cout<<learns.size()<<" * "<<learns[0].size()<<std::endl;

    std::cout<<"queries data shape: "<<std::endl;
    std::cout<<queries.size()<<" * "<<queries[0].size()<<std::endl;

    // (3) Train a product quantizer
    int M ;
    if(argc == 1){
        M =2;
    }else{
        M = atoi(argv[1]);
    }


    std::cout << "=== Train a product quantizer ===" << std::endl;
    pqtable::PQ pq(pqtable::PQ::Learn(learns, M));

    // (4) Encode vectors to PQ-codes
    std::cout << "=== Encode vectors into PQ codes ===" << std::endl;
    pqtable::UcharVecs codes = pq.Encode(bases);


    // (5) Build a PQTable
    std::cout << "=== Build PQTable ===" << std::endl;
    pqtable::PQTable tbl(pq.GetCodewords(), codes);


    // (6) Do search
    std::cout << "=== Do search ===" << std::endl;
    double t0 = pqtable::Elapsed();
    /*
    for(int q = 0; q < (int) queries.size(); ++q){
        std::pair<int, float> result = tbl.Query(queries[q]);  // result = (nearest_id, its_dist)
        std::cout << q << "th query: nearest_id=" << result.first << ", dist=" << result.second << std::endl;
    }
     */
    int top_k = 100;
    std::vector<std::vector<std::pair<int, float> > >
                                           ranked_scores(queries.size(), std::vector<std::pair<int, float> >(top_k));
    for(int q = 0; q < (int) queries.size(); ++q){
        ranked_scores[q] = tbl.Query(queries[q], top_k);
    }

    std::cout << ( pqtable::Elapsed() - t0) / queries.size() << " [sec/query]" << std::endl;


    std::vector<int> gt_index;
    gt_index.resize( queries.size());

    std::vector<float> gt_dis;
    gt_dis.resize( queries.size());
    float tmp_res = 0;
    std::cout << "=== Find ground truth ===" << std::endl;
    for(int q = 0; q < (int) queries.size(); ++q){
        int min_idx = -1;
        float min_dis = 1e20;
        for (size_t i = 0; i < bases.size(); ++i){
            tmp_res = eucl_dist_vec(bases[i],queries[q]);
            if (tmp_res < min_dis){
                min_dis = tmp_res;
                min_idx = i;
            }
        }
        gt_index[q] = min_idx;
        gt_dis[q] = min_dis;
    }
    /*
    std::cout << "=== Search Result ===" << std::endl;
    for(size_t q = 0; q < ranked_scores[0].size(); ++q){
        std::cout <<"#"<<q<< "# [ "<<  ranked_scores[0][q].second<< " vs. "<< eucl_dist_vec(bases[ranked_scores[0][q].first ],queries[0]) <<" ]. ";
    }
    std::cout << std::endl;
*/
    int n_1 = 0, n_10 = 0, n_100 = 0;
    for(size_t i = 0; i < queries.size(); i++) {
        int gt_nn = gt_index[i];
        // std::cout << i << "th query: nearest_id=" << gt_nn << ", dist=" << sqrt(gt_dis[i]) << std::endl;
        // std::cout << "PQ's nearest_id=" << ranked_scores[i][0].first  << ", with real dist = " << sqrt(eucl_dist_vec(queries[i], bases[gt_nn]) ) << std::endl;
        for(size_t j = 0; j < queries[0].size(); j++) {
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
