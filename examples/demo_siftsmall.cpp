#include "pq_table.h"
#include "utils.h"

float eucl_dist_vec(std::vector<float> a, std::vector<float> b){
    float res = 0;
#pragma omp parallel for
    for (int i = 0; i < a.size(); ++i){
        res += (a[i]-b[i])* (a[i]-b[i]);
    }

return res;
}

int main(){
    // (1) Make sure you have already downloaded siftsmall data in data/ by scripts/download_siftsmall.sh

    // (2) Read vectors
    std::vector<std::vector<float> > queries = pqtable::ReadTopN("../../data/siftsmall/siftsmall_query.fvecs", "fvecs");  // Because top_n is not set, read all vectors
    std::vector<std::vector<float> > bases = pqtable::ReadTopN("../../data/siftsmall/siftsmall_base.fvecs", "fvecs");
    std::vector<std::vector<float> > learns = pqtable::ReadTopN("../../data/siftsmall/siftsmall_learn.fvecs", "fvecs");

    std::cout<<"bases data shape: "<<std::endl;
    std::cout<<bases.size()<<" * "<<bases[0].size()<<std::endl;

    std::cout<<"learn data shape: "<<std::endl;
    std::cout<<learns.size()<<" * "<<learns[0].size()<<std::endl;


    std::cout<<"queries data shape: "<<std::endl;
    std::cout<<queries.size()<<" * "<<queries[0].size()<<std::endl;


    // (3) Train a product quantizer
    int M = 4;
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

    std::cout << (pqtable::Elapsed() - t0) / queries.size() * 1000 << " [msec/query]" << std::endl;

    std::vector<int> gt_index;
    gt_index.resize( queries.size());
    float tmp_res = 0;
    std::cout << "=== Find ground truth ===" << std::endl;
    for(int q = 0; q < (int) queries.size(); ++q){
        int min_idx = -1;
        float min_dis = 1e20;
        for (int i = 0; i < bases.size(); ++i){
            tmp_res = eucl_dist_vec(bases[i],queries[q]);
            if (tmp_res < min_dis){
                min_dis = tmp_res;
                min_idx = i;
            }
        }
        gt_index[q] = min_idx;

    }



    int n_1 = 0, n_10 = 0, n_100 = 0;
    for(int i = 0; i < queries.size(); i++) {
        int gt_nn = gt_index[i];
        for(int j = 0; j < k; j++) {
            if (ranked_scores[i][j].first == gt_nn ){
                if(j < 1) n_1++;
                if(j < 10) n_10++;
                if(j < 100) n_100++;
            }
        }
    }
    printf("R@1 = %.4f\n", n_1 / float(nq));
    printf("R@10 = %.4f\n", n_10 / float(nq));
    printf("R@100 = %.4f\n", n_100 / float(nq));




    return 0;
}
