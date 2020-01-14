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



float * fvecs_read_faiss (const char *fname,
                          size_t *d_out, size_t *n_out)
{
    FILE *f = fopen(fname, "r");
    if(!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);
    // std::cout<< "size : "<<n<<" . dim : "<<d<<std::endl;

    *d_out = d; *n_out = n;
    float *x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for(size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}



int *ivecs_read(const char *fname, size_t *d_out, size_t *n_out)
{
    return (int*)fvecs_read_faiss(fname, d_out, n_out);
}



int main(int argc, char *argv []){
    // (1) Make sure you have already downloaded siftsmall data in data/ by scripts/download_siftsmall.sh

    // (2) Read vectors
    std::vector<std::vector<float> > queries = pqtable::ReadTopN("../../gist/gist_query.fvecs", "fvecs");  // Because top_n is not set, read all vectors
    std::vector<std::vector<float> > bases = pqtable::ReadTopN("../../gist/gist_base.fvecs", "fvecs");
    std::vector<std::vector<float> > learns = pqtable::ReadTopN("../../gist/gist_learn.fvecs", "fvecs");
    //std::vector<std::vector<int> > label = pqtable::ReadTopN("../../data/siftsmall/siftsmall_learn.fvecs", "fvecs");

   std::cout<<"bases data shape: "<<std::endl;
    std::cout<<bases.size()<<" * "<<bases[0].size()<<std::endl;

    std::cout<<"learn data shape: "<<std::endl;
    std::cout<<learns.size()<<" * "<<learns[0].size()<<std::endl;

    std::cout<<"queries data shape: "<<std::endl;
    std::cout<<queries.size()<<" * "<<queries[0].size()<<std::endl;

    // (3) Train a product quantizer
    int M;
    assert(argc == 1 || argc == 2);
    if(argc == 1){
        M = 2;
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

    std::cout << ( pqtable::Elapsed() - t0) << " [msec]" << std::endl;
    char gnd_filename[50] = "../../gist/gist_groundtruth.ivecs";


    size_t nqq;
    size_t topk;
    int *gt_index = ivecs_read(gnd_filename, &topk, &nqq);

/*
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
