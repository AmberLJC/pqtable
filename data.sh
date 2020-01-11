bash scripts/download_sift1b.sh

cd data/

tar -xzf bigann_learn.bvecs.gz

tar -xzf bigann_base.bvecs.gz

tar -xzf bigann_query.bvecs.gz

cd ../build/

make -j 40

cd bin/

./demo_sift1b_train

./demo_sift1b_encode

./demo_sift1b_build_table

./demo_sift1b_search

