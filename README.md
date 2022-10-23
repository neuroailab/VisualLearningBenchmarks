# How Well Do Unsupervised Learning Algorithms Model Human Real-time and Life-long Learning?

Code to reproduce results in "How Well Do Unsupervised Learning Algorithms Model Human Real-time and Life-long Learning?" (to appear in NeurIPS 2022 Dataset and Benchmark Track).

In this work, we build two benchmarks to compare the real-time and life-long visual learning dynamics between unsupervised algorithms and humans. The two folders are correspondingly for each benchmark.

## Life-long benchmark

Check the README at the `life_long` folder. This benchmark needs the SAYCam dataset for training and the ImageNet dataset for testing to fully reproduce the results.

## Real-time benchmark

Check the README at the `real_time_related` folder. This benchmark provides the pretrained models (see the README at `real_time_related/real_time_test/pytorch_scripts`) but still needs the ImageNet and VGGFace2 datasets (see instructions in the same README about how to download this dataset).
