#!/bin/bash
for i in 0k_4.660023_0.0900_0.0009 1k_4.248024_0.1019_0.0014 2k_4.030818_0.1023_0.0018 3k_3.898883_0.1009_0.0020 4k_3.809681_0.0947_0.0018 5k_3.747255_0.0927_0.0018 6k_3.695711_0.0971_0.0022 7k_3.664356_0.0973_0.0023 8k_3.646517_0.0918_0.0020 9k_3.628164_0.0900_0.0020 10k_3.612882_0.0898_0.0023 11k_3.598083_0.0910_0.0024 16k_3.557172_0.0913_0.0026 17k_3.556151_0.0915_0.0023 18k_3.551231_0.0923_0.0022 19k_3.550244_0.0921_0.0024 20k_3.546327_0.0908_0.0022 25k_3.536638_0.0937_0.0023 27k_3.536686_0.0950_0.0025 30k_3.530621_0.0934_0.0021 33k_3.526204_0.0904_0.0020 37k_3.523051_0.0914_0.0022 38k_3.523850_0.0954_0.0025 39k_3.525022_0.0944_0.0022 40k_3.520265_0.0895_0.0021 50k_3.516380_0.0887_0.0018 60k_3.514197_0.0918_0.0011 61k_3.513682_0.0890_0.0011 70k_3.510970_0.0898_0.0017 79k_3.509613_0.0903_0.0018 87k_3.507851_0.0911_0.0018 92k_3.507298_0.0886_0.0019 97k_3.506227_0.0919_0.0017 103k_3.503578_0.0926_0.0023 110k_3.502673_0.0852_0.0020 115k_3.501992_0.0916_0.0020 
do 
    ./infer_align.sh $i
    python3 cut.py ./pred_dir/pred_rl_align_0909.txt ./pred_dir/pos.txt
done