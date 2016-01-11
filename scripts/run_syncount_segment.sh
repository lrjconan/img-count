python src/syncount_segment.py \
    -gpu $1 \
    -num_ex 10000 \
    -num_steps 10000 \
    -neg_pos_ratio 1 \
    -logs "/u/mren/public_html/results" \
    -results "/ais/gobi3/u/mren/results/img-count" \
    -localhost "www.cs.toronto.edu/~mren"
