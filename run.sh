python2 deep_sort_app.py \
    --sequence_dir=./resources/sequences/intersection_test/ \
    --detection_file=./resources/detections/sequences/intersection_test_pretrained_hard_cosine.npy  \
    --min_confidence=0 \
    --nn_budget=10 \
    --display=True \
    --nms_max_overlap=1 \
    --max_cosine_distance=0.2 \
    --max_age=10000 \
    --max_iou_distance=0.7
