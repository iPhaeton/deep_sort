python2 deep_sort_app.py \
    --sequence_dir=./resources/sequences/intersection_test/ \
    --detection_file=./resources/detections/sequences/intersection_test.npy  \
    --min_confidence=0.3 \
    --nn_budget=10000 \
    --display=True \
    --nms_max_overlap=1 \
    --max_cosine_distance=0.1 \
    --max_age=10000 \
    --max_iou_distance=0.7
