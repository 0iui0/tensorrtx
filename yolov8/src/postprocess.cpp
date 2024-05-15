#include "postprocess.h"
#include "utils.h"

cv::Rect get_rect_adapt_landmark(cv::Mat& img, float bbox[4], float lmk[kNumberOfPoints * 3]) {
    float l, r, t, b;
    float r_w = kInputW / (img.cols * 1.0);
    float r_h = kInputH / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] / r_w;
        r = bbox[2] / r_w;
        t = (bbox[1] - (kInputH - r_w * img.rows) / 2) / r_w;
        b = (bbox[3] - (kInputH - r_w * img.rows) / 2) / r_w;
        for (int i = 0; i < kNumberOfPoints * 3; i += 3) {
            lmk[i] /= r_w;
            lmk[i + 1] = (lmk[i + 1] - (kInputH - r_w * img.rows) / 2) / r_w;
            // lmk[i + 2]
        }
    } else {
        l = (bbox[0] - (kInputW - r_h * img.cols) / 2) / r_h;
        r = (bbox[2] - (kInputW - r_h * img.cols) / 2) / r_h;
        t = bbox[1] / r_h;
        b = bbox[3] / r_h;
        for (int i = 0; i < kNumberOfPoints * 3; i += 3) {
            lmk[i] = (lmk[i] - (kInputW - r_h * img.cols) / 2) / r_h;
            lmk[i + 1] /= r_h;
            // lmk[i + 2]
        }
    }
    l = std::max(0.0f, l);
    t = std::max(0.0f, t);
    int width = std::max(0, std::min(int(round(r - l)), img.cols - int(round(l))));
    int height = std::max(0, std::min(int(round(b - t)), img.rows - int(round(t))));

    return cv::Rect(int(round(l)), int(round(t)), width, height);
}

static float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
            (std::max)(lbox[0], rbox[0]),
            (std::min)(lbox[2], rbox[2]),
            (std::max)(lbox[1], rbox[1]),
            (std::min)(lbox[3], rbox[3]),
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    float unionBoxS = (lbox[2] - lbox[0]) * (lbox[3] - lbox[1]) + (rbox[2] - rbox[0]) * (rbox[3] - rbox[1]) - interBoxS;
    return interBoxS / unionBoxS;
}

static bool cmp(const Detection& a, const Detection& b) {
    if (a.conf == b.conf) {
        return a.bbox[0] < b.bbox[0];
    }
    return a.conf > b.conf;
}

void nms(std::vector<Detection>& res, float* output, float conf_thresh, float nms_thresh) {
    int det_size = sizeof(Detection) / sizeof(float);
    std::map<float, std::vector<Detection>> m;

    for (int i = 0; i < output[0]; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh)
            continue;
        Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0)
            m.emplace(det.class_id, std::vector<Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

void process_decode_ptr_host(std::vector<Detection>& res, const float* decode_ptr_host, int bbox_element, cv::Mat& img,
                             int count) {
    Detection det;
    for (int i = 0; i < count; i++) {
        int basic_pos = 1 + i * bbox_element;
        int keep_flag = decode_ptr_host[basic_pos + 6];
        if (keep_flag == 1) {
            det.bbox[0] = decode_ptr_host[basic_pos + 0];
            det.bbox[1] = decode_ptr_host[basic_pos + 1];
            det.bbox[2] = decode_ptr_host[basic_pos + 2];
            det.bbox[3] = decode_ptr_host[basic_pos + 3];
            det.conf = decode_ptr_host[basic_pos + 4];
            det.class_id = decode_ptr_host[basic_pos + 5];
            res.push_back(det);
        }
    }
}

void draw_bbox_keypoints_line(cv::Mat& img, std::vector<Detection>& res) {
    const std::vector<std::pair<int, int>> skeleton_pairs = {
            {0, 1}, {0, 2},  {0, 5}, {0, 6},  {1, 2},   {1, 3},   {2, 4},   {5, 6},   {5, 7},  {5, 11},
            {6, 8}, {6, 12}, {7, 9}, {8, 10}, {11, 12}, {11, 13}, {12, 14}, {13, 15}, {14, 16}};
    for (size_t j = 0; j < res.size(); j++) {
        cv::Rect r = get_rect_adapt_landmark(img, res[j].bbox, res[j].keypoints);
        cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
        cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2,
                    cv::Scalar(0xFF, 0xFF, 0xFF), 2);

        for (int k = 0; k < kNumberOfPoints * 3; k += 3) {
            if (res[j].keypoints[k + 2] > 0.5) {
                cv::circle(img, cv::Point((int)res[j].keypoints[k], (int)res[j].keypoints[k + 1]), 3,
                            cv::Scalar(0, 0x27, 0xC1), -1);
            }
        }

        for (const auto& bone : skeleton_pairs) {
            int kp1_idx = bone.first * 3;
            int kp2_idx = bone.second * 3;
            if (res[j].keypoints[kp1_idx + 2] > 0.5 && res[j].keypoints[kp2_idx + 2] > 0.5) {
                cv::Point p1((int)res[j].keypoints[kp1_idx], (int)res[j].keypoints[kp1_idx + 1]);
                cv::Point p2((int)res[j].keypoints[kp2_idx], (int)res[j].keypoints[kp2_idx + 1]);
                cv::line(img, p1, p2, cv::Scalar(0, 0x27, 0xC1), 2);
            }
        }
    }

}

