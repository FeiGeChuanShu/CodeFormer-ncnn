#ifndef FACE_H
#define FACE_H
#include <cmath>
#include "include/model.h"
namespace wsdsb{ 
class Face : public Model
{
public:
    Face();
    ~Face();
    int Load(const std::string& model_path) override;
    int Process(const cv::Mat& input_img, void* result) override;

protected:
    void Run(const std::vector<Tensor_t>& input_tensor, std::vector<Tensor_t>& output_tensor) override;
    void PreProcess(const void* input_data, std::vector<Tensor_t>& input_tensor) override;
    void PostProcess(const std::vector<Tensor_t>& input_tensor, std::vector<Tensor_t>& output_tensor, void* result) override;
private:
    void AlignFace(const cv::Mat& img, Object_t& objects);
    void draw_objects(const cv::Mat& bgr, const std::vector<Object_t>& objects);
private:
    float prob_threshold;
    float nms_threshold;
    const float norm_vals_[3] = { 1 / 255.0f, 1 / 255.0f, 1 / 255.0f };
    std::vector<cv::Point2f> face_template;
    ncnn::Net net_;
    std::vector<int> input_indexes_;
    std::vector<int> output_indexes_;
};
}
#endif // FACE_H
