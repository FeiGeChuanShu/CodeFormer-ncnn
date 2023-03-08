#ifndef GENERATOR_H
#define GENERATOR_H
#include "include/model.h"

namespace wsdsb{ 
class Generator : public Model
{
public:
    ~Generator();
    int Load(const std::string& model_path) override;
    int Process(const cv::Mat& input_img, void* result) override;
protected:
    void Run(const std::vector<Tensor_t>& input_tensor, std::vector<Tensor_t>& output_tensor) override;
    void PreProcess(const void* input_data, std::vector<Tensor_t>& input_tensor) override;
    void PostProcess(const std::vector<Tensor_t>& input_tensor, std::vector<Tensor_t>& output_tensor, void* result) override;

private:
    void Normlize(const ncnn::Mat& output, std::vector<float>& output_norm);
    void Tensor2Image(std::vector<float>& output_tensor, int img_h, int img_w, cv::Mat& output_img);
    ncnn::Net generator_net_;
    std::vector<int> input_indexes_;
    std::vector<int> output_indexes_;
};
}
#endif // GENERATOR_H
