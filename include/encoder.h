#ifndef ENCODER_H
#define ENCODER_H
#include "include/model.h"
namespace wsdsb{ 
class Encoder : public Model
{
public:
    ~Encoder();
    int Process(const cv::Mat& input_img, void* result);
    int Load(const std::string& model_path) override;
protected:
    void Run(const std::vector<Tensor_t>& input_tensor, std::vector<Tensor_t>& output_tensor) override;
    void PreProcess(const void* input_data, std::vector<Tensor_t>& input_tensor) override;
    void PostProcess(const std::vector<Tensor_t>& input_tensor, std::vector<Tensor_t>& output_tensor, void* result) override;
    
private:
    void Topk(const ncnn::Mat& cls_scores, int topk, std::vector<float>& top_idx);
    void GetCodebookFeat(const ncnn::Mat& soft_one_hot, ncnn::Mat& min_encodings);
    const float mean_vals_[3] = { 127.5f, 127.5f, 127.5f };
    const float norm_vals_[3] = { 1 / 127.5f, 1 / 127.5f, 1 / 127.5f };
    ncnn::Net encoder_net_;

    std::vector<int> input_indexes_;
    std::vector<int> output_indexes_;

};
}
#endif // ENCODER_H
