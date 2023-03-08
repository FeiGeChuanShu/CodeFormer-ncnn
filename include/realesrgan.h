#ifndef REALESRGAN_H
#define REALESRGAN_H

#include "include/model.h"
namespace wsdsb{ 
class RealESRGAN : public Model
{
public:
    RealESRGAN();
    ~RealESRGAN();
    int Load(const std::string& model_path) override;
    int Process(const cv::Mat& input_img, void* result) override;

protected:
    void Run(const std::vector<Tensor_t>& input_tensor, std::vector<Tensor_t>& output_tensor) override;
    void PreProcess(const void* input_data, std::vector<Tensor_t>& input_tensor) override;
    void PostProcess(const std::vector<Tensor_t>& input_tensor, std::vector<Tensor_t>& output_tensor, void* result) override;

private:
    void Tensor2Image(const ncnn::Mat& output_tensor, int img_h, int img_w, cv::Mat& output_img);
    int Padding(const cv::Mat& img, cv::Mat& pad_img, int& img_pad_h, int& img_pad_w);

    int scale;
    int tile_size;
    int tile_pad;
    const float norm_vals_[3] = { 1 / 255.0f, 1 / 255.0f, 1 / 255.0f };
    
private:
    ncnn::Net net_;
    std::vector<int> input_indexes_;
    std::vector<int> output_indexes_;
};
}
#endif // REALESRGAN_H
