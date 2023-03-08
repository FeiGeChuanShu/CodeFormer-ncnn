#ifndef MODEL_H
#define MODEL_H
#include <opencv2/opencv.hpp>
#include "net.h"

namespace wsdsb{ 

#define MAX_DET_FACE_COUNT 5
typedef struct _Object{
    cv::Rect_<float> rect;
    int label;
    float score;
    std::vector<cv::Point2f> pts;
    cv::Mat trans_inv;
    cv::Mat trans_img;
}Object_t;

typedef struct _Tensor{
    _Tensor() = default;
    _Tensor(ncnn::Mat& ncnn_data):data(ncnn_data){}
    ncnn::Mat data;
    int pad_h;
    int pad_w;
    int img_h;
    int img_w;
    int in_h;
    int in_w;
    float scale;
}Tensor_t;

typedef struct _CodeFormerResult{
    Object_t object;
    std::vector<Tensor_t> output_tensors;
    cv::Mat restored_face;
}CodeFormerResult_t;

typedef struct _PipeResult{
    int face_count;
    Object_t object[MAX_DET_FACE_COUNT];
    CodeFormerResult_t codeformer_result[MAX_DET_FACE_COUNT];
}PipeResult_t;


class Model
{
public:
    virtual ~Model(){};
    virtual int Load(const std::string& model_path) = 0;
    virtual int Process(const cv::Mat& input_img, void* result) = 0;
protected:
    
    virtual void Run(const std::vector<Tensor_t>& input_tensor, std::vector<Tensor_t>& output_tensor) = 0;
    virtual void PreProcess(const void* input_data, std::vector<Tensor_t>& input_tensor) = 0;
    virtual void PostProcess(const std::vector<Tensor_t>& input_tensor, std::vector<Tensor_t>& output_tensor, void* result) = 0;

};
}
#endif // MODEL_H
