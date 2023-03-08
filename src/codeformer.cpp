// codeformer implemented with ncnn library
#include "include/encoder.h"
#include "include/generator.h"
#include "include/codeformer.h"
namespace wsdsb{ 
CodeFormer::CodeFormer():encoder_(new Encoder),generator_(new Generator)
{
}
CodeFormer::~CodeFormer()
{
    generator_.reset();
    encoder_.reset();
}

int CodeFormer::Load(const std::string& model_path)
{
    int ret = encoder_->Load(model_path);
    if(ret < 0)
    {
        return -1;
    }

    ret = generator_->Load(model_path);
    if(ret < 0)
    {
        return -1;
    }
    return 0;
}

int CodeFormer::Process(const cv::Mat& img, CodeFormerResult_t& model_result)
{
    encoder_->Process(img, (void*)&model_result); // output_tensor
    generator_->Process(img, (void*)&model_result); // restored_face

    return 0;
}
}