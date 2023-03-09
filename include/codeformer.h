// codeformer implemented with ncnn library

#ifndef CODEFORMER_H
#define CODEFORMER_H
#include <memory>
#include "include/model.h"
namespace wsdsb{ 
class Encoder;
class Generator;
class CodeFormer
{
public:
    CodeFormer();
    ~CodeFormer();
    
    int Load(const std::string& model_path);
    int Process(const cv::Mat& input_img, CodeFormerResult_t& model_result);

private:
    std::unique_ptr<Encoder> encoder_;
    std::unique_ptr<Generator> generator_;
};
}
#endif // CODEFORMER_H
