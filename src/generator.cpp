#include "include/generator.h"
namespace wsdsb{ 
Generator::~Generator()
{
    generator_net_.clear();
}

int Generator::Load(const std::string& model_path)
{
    std::string generator_param_path = model_path + "/generator.param";
    std::string generator_model_path = model_path + "/generator.bin";

    generator_net_.opt.use_vulkan_compute = false;
    generator_net_.opt.use_fp16_packed = false;
    generator_net_.opt.use_fp16_storage = false;
    generator_net_.opt.use_fp16_arithmetic = false;
    generator_net_.opt.use_bf16_storage = true;

    int ret = generator_net_.load_param(generator_param_path.c_str());
    if (ret < 0)
    {
        fprintf(stderr, "open param file %s failed\n", generator_param_path.c_str());
        return -1;
    }
    ret = generator_net_.load_model(generator_model_path.c_str());
    if (ret < 0)
    {
        fprintf(stderr, "open bin file %s failed\n", generator_model_path.c_str());
        return -1;
    }

    input_indexes_.resize(6);
    const auto &blobs = generator_net_.blobs();
    for (int i = 0; i != blobs.size(); ++i) 
    {
        const auto &b = blobs[i];
        if (b.name == "enc_feat_32")  
            input_indexes_[0] = i;
        if (b.name == "enc_feat_64")  
            input_indexes_[1] = i;
        if (b.name == "enc_feat_128") 
            input_indexes_[2] = i;
        if (b.name == "enc_feat_256") 
            input_indexes_[3] = i;
        if (b.name == "style_feat")      
            input_indexes_[4] = i;
        if (b.name == "input") 
            input_indexes_[5] = i;
    }

    for(const auto& output : generator_net_.output_indexes())
    {
        output_indexes_.push_back(output);
    }

    return 0;
}

void Generator::PreProcess(const void* input_data, std::vector<Tensor_t>& input_tensor)
{

}
void Generator::Normlize(const ncnn::Mat& output, std::vector<float>& output_norm)
{
    int size = output.c * output.h * output.w;

    output_norm.resize(size);
    std::copy((float*)output.data, ((float*)output.data) + size, output_norm.begin());

    for (int i = 0; i != output_norm.size(); ++i) 
    {
        float val = output_norm[i] > 1.0 ? 1.0 : (output_norm[i] < -1.0 ? -1.0 : output_norm[i]);
        output_norm[i] = (val + 1.0) / 2.0;
    }
}

void Generator::Tensor2Image(std::vector<float>& output_tensor, int img_h, int img_w, cv::Mat& output_img)
{
    std::vector<cv::Mat> mat_list;
    for (int i = 0; i != 3; ++i) 
    {
        cv::Mat mat = cv::Mat(img_h, img_w, CV_32FC1, (void*)(output_tensor.data() + i * img_w * img_h));
        mat_list.push_back(mat);
    }

    cv::Mat result_img_f;
    cv::merge(mat_list, result_img_f);

    cv::Mat result_img;
    result_img_f.convertTo(result_img, CV_8UC3, 255.0, 0);

    cv::cvtColor(result_img, output_img, cv::COLOR_RGB2BGR);
}
void Generator::PostProcess(const std::vector<Tensor_t>& input_tensor, std::vector<Tensor_t>& output_tensor, void* result)
{
    ncnn::Mat& out = output_tensor[0].data;

    std::vector<float> out_norm;
    Normlize(out, out_norm);

    cv::Mat out_img;
    Tensor2Image(out_norm, out.h, out.w, out_img);

    out_img.copyTo(((CodeFormerResult_t*)result)->restored_face);
}
void Generator::Run(const std::vector<Tensor_t>& input_tensor, std::vector<Tensor_t>& output_tensor)
{
    ncnn::Extractor generator_ex = generator_net_.create_extractor();

    for(int i = 0; i != input_indexes_.size(); ++i)
    {
        generator_ex.input(input_indexes_[i], input_tensor[i].data);
    }

    for(int i = 0; i != output_indexes_.size(); ++i)
    {
        ncnn::Mat out;
        generator_ex.extract(output_indexes_[i], out);
        //Tensor_t tensor_t(out);
        //tensor_t.data = out;
        output_tensor.push_back(Tensor_t(out));
    }
}

int Generator::Process(const cv::Mat& input_img, void* result)
{
    std::vector<Tensor_t> output_tensor;
    Run(((CodeFormerResult_t*)result)->output_tensors, output_tensor);
    PostProcess(((CodeFormerResult_t*)result)->output_tensors, output_tensor, result);

    return 0;
}
}