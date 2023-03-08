#include "include/encoder.h"
namespace wsdsb{ 
Encoder::~Encoder()
{
    encoder_net_.clear();
}

void Encoder::Topk(const ncnn::Mat& cls_scores, int topk, std::vector<float>& top_idx)
{
    top_idx.resize(cls_scores.h);
    for (int n = 0; n != cls_scores.h; ++n) 
    {
        const float* cls_ptr = cls_scores.row(n);
        int size = cls_scores.w;
        std::vector<std::pair<float, int> > vec;
        vec.resize(size);
        for (int i = 0; i != size; ++i)
        {
            vec[i] = std::make_pair(cls_ptr[i], i);
        }

        std::partial_sort(vec.begin(), vec.begin() + 1, vec.end(), 
            [](const std::pair<float, int>& lhs, const std::pair<float, int>& rhs) -> bool{return lhs.first > rhs.first;});

        int index = vec[0].second;
        top_idx[n] = index;    
    }

}
void Encoder::GetCodebookFeat(const ncnn::Mat& soft_one_hot, ncnn::Mat& min_encodings)
{
    std::vector<float> top_idx;
    Topk(soft_one_hot, 1, top_idx);

    min_encodings.create(1024, 256);
    min_encodings.fill(0.f);
    for (int i = 0; i != soft_one_hot.h; ++i) 
    {
        float* ptr = min_encodings.row(i);
        ptr[static_cast<int>(top_idx[i])] = 1;
    }
}

int Encoder::Load(const std::string& model_path)
{
    std::string encoder_param_path = model_path + "/encoder.param";
    std::string encoder_model_path = model_path + "/encoder.bin";
    encoder_net_.opt.use_vulkan_compute = false;
    //encoder_net_.opt.num_threads = 4;
    encoder_net_.opt.use_fp16_packed = false;
    encoder_net_.opt.use_fp16_storage = false;
    encoder_net_.opt.use_fp16_arithmetic = false;
    encoder_net_.opt.use_bf16_storage = true;

    int ret = encoder_net_.load_param(encoder_param_path.c_str());
    if (ret < 0)
    {
        fprintf(stderr, "open param file %s failed\n", encoder_param_path.c_str());
        return -1;
    }
    ret = encoder_net_.load_model(encoder_model_path.c_str());
    if (ret < 0)
    {
        fprintf(stderr, "open bin file %s failed\n", encoder_model_path.c_str());
        return -1;
    }
    
    for(const auto& input : encoder_net_.input_indexes())
    {
        input_indexes_.push_back(input);
    }

    output_indexes_.resize(6);
    const auto &blobs = encoder_net_.blobs();
    for (int i = 0; i != blobs.size(); ++i) 
    {
        const auto &b = blobs[i];
        if (b.name == "enc_feat_32")  
            output_indexes_[0] = i;
        if (b.name == "enc_feat_64")  
            output_indexes_[1] = i;
        if (b.name == "enc_feat_128") 
            output_indexes_[2] = i;
        if (b.name == "enc_feat_256") 
            output_indexes_[3] = i;
        if (b.name == "lq_feat")      
            output_indexes_[4] = i;
        if (b.name == "soft_one_hot") 
            output_indexes_[5] = i;
    }

    return 0;
}

void Encoder::Run(const std::vector<Tensor_t>& input_tensor, std::vector<Tensor_t>& output_tensor)
{
    ncnn::Extractor encoder_ex = encoder_net_.create_extractor();

    for(int i = 0; i != input_indexes_.size(); ++i)
    {
        encoder_ex.input(input_indexes_[i], input_tensor[i].data);
    }

    for(int i = 0; i != output_indexes_.size(); ++i)
    {
        ncnn::Mat out;
        encoder_ex.extract(output_indexes_[i], out);
        output_tensor.push_back(Tensor_t(out));
    }
}

void Encoder::PreProcess(const void* input_data, std::vector<Tensor_t>& input_tensor)
{
    ncnn::Mat in = ncnn::Mat::from_pixels(((cv::Mat*)input_data)->data, ncnn::Mat::PIXEL_BGR2RGB, ((cv::Mat*)input_data)->cols, ((cv::Mat*)input_data)->rows);
    in.substract_mean_normalize(mean_vals_, norm_vals_);

    input_tensor.push_back(Tensor_t(in));
}
void Encoder::PostProcess(const std::vector<Tensor_t>& input_tensor, std::vector<Tensor_t>& output_tensor, void* result)
{
    ncnn::Mat min_encodings;
    GetCodebookFeat(output_tensor[5].data, min_encodings);
    output_tensor.pop_back();

    output_tensor.push_back(Tensor_t(min_encodings));
    
    ((CodeFormerResult_t*)result)->output_tensors.resize(7);
    std::copy(output_tensor.begin(), output_tensor.end(), ((CodeFormerResult_t*)result)->output_tensors.begin());

}

int Encoder::Process(const cv::Mat& input_img, void* result)
{
    if(input_img.empty()){
        fprintf(stderr, "CodeFormer: input_image is empty!\n");
        return -1;
    }

    std::vector<Tensor_t> input_tensor;
    PreProcess((void*)&input_img, input_tensor);

    std::vector<Tensor_t> output_tensor;
    Run(input_tensor, output_tensor);

    PostProcess(input_tensor, output_tensor, result);

    return 0;
}
}