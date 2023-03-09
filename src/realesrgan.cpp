#include "include/realesrgan.h"
namespace wsdsb{ 
RealESRGAN::RealESRGAN():scale(2),tile_size(400),tile_pad(10)
{

}

RealESRGAN::~RealESRGAN()
{
    net_.clear();
}

int RealESRGAN::Load(const std::string& model_path)
{
    std::string net_param_path = model_path + "/real_esrgan.param";
    std::string net_model_path = model_path + "/real_esrgan.bin";

    int ret = net_.load_param(net_param_path.c_str());
    if (ret < 0)
    {
        fprintf(stderr, "open param file %s failed\n", net_param_path.c_str());
        return -1;
    }
    ret = net_.load_model(net_model_path.c_str());
    if (ret < 0)
    {
        fprintf(stderr, "open bin file %s failed\n", net_model_path.c_str());
        return -1;
    }

    for(const auto& input : net_.input_indexes())
    {
        input_indexes_.push_back(input);
    }
    for(const auto& output : net_.output_indexes())
    {
        output_indexes_.push_back(output);
    }
    return 0;
}
void RealESRGAN::Tensor2Image(const ncnn::Mat& output_tensor, int img_h, int img_w, cv::Mat& output_img)
{
    std::vector<cv::Mat> mat_list;
    for (int i = 0; i != 3; ++i) 
    {
        cv::Mat mat = cv::Mat(img_h, img_w, CV_32FC1, (void*)((float*)output_tensor.data + i * img_w * img_h));
        mat_list.push_back(mat);
    }

    cv::Mat result_img_f;
    cv::merge(mat_list, result_img_f);

    cv::Mat result_img;
    result_img_f.convertTo(result_img, CV_8UC3, 255.0, 0);

    cv::cvtColor(result_img, output_img, cv::COLOR_RGB2BGR);
}

int RealESRGAN::Padding(const cv::Mat& img, cv::Mat& pad_img,int& img_pad_h,int& img_pad_w)
{
    if (img.cols % 2 != 0)
    {
        img_pad_w = (2 - img.cols % 2);
    }
    if (img.rows % 2 != 0)
    {
        img_pad_h = (2 - img.rows % 2);
    }
    cv::copyMakeBorder(img, pad_img, 0, img_pad_h, 0, img_pad_w, cv::BORDER_CONSTANT,cv::Scalar(0));

    return 0;
}

int RealESRGAN::Process(const cv::Mat& input_img, void* result)
{
    if(input_img.empty()){
        fprintf(stderr, "RealESRGAN: input_img is empty\n");
        return -1;
    }
    cv::Mat pad_inimage;
    int img_pad_w = 0, img_pad_h = 0;
    Padding(input_img, pad_inimage, img_pad_w, img_pad_h);
    
    int tiles_x = std::ceil((float)input_img.cols / tile_size);
    int tiles_y = std::ceil((float)input_img.rows / tile_size);

    cv::Mat out = cv::Mat(cv::Size(pad_inimage.cols * 2, pad_inimage.rows * 2), CV_8UC3);
    for (int i = 0; i < tiles_y; i++)
    {
        for (int j = 0; j < tiles_x; j++)
        {
            int ofs_x = j * tile_size;
            int ofs_y = i * tile_size;

            int input_start_x = ofs_x;
            int input_end_x = std::min(ofs_x + tile_size, pad_inimage.cols);
            int input_start_y = ofs_y;
            int input_end_y = std::min(ofs_y + tile_size, pad_inimage.rows);

            int input_start_x_pad = std::max(input_start_x - tile_pad, 0);
            int input_end_x_pad = std::min(input_end_x + tile_pad, pad_inimage.cols);
            int input_start_y_pad = std::max(input_start_y - tile_pad, 0);
            int input_end_y_pad = std::min(input_end_y + tile_pad, pad_inimage.rows);

            int input_tile_width = input_end_x - input_start_x;
            int input_tile_height = input_end_y - input_start_y;

            cv::Mat input_tile = pad_inimage(cv::Rect(input_start_x_pad, input_start_y_pad, input_end_x_pad- input_start_x_pad, input_end_y_pad- input_start_y_pad)).clone();
            
            std::vector<Tensor_t> input_tensor;
            PreProcess((void*)&input_tile, input_tensor);

            std::vector<Tensor_t> output_tensor;
            Run(input_tensor, output_tensor);

            cv::Mat out_tile;
            PostProcess(input_tensor, output_tensor, (void*)&out_tile);
            
            int output_start_x = input_start_x * scale;
            int output_end_x = input_end_x * scale;
            int output_start_y = input_start_y * scale;
            int output_end_y = input_end_y * scale;

            int output_start_x_tile = (input_start_x - input_start_x_pad) * scale;
            int output_end_x_tile = output_start_x_tile + input_tile_width * scale;
            int output_start_y_tile = (input_start_y - input_start_y_pad) * scale;
            int output_end_y_tile = output_start_y_tile + input_tile_height * scale;
            cv::Rect tile_roi = cv::Rect(output_start_x_tile, output_start_y_tile,
                output_end_x_tile - output_start_x_tile,
                output_end_y_tile - output_start_y_tile);
            cv::Rect out_roi = cv::Rect(output_start_x, output_start_y,
                output_end_x - output_start_x, output_end_y - output_start_y);
            out_tile(tile_roi).copyTo(out(out_roi));
        }
    }

    out.copyTo(*((cv::Mat*)result));
    
    return 0;
}
void RealESRGAN::Run(const std::vector<Tensor_t>& input_tensor, std::vector<Tensor_t>& output_tensor)
{
    ncnn::Extractor net_ex = net_.create_extractor();

    for(int i = 0; i != input_indexes_.size(); ++i)
    {
        net_ex.input(input_indexes_[i], input_tensor[i].data);
    }

    for(int i = 0; i != output_indexes_.size(); ++i)
    {
        ncnn::Mat out;
        net_ex.extract(output_indexes_[i], out);
        output_tensor.push_back(Tensor_t(out));
    }
}
void RealESRGAN::PreProcess(const void* input_data, std::vector<Tensor_t>& input_tensor)
{
    ncnn::Mat in = ncnn::Mat::from_pixels(((cv::Mat*)input_data)->data, ncnn::Mat::PIXEL_BGR2RGB, ((cv::Mat*)input_data)->cols, ((cv::Mat*)input_data)->rows);
    in.substract_mean_normalize(0, norm_vals_);
    input_tensor.push_back(in);
}
void RealESRGAN::PostProcess(const std::vector<Tensor_t>& input_tensor, std::vector<Tensor_t>& output_tensor, void* result)
{
    ncnn::Mat& out = output_tensor[0].data;

    cv::Mat out_img;
    Tensor2Image(out, out.h, out.w, out_img);

    out_img.copyTo(*(cv::Mat*)result);
}
}