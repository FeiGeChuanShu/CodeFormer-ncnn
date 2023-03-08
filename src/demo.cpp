#include <opencv2/opencv.hpp>
#include "include/pipeline.h"

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s [imagepath] [bg_enhance]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat img = cv::imread(imagepath, 1);
    if (img.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    wsdsb::PipelineConfig_t pipeline_config_t;
    pipeline_config_t.model_path = "../models/";
    pipeline_config_t.bg_upsample = atoi(argv[2]);

    cv::Mat bg_upsample;
    wsdsb::PipeLine pipe;
    pipe.CreatePipeLine(pipeline_config_t);

    pipe.Apply(img, bg_upsample);
    cv::imwrite("result.png",bg_upsample);
    
    return 0;
}
