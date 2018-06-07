#include <vector>

#include <uima/api.hpp>

// python
#include <boost/python.hpp>
#include <boost/numpy.hpp>

// opencv
#include <opencv2/opencv.hpp>

#include <pcl/point_types.h>
#include <rs/types/all_types.h>
//RS
#include <rs/scene_cas.h>
#include <rs/utils/common.h>
#include <rs/utils/time.h>


namespace python = boost::python;
namespace np = boost::numpy;

using namespace uima;


class FasterRCNNAnnotator : public Annotator
{
private:
  struct ObjectBoundingBox
  {
    int label;
    cv::Rect bbox, bboxHires;
    float score;
  };

  std::vector<ObjectBoundingBox> bboxes;
  cv::Mat color;
  python::object predictor;

public:

  TyErrorId initialize(AnnotatorContext &ctx)
  {
    outInfo("initialize");
    Py_Initialize();
    np::initialize();
    python::object rs_test_module = python::import("rs_test");
    predictor = rs_test_module.attr("FasterRCNNVGG16Predictor");
    python::object predictor_init_func = predictor.attr("__init__");
    predictor_init_func(-1);
    return UIMA_ERR_NONE;
  }

  TyErrorId destroy()
  {
    outInfo("destroy");
    return UIMA_ERR_NONE;
  }

  TyErrorId process(CAS &tcas, ResultSpecification const &res_spec)
  {
    outInfo("process start");
    rs::StopWatch clock;
    rs::SceneCas cas(tcas);
    cas.get(VIEW_COLOR_IMAGE_HD, color);
    cv::Size color_size = color.size();
    int color_height = color_size.height;
    int color_width = color_size.width;

    python::object predict_func = predictor.attr("predict");
    python::tuple shape = python::make_tuple(color_height, color_width, 3);
    np::ndarray img = np::zeros(shape, np::dtype::get_builtin<uint>());
    uint* img_ptr = reinterpret_cast<uint *>(img.get_data());
    for (size_t i=0; i < color_width * color_height * 3; i++)
    {
      *(img_ptr + i) = color.data[i];
    }
    python::tuple return_tuple = python::extract<python::tuple>(predict_func(img));
    np::ndarray bbox = python::extract<np::ndarray>(return_tuple[0]);
    np::ndarray label = python::extract<np::ndarray>(return_tuple[1]);
    np::ndarray score = python::extract<np::ndarray>(return_tuple[2]);
    int n_rois = bbox.get_shape()[0];
    bboxes.resize(n_rois);

    float *bbox_ptr = reinterpret_cast<float *>(bbox.get_data());
    int *label_ptr = reinterpret_cast<int *>(label.get_data());
    float *score_ptr = reinterpret_cast<float *>(score.get_data());

    for (size_t i=0; i < n_rois; i++) {
      int y_min = (int)std::round(std::min((float)0, *(bbox_ptr + 4 * i)));
      int x_min = (int)std::round(std::min((float)0, *(bbox_ptr + 4 * i + 1)));
      int y_max = (int)std::round(std::max((float)color_height, *(bbox_ptr + 4 * i + 2)));
      int x_max = (int)std::round(std::max((float)color_width, *(bbox_ptr + 4 * i + 3)));
      bboxes[i].bbox = cv::Rect(
        x_min, y_min, x_max - x_min + 1, y_max - y_min + 1);
      bboxes[i].bboxHires = cv::Rect(
        2 * x_min, 2 * y_min,
        2 * (x_max - x_min + 1), 2 * (y_max - y_min + 1));
      bboxes[i].label = *(label_ptr + i);
      bboxes[i].score = *(score_ptr + i);
    }

    return UIMA_ERR_NONE;
  }

  void drawImage(cv::Mat &disp)
  {
    disp = color.clone();
    for(size_t i = 0; i < bboxes.size(); ++i)
    {
      cv::rectangle(disp, bboxes[i].bboxHires, rs::common::cvScalarColors[i % rs::common::numberOfColors]);
    }
  }
};

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(FasterRCNNAnnotator)