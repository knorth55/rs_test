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
#include <rs/DrawingAnnotator.h>
#include <rs/utils/common.h>
#include <rs/utils/time.h>


namespace python = boost::python;
namespace np = boost::numpy;

using namespace uima;


class FasterRCNNAnnotator : public DrawingAnnotator
{
private:
  struct ObjectBoundingBox
  {
    int label;
    cv::Rect bbox;
    float score;
  };

  std::vector<ObjectBoundingBox> bboxes;
  cv::Mat color;

  std::string structure;
  std::string pretrained_model;
  int gpu;
  float score_thresh;
  python::object predictor;
  python::tuple label_names;

public:

  FasterRCNNAnnotator(): DrawingAnnotator(__func__) {}

  TyErrorId initialize(AnnotatorContext &ctx)
  {
    outInfo("initialize");
    ctx.extractValue("structure", structure);
    ctx.extractValue("pretrained_model", pretrained_model);
    ctx.extractValue("gpu", gpu);
    ctx.extractValue("score_thresh", score_thresh);

    Py_Initialize();
    np::initialize();
    python::object rs_test_module = python::import("rs_test");
    predictor = rs_test_module.attr("FasterRCNNPredictor")
        (structure.c_str(), pretrained_model.c_str(), gpu, score_thresh);
    label_names = python::extract<python::tuple>(predictor.attr("label_names"));
    return UIMA_ERR_NONE;
  }

  TyErrorId destroy()
  {
    outInfo("destroy");
    return UIMA_ERR_NONE;
  }

private:
  TyErrorId processWithLock(CAS &tcas, ResultSpecification const &res_spec)
  {
    outInfo("process start");
    rs::StopWatch clock;
    rs::SceneCas cas(tcas);
    rs::Scene scene = cas.getScene();

    cas.get(VIEW_COLOR_IMAGE_HD, color);
    cv::Size color_size = color.size();
    int color_height = color_size.height;
    int color_width = color_size.width;

    python::tuple shape = python::make_tuple(color_height, color_width, 3);
    np::ndarray img = np::zeros(shape, np::dtype::get_builtin<uint8_t>());
    uint8_t* img_ptr = reinterpret_cast<uint8_t *>(img.get_data());
    for (int i=0; i < color_width * color_height * 3; i++)
    {
      *(img_ptr + i) = color.data[i];
    }

    outInfo("FasterRCNN prediction start");
    python::tuple return_tuple = python::extract<python::tuple>(
      predictor.attr("predict")(img));
    outInfo("FasterRCNN prediction finish");
    np::ndarray bbox = python::extract<np::ndarray>(return_tuple[0]);
    np::ndarray label = python::extract<np::ndarray>(return_tuple[1]);
    np::ndarray score = python::extract<np::ndarray>(return_tuple[2]);
    int n_rois = bbox.get_shape()[0];
    bboxes.resize(n_rois);

    int *bbox_ptr = reinterpret_cast<int *>(bbox.get_data());
    int *label_ptr = reinterpret_cast<int *>(label.get_data());
    float *score_ptr = reinterpret_cast<float *>(score.get_data());

    for (int i = 0; i < n_rois; i++) {
      // bbox
      int y_min = std::max(0, *(bbox_ptr + 4 * i));
      int x_min = std::max(0, *(bbox_ptr + 4 * i + 1));
      int y_max = std::min(color_height, *(bbox_ptr + 4 * i + 2));
      int x_max = std::min(color_width, *(bbox_ptr + 4 * i + 3));

      cv::Rect bbox_hires, bbox_lowres;
      bbox_hires = cv::Rect(x_min, y_min, x_max - x_min, y_max - y_min);
      bbox_lowres = cv::Rect(
        x_min / 2, y_min / 2, (x_max - x_min) / 2, (y_max - y_min) / 2);

      bboxes[i].bbox = bbox_hires;
      bboxes[i].label = *(label_ptr + i);
      bboxes[i].score = *(score_ptr + i);
      std::string label_name = python::extract<std::string>(label_names[bboxes[i].label]);
      outInfo("label " << i << "th: " << label_name);
      outInfo("score " << i << "th: " << bboxes[i].score);
      outInfo("bbox " << i << "th: ("
          << y_min << ", " << x_min << ", " << y_max << ", " << x_max << ")");

      // annotate
      rs::ImageROI roi = rs::create<rs::ImageROI>(tcas);
      roi.roi(rs::conversion::to(tcas, bbox_lowres));
      roi.roi_hires(rs::conversion::to(tcas, bbox_hires));
      rs::Cluster cluster = rs::create<rs::Cluster>(tcas);
      cluster.rois(roi);
      rs::Classification classification = rs::create<rs::Classification>(tcas);
      classification.classification_type.set("CLASS");
      classification.classname.set(label_name);
      classification.classifier.set("FasterRCNN");
      classification.source.set("FasterRCNNAnnotator");
      cluster.annotations.append(classification);
      scene.identifiables.append(cluster);
    }

    return UIMA_ERR_NONE;
  }

  void drawImageWithLock(cv::Mat &disp)
  {
    disp = color.clone();
    for(int i = 0; i < bboxes.size(); i++)
    {
      int label = bboxes[i].label;
      float score = bboxes[i].score;
      std::string label_name = python::extract<std::string>(label_names[label]);
      std::ostringstream ss;
      ss << "label: " << label_name << ", score: " << score;
      std::string bbox_text = ss.str();
      cv::Rect bbox = bboxes[i].bbox;
      cv::Scalar color = rs::common::cvScalarColors[i % rs::common::numberOfColors];
      cv::putText(disp, bbox_text, cv::Point(bbox.x, bbox.y), 0, 0.5, color);
      cv::rectangle(disp, bbox, color);
    }
  }
};

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(FasterRCNNAnnotator)
