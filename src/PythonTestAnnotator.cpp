#include <uima/api.hpp>
#include <boost/python.hpp>
#include <boost/numpy.hpp>

#include <pcl/point_types.h>
#include <rs/types/all_types.h>
//RS
#include <rs/scene_cas.h>
#include <rs/utils/time.h>

namespace python = boost::python;
namespace np = boost::numpy;

using namespace uima;


class PythonTestAnnotator : public Annotator
{
private:
  float test_param;

public:

  TyErrorId initialize(AnnotatorContext &ctx)
  {
    outInfo("initialize");
    Py_Initialize();
    np::initialize();
    ctx.extractValue("test_param", test_param);
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
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGBA>);
    outInfo("Test param =  " << test_param);
    cas.get(VIEW_CLOUD,*cloud_ptr);

    // default math package test
    outInfo("Cloud size: " << cloud_ptr->points.size());
    python::object math_module = python::import("math");
    python::object math_sqrt_func = math_module.attr("sqrt");
    float res = python::extract<float>(math_sqrt_func(cloud_ptr->points.size()));
    outInfo("Sqrt cloud size by Python: " << res);

    // rs_test ROS package test
    python::object rs_test_module = python::import("rs_test");
    python::object rs_test_random_add_func = rs_test_module.attr("random_add");
    float add_res = python::extract<int>(rs_test_random_add_func(cloud_ptr->points.size()));
    outInfo("Random add cloud size by python: " << add_res);

    // boost numpy test
    python::object rs_test_random_add_arr_func = rs_test_module.attr("random_add_array");
    python::tuple shape = python::make_tuple(cloud_ptr->points.size());
    np::ndarray arr = np::zeros(shape, np::dtype::get_builtin<float>());
    np::ndarray res_arr = python::extract<np::ndarray>(rs_test_random_add_arr_func(arr));
    Py_intptr_t const* res_arr_shape = res_arr.get_shape();
    outInfo("Numpy array dimension: " << res_arr.get_nd());
    outInfo("Numpy array shape: " << res_arr_shape[0]);
    outInfo("Random add array: " << python::extract<char const *>(python::str(res_arr)));

    // get pointer of numpy array
    double *p = reinterpret_cast<double *>(res_arr.get_data());
    for (size_t i = 0; i < 10; i++)
    {
      outInfo("Numpy array " << i << "th element: " << *(p + i));
    }

    outInfo("took: " << clock.getTime() << " ms.");
    return UIMA_ERR_NONE;
  }
};

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(PythonTestAnnotator)
