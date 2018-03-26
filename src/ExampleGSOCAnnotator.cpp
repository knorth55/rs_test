#include <uima/api.hpp>

#include <pcl/point_types.h>
#include <rs/types/all_types.h>
#include <rs_test/types/all_types.h>

//RS
#include <rs/scene_cas.h>
#include <rs/utils/time.h>

#include <pcl/common/centroid.h>

using namespace uima;


class ExampleGSOCAnnotator : public Annotator
{
private:

public:

  TyErrorId initialize(AnnotatorContext &ctx)
  {
    outInfo("initialize");
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
    rs::Scene scene = cas.getScene();
    cas.get(VIEW_CLOUD, *cloud_ptr);

    std::vector<rs::Cluster> clusters;
    scene.identifiables.filter(clusters);
    int idx = 0;

    for (auto cluster:clusters)
    {
        rs::Cluster &c = cluster;
        pcl::PointIndices indices;
        rs::conversion::from(((rs::ReferenceClusterPoints)c.points()).indices(), indices);

        Eigen::Vector4d pCentroid;
        pcl::compute3DCentroid(*cloud_ptr, indices, pCentroid);
        rs_test::ExampleAnnotation annotation = rs::create<rs_test::ExampleAnnotation>(tcas);
        rs_test::Centroid centroid = rs::create<rs_test::Centroid>(tcas);
        centroid.x.set(pCentroid[0]);
        centroid.y.set(pCentroid[1]);
        centroid.z.set(pCentroid[2]);
        annotation.centroid.set(centroid);
        annotation.clusterId.set(idx++);
        outInfo("ID: " <<annotation.clusterId());
        outInfo("    x =" <<annotation.centroid().x());
        outInfo("    y =" <<annotation.centroid().y());
        outInfo("    z =" <<annotation.centroid().z());
        c.annotations.append(annotation);
    }

    outInfo("took: " << clock.getTime() << " ms.");
    return UIMA_ERR_NONE;
  }
};

// This macro exports an entry point that is used to create the annotator.
MAKE_AE(ExampleGSOCAnnotator)
