#include<iostream>
#include<string>
using namespace std;

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
using namespace cv;

#include<pcl/io/pcd_io.h>
#include<pcl/point_types.h>
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

const double camera_factor = 1000;
const double camera_cx = 325.5;
const double camera_cy = 253.5;
const double camera_fx = 518.0;
const double camera_fy = 519.0;

int main(int argc,char **argv)
{
  Mat rgb,depth;
  rgb=imread("./data/rgb.png");
  depth=imread("./data/depth.png");
//使用智能指针，创建一个空点云，这种指针用完会自动释放
  PointCloud::Ptr cloud(new PointCloud);
  //遍历深度图
  for(int m=0;m<depth.rows;m++)
  {
    for(int n=0;n<depth.cols;n++)
    {
      //获取深度图中（m,n）的值
      ushort d=depth.ptr<ushort>(m)[n];
      //d可能没有值，如若此，则跳过此点
      if(d==0)
	continue;
      PointT p;
      p.z=double(d)/camera_factor;
      p.x=(n-camera_cx)*p.z / camera_fx;
      p.y=(m-camera_cy)*p.z /camera_fy;
      
      //从RGB获取他的颜色
      p.b=rgb.ptr<uchar>(m)[n*3];
      p.g=rgb.ptr<uchar>(m)[n*3+1];
      p.r=rgb.ptr<uchar>(m)[n*3+2];
      
      //把p加入到点云
      cloud->points.push_back(p);
    }
  }
  
  //设置并保存点云
  cloud->height=1;
  cloud->width=cloud->points.size();
  cout<<"point cloud size = "<<cloud->points.size()<<endl;
  cloud->is_dense=false;
  pcl::io::savePCDFile("./pointcloud.pcd",*cloud);
  //清除数据
  cloud->points.clear();
  cout<<"Point cloud saved."<<endl;
  return 0;
}
